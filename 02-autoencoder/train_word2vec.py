"""Train Word2Vec (Skip-Gram and CBOW) on a small corpus.

Demonstrates training word embeddings on a small hardcoded corpus of sentences
about nature and animals.  After training, prints the nearest neighbors for
a few example words using cosine similarity.

Usage:
    python train_word2vec.py
"""

import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import shared utilities from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from word2vec import CBOW, SkipGram

from utils import Timer, get_device, set_seed

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

CORPUS = """
The quick brown fox jumps over the lazy dog near the river.
A small cat sits on the warm stone by the river bank.
The dog and the cat are friends who play in the garden.
Birds sing in the tall trees near the old stone wall.
The river flows through the green valley below the mountain.
Fish swim in the clear water of the river near the forest.
The forest is full of tall trees and wild animals.
A brown bear walks through the forest looking for food.
The fox hides in the bushes near the edge of the forest.
Rabbits play in the green grass near the garden wall.
The mountain rises above the clouds in the morning light.
Eagles fly high above the mountain and the valley below.
The wind blows through the trees making the leaves dance.
Rain falls on the forest and fills the river with water.
The sun shines on the garden where flowers grow and bloom.
Small birds build nests in the branches of the old tree.
The cat watches the birds from the window of the house.
A deer runs through the forest past the river and the stones.
The garden is full of colorful flowers and green plants.
The old tree stands alone in the middle of the green field.
Wolves howl at the moon from the top of the mountain.
The bear catches fish in the shallow water of the river.
Butterflies dance among the flowers in the warm garden.
The dog chases the fox through the tall grass near the wall.
Stars shine bright in the clear sky above the quiet valley.
"""

# ---------------------------------------------------------------------------
# Vocabulary and data preparation
# ---------------------------------------------------------------------------

MIN_WORD_FREQ = 2  # Discard very rare words
WINDOW_SIZE = 3
NUM_NEGATIVES = 5


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into words, stripping punctuation."""
    cleaned = []
    for word in text.lower().split():
        stripped = word.strip(".,;:!?\"'()[]")
        if stripped:
            cleaned.append(stripped)
    return cleaned


def build_vocab(
    tokens: list[str], min_freq: int = 2
) -> tuple[dict[str, int], dict[int, str], list[int]]:
    """Build vocabulary from tokens, filtering by minimum frequency.

    Returns:
        word2idx: Mapping from word to index.
        idx2word: Mapping from index to word.
        filtered_indices: Token indices for tokens that passed the frequency filter.
    """
    freq = Counter(tokens)
    # Keep only words that appear at least min_freq times
    vocab_words = sorted(w for w, c in freq.items() if c >= min_freq)

    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # Convert corpus to indices, skipping out-of-vocabulary words
    filtered_indices = [word2idx[t] for t in tokens if t in word2idx]

    return word2idx, idx2word, filtered_indices


def build_skipgram_pairs(indices: list[int], window_size: int) -> list[tuple[int, int]]:
    """Generate (center, context) pairs for skip-gram training."""
    pairs = []
    for i, center in enumerate(indices):
        start = max(0, i - window_size)
        end = min(len(indices), i + window_size + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, indices[j]))
    return pairs


def build_cbow_pairs(indices: list[int], window_size: int) -> list[tuple[list[int], int]]:
    """Generate (context_words, center) pairs for CBOW training.

    Each context is a fixed-size list of 2*window_size words.  Pairs where the
    full window does not fit (near corpus boundaries) are skipped.
    """
    pairs = []
    for i in range(window_size, len(indices) - window_size):
        context = indices[i - window_size : i] + indices[i + 1 : i + window_size + 1]
        pairs.append((context, indices[i]))
    return pairs


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SkipGramDataset(Dataset):
    """PyTorch dataset for skip-gram (center, context) pairs."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        vocab_size: int,
        num_negatives: int,
    ) -> None:
        self.pairs = pairs
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center, context = self.pairs[idx]
        negatives = torch.randint(0, self.vocab_size, (self.num_negatives,))
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            negatives,
        )


class CBOWDataset(Dataset):
    """PyTorch dataset for CBOW (context, center) pairs."""

    def __init__(
        self,
        pairs: list[tuple[list[int], int]],
        vocab_size: int,
        num_negatives: int,
    ) -> None:
        self.pairs = pairs
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, center = self.pairs[idx]
        negatives = torch.randint(0, self.vocab_size, (self.num_negatives,))
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(center, dtype=torch.long),
            negatives,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_skipgram(
    model: SkipGram,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> None:
    """Train the skip-gram model."""
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for center, context, negatives in loader:
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            loss = model(center, context, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Skip-Gram] Epoch {epoch:>3}/{num_epochs} | Loss: {avg_loss:.4f}")


def train_cbow(
    model: CBOW,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> None:
    """Train the CBOW model."""
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for context, center, negatives in loader:
            context = context.to(device)
            center = center.to(device)
            negatives = negatives.to(device)

            loss = model(context, center, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [CBOW]      Epoch {epoch:>3}/{num_epochs} | Loss: {avg_loss:.4f}")


# ---------------------------------------------------------------------------
# Nearest neighbors
# ---------------------------------------------------------------------------


def find_nearest(
    word: str,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    embeddings: torch.Tensor,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find the top-k nearest neighbors of a word by cosine similarity."""
    if word not in word2idx:
        return []

    idx = word2idx[word]
    query = embeddings[idx].unsqueeze(0)  # (1, D)

    # Cosine similarity against all embeddings
    cos_sim = F.cosine_similarity(query, embeddings, dim=1)

    # Exclude the query word itself, then take top-k
    cos_sim[idx] = -1.0
    top_indices = cos_sim.argsort(descending=True)[:top_k]

    return [(idx2word[i.item()], cos_sim[i].item()) for i in top_indices]


def show_nearest_neighbors(
    model_name: str,
    embeddings: torch.Tensor,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    query_words: list[str],
) -> None:
    """Print nearest neighbors for a list of query words."""
    print(f"\n--- {model_name} Nearest Neighbors ---")
    for word in query_words:
        neighbors = find_nearest(word, word2idx, idx2word, embeddings, top_k=5)
        if neighbors:
            neighbor_str = ", ".join(f"{w} ({sim:.3f})" for w, sim in neighbors)
            print(f"  {word:>12s} -> {neighbor_str}")
        else:
            print(f"  {word:>12s} -> (not in vocabulary)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EMBED_DIM = 50
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.01


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Tokenize and build vocabulary
    tokens = tokenize(CORPUS)
    word2idx, idx2word, filtered_indices = build_vocab(tokens, min_freq=MIN_WORD_FREQ)
    vocab_size = len(word2idx)
    print(f"Corpus: {len(tokens)} tokens, Vocabulary: {vocab_size} words")

    # Build training pairs
    sg_pairs = build_skipgram_pairs(filtered_indices, WINDOW_SIZE)
    cbow_pairs = build_cbow_pairs(filtered_indices, WINDOW_SIZE)
    print(f"Skip-gram pairs: {len(sg_pairs)}, CBOW pairs: {len(cbow_pairs)}")

    # ----- Train Skip-Gram -----
    print("\nTraining Skip-Gram...")
    sg_dataset = SkipGramDataset(sg_pairs, vocab_size, NUM_NEGATIVES)
    sg_loader = DataLoader(sg_dataset, batch_size=BATCH_SIZE, shuffle=True)

    sg_model = SkipGram(vocab_size, EMBED_DIM).to(device)
    sg_optimizer = torch.optim.Adam(sg_model.parameters(), lr=LEARNING_RATE)

    with Timer() as sg_timer:
        train_skipgram(sg_model, sg_loader, sg_optimizer, device, NUM_EPOCHS)
    print(f"Skip-Gram training completed in {sg_timer}")

    # ----- Train CBOW -----
    print("\nTraining CBOW...")
    cbow_dataset = CBOWDataset(cbow_pairs, vocab_size, NUM_NEGATIVES)
    cbow_loader = DataLoader(cbow_dataset, batch_size=BATCH_SIZE, shuffle=True)

    cbow_model = CBOW(vocab_size, EMBED_DIM).to(device)
    cbow_optimizer = torch.optim.Adam(cbow_model.parameters(), lr=LEARNING_RATE)

    with Timer() as cbow_timer:
        train_cbow(cbow_model, cbow_loader, cbow_optimizer, device, NUM_EPOCHS)
    print(f"CBOW training completed in {cbow_timer}")

    # ----- Show nearest neighbors -----
    query_words = ["river", "forest", "cat", "dog", "mountain", "garden", "tree"]

    sg_embeddings = sg_model.get_embeddings().cpu()
    show_nearest_neighbors("Skip-Gram", sg_embeddings, word2idx, idx2word, query_words)

    cbow_embeddings = cbow_model.get_embeddings().cpu()
    show_nearest_neighbors("CBOW", cbow_embeddings, word2idx, idx2word, query_words)


if __name__ == "__main__":
    main()
