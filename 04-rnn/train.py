"""Train four RNN variants on a synthetic sentiment classification task.

Generates a small corpus of positive / negative sentences, builds a vocabulary,
trains SimpleRNN, GRU, LSTM, and BiLSTM classifiers, and saves a comparison
bar chart of their final test accuracies to ``model_comparison.png``.
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Import shared utilities from the project root
# ---------------------------------------------------------------------------
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import BiLSTMClassifier, GRUClassifier, LSTMClassifier, SimpleRNN  # noqa: E402, I001
from utils import Timer, get_device, set_seed  # noqa: E402, I001

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.0
NUM_SAMPLES = 2000  # total synthetic samples (50/50 split pos/neg)

# ---------------------------------------------------------------------------
# Synthetic sentiment dataset
# ---------------------------------------------------------------------------
POSITIVE_WORDS = [
    "amazing",
    "awesome",
    "beautiful",
    "brilliant",
    "excellent",
    "fantastic",
    "good",
    "great",
    "happy",
    "incredible",
    "love",
    "nice",
    "outstanding",
    "perfect",
    "pleasant",
    "superb",
    "terrific",
    "wonderful",
]

NEGATIVE_WORDS = [
    "awful",
    "bad",
    "boring",
    "disappointing",
    "dreadful",
    "hate",
    "horrible",
    "lousy",
    "mediocre",
    "miserable",
    "poor",
    "sad",
    "terrible",
    "ugly",
    "unpleasant",
    "weak",
    "worse",
    "worst",
]

NEUTRAL_WORDS = [
    "a",
    "an",
    "and",
    "but",
    "film",
    "food",
    "I",
    "is",
    "it",
    "movie",
    "product",
    "really",
    "so",
    "the",
    "thing",
    "this",
    "very",
    "was",
]

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _generate_sentence(sentiment_words: list[str], rng: random.Random) -> list[str]:
    """Build a random short sentence mixing sentiment and neutral words."""
    length = rng.randint(4, 10)
    num_sentiment = rng.randint(1, min(3, length))
    num_neutral = length - num_sentiment

    tokens = rng.choices(sentiment_words, k=num_sentiment) + rng.choices(
        NEUTRAL_WORDS, k=num_neutral
    )
    rng.shuffle(tokens)
    return tokens


def generate_dataset(
    num_samples: int,
    seed: int = 42,
) -> tuple[list[list[str]], list[int]]:
    """Create a synthetic sentiment corpus.

    Returns:
        sentences: List of tokenised sentences (list of word strings).
        labels:    0 = negative, 1 = positive.
    """
    rng = random.Random(seed)
    sentences: list[list[str]] = []
    labels: list[int] = []

    half = num_samples // 2
    for _ in range(half):
        sentences.append(_generate_sentence(POSITIVE_WORDS, rng))
        labels.append(1)
    for _ in range(half):
        sentences.append(_generate_sentence(NEGATIVE_WORDS, rng))
        labels.append(0)

    # Shuffle together
    combined = list(zip(sentences, labels, strict=True))
    rng.shuffle(combined)
    sentences, labels = zip(*combined, strict=True)  # type: ignore[assignment]
    return list(sentences), list(labels)


class Vocabulary:
    """Simple word-to-index mapping with special tokens."""

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word: dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, sentences: list[list[str]]) -> "Vocabulary":
        """Populate the vocabulary from a corpus of tokenised sentences."""
        for sent in sentences:
            for word in sent:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        return self

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(t, unk) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)


class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenised sentences and integer labels."""

    def __init__(
        self,
        sentences: list[list[str]],
        labels: list[int],
        vocab: Vocabulary,
    ) -> None:
        self.encoded = [torch.tensor(vocab.encode(s), dtype=torch.long) for s in sentences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoded[idx], self.labels[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences in a batch and return lengths.

    Returns:
        padded: (B, max_len) padded token-index tensor.
        labels: (B,) label tensor.
        lengths: (B,) original lengths.
    """
    sequences, labels = zip(*batch, strict=True)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, labels, lengths


# ---------------------------------------------------------------------------
# Training and testing helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch; return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for padded, labels, lengths in loader:
        padded = padded.to(device)
        labels = labels.to(device)

        logits = model(padded, lengths)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * padded.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += padded.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def test_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Test the model; return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for padded, labels, lengths in loader:
        padded = padded.to(device)
        labels = labels.to(device)

        logits = model(padded, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * padded.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += padded.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def save_comparison_chart(
    names: list[str],
    accuracies: list[float],
    path: str = "model_comparison.png",
) -> None:
    """Save a bar chart comparing final test accuracies of all models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    bars = ax.bar(names, [a * 100 for a in accuracies], color=colors, width=0.5)

    for bar, acc in zip(bars, accuracies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc * 100:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("RNN Variant Comparison -- Synthetic Sentiment Classification")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nComparison chart saved to {path}")


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}\n")

    # ---- Data preparation ------------------------------------------------
    sentences, labels = generate_dataset(NUM_SAMPLES, seed=SEED)
    vocab = Vocabulary().build(sentences)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total samples:   {NUM_SAMPLES}\n")

    # Train / test split (80 / 20)
    split = int(0.8 * len(sentences))
    train_ds = SentimentDataset(sentences[:split], labels[:split], vocab)
    test_ds = SentimentDataset(sentences[split:], labels[split:], vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ---- Models to compare -----------------------------------------------
    model_specs: list[tuple[str, nn.Module]] = [
        (
            "SimpleRNN",
            SimpleRNN(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=2,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            ),
        ),
        (
            "GRU",
            GRUClassifier(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=2,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            ),
        ),
        (
            "LSTM",
            LSTMClassifier(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=2,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            ),
        ),
        (
            "BiLSTM",
            BiLSTMClassifier(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=2,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            ),
        ),
    ]

    criterion = nn.CrossEntropyLoss()
    model_names: list[str] = []
    final_accuracies: list[float] = []

    for name, model in model_specs:
        set_seed(SEED)  # reset seed so each model sees the same batches
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print("=" * 60)
        print(f"Training {name}  ({total_params:,} parameters)")
        print("=" * 60)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        with Timer() as timer:
            for epoch in range(1, EPOCHS + 1):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                test_loss, test_acc = test_model(model, test_loader, criterion, device)
                print(
                    f"  Epoch {epoch:>2}/{EPOCHS}  "
                    f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                    f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}"
                )

        print(f"  Completed in {timer}\n")

        # Final test
        _, test_acc = test_model(model, test_loader, criterion, device)
        model_names.append(name)
        final_accuracies.append(test_acc)

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Final Test Accuracies")
    print("=" * 60)
    for name, acc in zip(model_names, final_accuracies, strict=True):
        print(f"  {name:<12s}  {acc * 100:.1f}%")

    save_comparison_chart(model_names, final_accuracies)


if __name__ == "__main__":
    main()
