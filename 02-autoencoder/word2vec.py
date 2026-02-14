"""Word2Vec models: Skip-Gram and CBOW with negative sampling.

Word2Vec learns dense vector representations (embeddings) for words such that
words appearing in similar contexts end up close together in the embedding space.

Two architectures:

1. **Skip-Gram**: Given a center word, predict surrounding context words.
   Objective: maximize P(context | center) for observed (center, context) pairs.

2. **CBOW** (Continuous Bag of Words): Given surrounding context words, predict
   the center word.  Objective: maximize P(center | context).

Both use **negative sampling** instead of a full softmax over the vocabulary,
which would be prohibitively expensive for large vocabularies.  Negative sampling
approximates the softmax by contrasting true (center, context) pairs against
randomly sampled "negative" pairs.

Negative sampling loss for a positive pair (w, c) with negatives {n_1, ..., n_k}:

    L = -log(sigma(v_c . v_w)) - sum_i log(sigma(-v_{n_i} . v_w))

where sigma is the sigmoid function and v_x denotes the embedding of word x.

Example:
    >>> vocab_size, embed_dim = 5000, 100
    >>> skipgram = SkipGram(vocab_size, embed_dim)
    >>> center = torch.tensor([1, 42, 7])
    >>> context = torch.tensor([2, 43, 8])
    >>> negatives = torch.randint(0, vocab_size, (3, 5))
    >>> loss = skipgram(center, context, negatives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    """Skip-Gram Word2Vec model with negative sampling.

    Architecture:
        center word index --> Embedding(vocab, dim) --> center vector
        context word index --> Embedding(vocab, dim) --> context vector
        negative word indices --> Embedding(vocab, dim) --> negative vectors

    The model learns two embedding matrices:
      - `center_embeddings`: lookup for center words
      - `context_embeddings`: lookup for context/negative words

    The loss encourages high dot-product between true (center, context) pairs
    and low dot-product between (center, negative) pairs.

    Args:
        vocab_size: Number of words in the vocabulary.
        embed_dim: Dimensionality of the word embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Initialize with small random values (uniform in [-0.5/dim, 0.5/dim])
        init_range = 0.5 / embed_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(
        self,
        center: torch.Tensor,
        context: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative sampling loss for skip-gram.

        Args:
            center: Center word indices, shape (batch_size,).
            context: Positive context word indices, shape (batch_size,).
            negatives: Negative sample indices, shape (batch_size, num_negatives).

        Returns:
            Scalar loss averaged over the batch.
        """
        # Lookup embeddings
        center_emb = self.center_embeddings(center)  # (B, D)
        context_emb = self.context_embeddings(context)  # (B, D)
        neg_emb = self.context_embeddings(negatives)  # (B, K, D)

        # Positive score: dot product between center and true context
        # (B, D) * (B, D) -> (B,) after sum
        pos_score = (center_emb * context_emb).sum(dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative scores: dot product between center and each negative sample
        # (B, 1, D) bmm (B, D, K) -> (B, 1, K) -> (B, K)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        # Maximize pos_loss + neg_loss => minimize -(pos_loss + neg_loss)
        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self) -> torch.Tensor:
        """Return the learned center word embeddings.

        Returns:
            Embedding weight matrix of shape (vocab_size, embed_dim).
        """
        return self.center_embeddings.weight.data


class CBOW(nn.Module):
    """Continuous Bag of Words (CBOW) model with negative sampling.

    Architecture:
        context word indices --> Embedding(vocab, dim) --> mean context vector
        center word index --> Embedding(vocab, dim) --> center vector
        negative word indices --> Embedding(vocab, dim) --> negative vectors

    Unlike Skip-Gram, CBOW averages the embeddings of all context words to
    produce a single context representation, then predicts the center word.

    Args:
        vocab_size: Number of words in the vocabulary.
        embed_dim: Dimensionality of the word embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)

        init_range = 0.5 / embed_dim
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(
        self,
        context: torch.Tensor,
        center: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative sampling loss for CBOW.

        Args:
            context: Context word indices, shape (batch_size, 2 * window_size).
            center: Center word indices, shape (batch_size,).
            negatives: Negative sample indices, shape (batch_size, num_negatives).

        Returns:
            Scalar loss averaged over the batch.
        """
        # Average the context word embeddings
        ctx_emb = self.context_embeddings(context)  # (B, 2*W, D)
        ctx_mean = ctx_emb.mean(dim=1)  # (B, D)

        # Center word embedding (used as target)
        center_emb = self.center_embeddings(center)  # (B, D)
        neg_emb = self.center_embeddings(negatives)  # (B, K, D)

        # Positive score: dot product of context mean and center
        pos_score = (ctx_mean * center_emb).sum(dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative scores: dot product of context mean and each negative
        neg_score = torch.bmm(neg_emb, ctx_mean.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self) -> torch.Tensor:
        """Return the learned context word embeddings.

        Returns:
            Embedding weight matrix of shape (vocab_size, embed_dim).
        """
        return self.context_embeddings.weight.data
