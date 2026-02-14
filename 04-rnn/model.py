"""Recurrent neural network models for sequence classification.

Four variants are provided, all following the same embedding -> recurrent -> fc pattern:
  - SimpleRNN:      Vanilla RNN (Elman network)
  - GRUClassifier:  Gated Recurrent Unit
  - LSTMClassifier: Long Short-Term Memory
  - BiLSTMClassifier: Bidirectional LSTM
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleRNN(nn.Module):
    """Vanilla RNN for sequence classification.

    Architecture: Embedding -> nn.RNN -> Linear
    Uses the last hidden state as the sequence representation.

    Args:
        vocab_size:  Size of the token vocabulary.
        embed_dim:   Dimensionality of token embeddings.
        hidden_dim:  Number of features in the RNN hidden state.
        output_dim:  Number of output classes.
        num_layers:  Number of stacked RNN layers.
        dropout:     Dropout between RNN layers (applied when num_layers > 1).
        pad_idx:     Padding token index (embeddings are zeroed out).

    Example:
        >>> model = SimpleRNN(vocab_size=5000, embed_dim=64, hidden_dim=128, output_dim=2)
        >>> tokens = torch.randint(0, 5000, (32, 20))  # (batch, seq_len)
        >>> logits = model(tokens)                       # (32, 2)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:       Token indices of shape (batch, seq_len).
            lengths: Actual (unpadded) lengths per sample for packing.
                     If *None*, no packing is performed.

        Returns:
            Logits of shape (batch, output_dim).
        """
        embedded = self.embedding(x)  # (B, T, E)

        if lengths is not None:
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.rnn(packed)
        else:
            _, hidden = self.rnn(embedded)

        # hidden: (num_layers, B, H) -> take the last layer
        hidden = hidden[-1]  # (B, H)
        return self.fc(self.dropout(hidden))


class GRUClassifier(nn.Module):
    """GRU-based sequence classifier.

    Architecture: Embedding -> nn.GRU -> Linear
    Uses the last hidden state of the final GRU layer.

    Args:
        vocab_size:  Size of the token vocabulary.
        embed_dim:   Dimensionality of token embeddings.
        hidden_dim:  Number of features in the GRU hidden state.
        output_dim:  Number of output classes.
        num_layers:  Number of stacked GRU layers.
        dropout:     Dropout between GRU layers (applied when num_layers > 1).
        pad_idx:     Padding token index.

    Example:
        >>> model = GRUClassifier(vocab_size=5000, embed_dim=64, hidden_dim=128, output_dim=2)
        >>> logits = model(torch.randint(0, 5000, (32, 20)))  # (32, 2)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:       Token indices of shape (batch, seq_len).
            lengths: Actual (unpadded) lengths per sample for packing.

        Returns:
            Logits of shape (batch, output_dim).
        """
        embedded = self.embedding(x)

        if lengths is not None:
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(embedded)

        hidden = hidden[-1]  # (B, H)
        return self.fc(self.dropout(hidden))


class LSTMClassifier(nn.Module):
    """LSTM-based sequence classifier.

    Architecture: Embedding -> nn.LSTM -> Linear
    Uses the last hidden state *h_n* (not the cell state) of the final layer.

    Args:
        vocab_size:  Size of the token vocabulary.
        embed_dim:   Dimensionality of token embeddings.
        hidden_dim:  Number of features in the LSTM hidden state.
        output_dim:  Number of output classes.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout between LSTM layers (applied when num_layers > 1).
        pad_idx:     Padding token index.

    Example:
        >>> model = LSTMClassifier(vocab_size=5000, embed_dim=64, hidden_dim=128, output_dim=2)
        >>> logits = model(torch.randint(0, 5000, (32, 20)))  # (32, 2)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:       Token indices of shape (batch, seq_len).
            lengths: Actual (unpadded) lengths per sample for packing.

        Returns:
            Logits of shape (batch, output_dim).
        """
        embedded = self.embedding(x)

        if lengths is not None:
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _cell) = self.lstm(packed)
        else:
            _, (hidden, _cell) = self.lstm(embedded)

        hidden = hidden[-1]  # (B, H)
        return self.fc(self.dropout(hidden))


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier.

    Architecture: Embedding -> nn.LSTM(bidirectional=True) -> Linear
    Concatenates the final forward and backward hidden states to form a
    representation of size 2 * hidden_dim, which is projected to the output.

    Args:
        vocab_size:  Size of the token vocabulary.
        embed_dim:   Dimensionality of token embeddings.
        hidden_dim:  Number of features in each direction's hidden state.
        output_dim:  Number of output classes.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout between LSTM layers (applied when num_layers > 1).
        pad_idx:     Padding token index.

    Example:
        >>> model = BiLSTMClassifier(vocab_size=5000, embed_dim=64, hidden_dim=128, output_dim=2)
        >>> logits = model(torch.randint(0, 5000, (32, 20)))  # (32, 2)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Forward + backward hidden states are concatenated -> 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:       Token indices of shape (batch, seq_len).
            lengths: Actual (unpadded) lengths per sample for packing.

        Returns:
            Logits of shape (batch, output_dim).
        """
        embedded = self.embedding(x)

        if lengths is not None:
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _cell) = self.lstm(packed)
        else:
            _, (hidden, _cell) = self.lstm(embedded)

        # hidden shape: (num_layers * 2, B, H) for bidirectional
        # Last layer forward: hidden[-2], last layer backward: hidden[-1]
        forward_hidden = hidden[-2]  # (B, H)
        backward_hidden = hidden[-1]  # (B, H)
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # (B, 2H)

        return self.fc(self.dropout(combined))
