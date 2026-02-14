"""Perceptron implementations: from scratch and with PyTorch nn.Module.

A perceptron is the simplest neural network — a single neuron that performs
binary classification by learning a linear decision boundary.

    y = sign(w . x + b)

This module provides two implementations:
  1. PerceptronScratch — manual weight updates using the perceptron learning rule
  2. Perceptron — clean nn.Module wrapper using PyTorch autograd
"""

import torch
from torch import nn

# ---------------------------------------------------------------------------
# From-scratch implementation (no autograd)
# ---------------------------------------------------------------------------


class PerceptronScratch:
    """Perceptron with manual weight updates (no autograd).

    Implements the classic Rosenblatt perceptron learning rule:
        if y_pred != y_true:
            w <- w + lr * y_true * x
            b <- b + lr * y_true

    Args:
        n_features: Number of input features.
        lr: Learning rate for weight updates.
    """

    def __init__(self, n_features: int, lr: float = 0.01) -> None:
        self.lr = lr
        # Initialize weights to zeros (classic perceptron convention)
        self.weights = torch.zeros(n_features)
        self.bias = torch.zeros(1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute perceptron output: sign(w . x + b).

        Args:
            x: Input tensor of shape (n_samples, n_features) or (n_features,).

        Returns:
            Predictions in {-1, +1} of shape (n_samples,) or scalar.
        """
        linear = x @ self.weights + self.bias  # (n_samples,)
        return torch.sign(linear)

    def train(
        self, x: torch.Tensor, y: torch.Tensor, n_epochs: int = 100
    ) -> list[dict[str, float]]:
        """Train the perceptron using the perceptron learning rule.

        Args:
            x: Training inputs of shape (n_samples, n_features).
            y: Labels in {-1, +1} of shape (n_samples,).
            n_epochs: Number of passes through the training data.

        Returns:
            List of dicts with 'epoch', 'errors', and 'accuracy' per epoch.
        """
        history = []
        n_samples = x.shape[0]

        for epoch in range(n_epochs):
            errors = 0
            for i in range(n_samples):
                y_pred = self.predict(x[i])
                # Update only on misclassification
                if y_pred != y[i]:
                    # Perceptron learning rule
                    self.weights += self.lr * y[i] * x[i]
                    self.bias += self.lr * y[i]
                    errors += 1

            accuracy = 1.0 - errors / n_samples
            history.append({"epoch": epoch + 1, "errors": errors, "accuracy": accuracy})

            # Early stopping: converged
            if errors == 0:
                break

        return history


# ---------------------------------------------------------------------------
# PyTorch nn.Module implementation
# ---------------------------------------------------------------------------


class Perceptron(nn.Module):
    """Single-layer perceptron using PyTorch nn.Linear.

    For training this module, use a loss function like BCEWithLogitsLoss
    (treating the raw linear output as logits) and a standard optimizer.

    During inference, apply sign() to the output for {-1, +1} predictions,
    or sigmoid for probability estimates in [0, 1].

    Args:
        n_features: Number of input features.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw linear output (logit).

        Args:
            x: Input tensor of shape (batch_size, n_features).

        Returns:
            Raw output of shape (batch_size, 1).
        """
        return self.linear(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels in {-1, +1}.

        Args:
            x: Input tensor of shape (batch_size, n_features).

        Returns:
            Predictions of shape (batch_size,).
        """
        with torch.no_grad():
            logits = self.forward(x).squeeze(-1)
            return torch.sign(logits)
