"""Multi-Layer Perceptron (MLP) model with configurable architecture."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected feedforward neural network.

    Supports classification (multi-class via softmax) and regression tasks
    with configurable hidden layers, activation functions, and dropout.

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer sizes (e.g. [256, 128]).
        output_dim: Number of output units.
        activation: Activation function class (default: nn.ReLU).
        dropout: Dropout probability applied after each hidden layer.
        task: "classification" or "regression".

    Example:
        >>> model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10, dropout=0.2)
        >>> x = torch.randn(32, 784)
        >>> logits = model(x)  # (32, 10)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        task: str = "classification",
    ) -> None:
        super().__init__()

        if task not in ("classification", "regression"):
            msg = f"task must be 'classification' or 'regression', got '{task}'"
            raise ValueError(msg)

        self.task = task

        # Build layers dynamically from the list of hidden sizes
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        # Output projection (no activation — raw logits / predictions)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        For classification the model returns raw logits during training
        (use CrossEntropyLoss which applies log-softmax internally) and
        softmax probabilities during evaluation.

        For regression the model returns raw predictions.
        """
        out = self.network(x)

        if self.task == "classification" and not self.training:
            out = torch.softmax(out, dim=-1)

        return out
