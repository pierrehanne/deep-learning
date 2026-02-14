"""Convolutional Neural Network (CNN) for image classification."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN with configurable conv blocks followed by fully connected layers.

    Architecture: N conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
    followed by a classifier head (Linear -> ReLU -> Dropout -> Linear).

    Default configuration targets CIFAR-10 (32x32x3 images, 10 classes).

    Args:
        in_channels: Number of input channels (3 for RGB, 1 for grayscale).
        num_classes: Number of output classes.
        filters: List of filter counts for each conv block (e.g. [32, 64]).
        fc_dim: Hidden dimension of the fully connected classifier head.
        dropout: Dropout probability in the classifier head.
        kernel_size: Convolution kernel size (applied to all conv layers).
        pool_size: Max-pooling kernel size (applied after each conv block).

    Example:
        >>> model = CNN(in_channels=3, num_classes=10)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> logits = model(x)  # (4, 10)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        filters: list[int] | None = None,
        fc_dim: int = 256,
        dropout: float = 0.5,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()

        if filters is None:
            filters = [32, 64]

        # -----------------------------------------------------------------
        # Convolutional feature extractor
        # -----------------------------------------------------------------
        conv_blocks: list[nn.Module] = []
        prev_channels = in_channels

        for num_filters in filters:
            conv_blocks.extend(
                [
                    nn.Conv2d(
                        prev_channels,
                        num_filters,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # "same" padding before pooling
                    ),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=pool_size),
                ]
            )
            prev_channels = num_filters

        self.features = nn.Sequential(*conv_blocks)

        # -----------------------------------------------------------------
        # Compute the flattened feature size by doing a dummy forward pass
        # through the conv layers.  This keeps the model flexible to any
        # input resolution without hard-coding spatial dimensions.
        # -----------------------------------------------------------------
        self._flat_features = self._get_flat_features(in_channels)

        # -----------------------------------------------------------------
        # Classifier head
        # -----------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self._flat_features, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_flat_features(self, in_channels: int, size: int = 32) -> int:
        """Return the number of features after the conv blocks for a given spatial size."""
        dummy = torch.zeros(1, in_channels, size, size)
        with torch.no_grad():
            out = self.features(dummy)
        return out.numel()

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features -> flatten -> classifier.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
