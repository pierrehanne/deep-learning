"""Autoencoder with configurable symmetric encoder-decoder architecture.

An autoencoder learns compressed representations by training to reconstruct its
input through a bottleneck layer:

    Input x --> [Encoder] --> Latent z --> [Decoder] --> Reconstruction x_hat

The encoder maps high-dimensional input to a lower-dimensional latent space,
and the decoder maps it back.  The network is trained to minimize reconstruction
loss ||x - x_hat||^2, forcing the bottleneck to capture the most salient
features of the data.

Example:
    >>> model = Autoencoder(input_dim=784, hidden_dims=[256, 64])
    >>> # Encoder: 784 -> 256 -> 64
    >>> # Decoder: 64 -> 256 -> 784
    >>> x = torch.randn(32, 784)
    >>> x_hat = model(x)  # (32, 784)
    >>> z = model.encode(x)  # (32, 64)
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Vanilla autoencoder with symmetric encoder and decoder.

    The encoder compresses the input through progressively smaller layers,
    and the decoder mirrors this architecture to reconstruct the original input.
    ReLU activations are used between hidden layers, and a sigmoid activation
    is applied to the decoder output (suitable for inputs normalized to [0, 1]).

    Args:
        input_dim: Dimensionality of the input (e.g. 784 for flattened 28x28 MNIST).
        hidden_dims: List of hidden layer sizes for the encoder.  The decoder
            uses these in reverse order.  The last element is the bottleneck
            (latent) dimension.

    Example:
        >>> ae = Autoencoder(input_dim=784, hidden_dims=[256, 64])
        >>> ae.encode(torch.randn(8, 784)).shape
        torch.Size([8, 64])
        >>> ae(torch.randn(8, 784)).shape
        torch.Size([8, 784])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()

        if not hidden_dims:
            msg = "hidden_dims must contain at least one dimension (the bottleneck)"
            raise ValueError(msg)

        # ----- Encoder -----
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # ----- Decoder (mirror of encoder) -----
        decoder_dims = list(reversed(hidden_dims))
        decoder_layers: list[nn.Module] = []
        prev_dim = decoder_dims[0]
        for h_dim in decoder_dims[1:]:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        # Final layer maps back to input_dim with sigmoid output
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        self.latent_dim = hidden_dims[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into the latent space.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Latent representation of shape (batch_size, latent_dim).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim).

        Returns:
            Reconstruction of shape (batch_size, input_dim).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Reconstruction of shape (batch_size, input_dim).
        """
        z = self.encode(x)
        return self.decode(z)
