"""Train an autoencoder on MNIST and visualize reconstructions.

Trains a vanilla autoencoder (784 -> 256 -> 64 -> 256 -> 784) on the MNIST
handwritten digit dataset.  After training, saves a side-by-side comparison
of original and reconstructed digits to `reconstruction.png`.

Usage:
    python train_autoencoder.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import shared utilities from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoencoder import Autoencoder

from utils import Timer, get_device, set_seed

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
HIDDEN_DIMS = [256, 64]
INPUT_DIM = 784  # 28 * 28

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load MNIST with flattened 28x28 images normalized to [0, 1]."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Scales to [0, 1]
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
        ]
    )

    data_dir = Path(__file__).resolve().parent / "data"

    train_dataset = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: Autoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for images, _ in loader:
        images = images.to(device)

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: Autoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate and return the average loss."""
    model.eval()
    total_loss = 0.0

    for images, _ in loader:
        images = images.to(device)
        reconstructed = model(images)
        total_loss += criterion(reconstructed, images).item() * images.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_reconstruction_plot(
    model: Autoencoder,
    loader: DataLoader,
    device: torch.device,
    n_images: int = 10,
    save_path: str = "reconstruction.png",
) -> None:
    """Save a plot comparing original and reconstructed images."""
    model.eval()

    # Grab one batch
    images, _ = next(iter(loader))
    images = images[:n_images].to(device)
    reconstructed = model(images).cpu()
    images = images.cpu()

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3))

    for i in range(n_images):
        # Original
        axes[0, i].imshow(images[i].view(28, 28), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        # Reconstructed
        axes[1, i].imshow(reconstructed[i].view(28, 28), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / save_path
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    print(f"Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")

    model = Autoencoder(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS).to(device)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    with Timer() as timer:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_loss = evaluate(model, test_loader, criterion, device)
            print(
                f"Epoch {epoch:>2}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_loss:.6f}"
            )

    print(f"\nTraining completed in {timer}")

    # Save reconstruction visualization
    save_reconstruction_plot(model, test_loader, device)


if __name__ == "__main__":
    main()
