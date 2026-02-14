"""Train an MLP on MNIST and evaluate on the test set."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Import shared utilities from the project root
# ---------------------------------------------------------------------------
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT_DIR)

from model import MLP  # noqa: E402

from utils import Timer, get_device, set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.2
SEED = 42


def get_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Download MNIST and return train / test DataLoaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST global mean / std
        ]
    )

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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

    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)  # flatten 28x28 -> 784
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def test_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Test the model on a dataset; return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def save_training_curve(losses: list[float], path: str = "training_curve.png") -> None:
    """Plot and save the per-epoch training loss curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MLP Training Loss on MNIST")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curve saved to {path}")


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model = MLP(
        input_dim=784,
        hidden_dims=HIDDEN_DIMS,
        output_dim=10,
        dropout=DROPOUT,
        task="classification",
    ).to(device)

    print(f"\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_losses: list[float] = []

    with Timer() as timer:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = test_model(model, test_loader, criterion, device)
            train_losses.append(train_loss)

            print(
                f"Epoch {epoch:>2}/{EPOCHS}  "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}"
            )

    print(f"\nTraining completed in {timer}")

    # ------------------------------------------------------------------
    # Final test
    # ------------------------------------------------------------------
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    # ------------------------------------------------------------------
    # Save training curve
    # ------------------------------------------------------------------
    save_training_curve(train_losses)


if __name__ == "__main__":
    main()
