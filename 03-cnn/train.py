"""Train a CNN on CIFAR-10 and evaluate on the test set."""

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

from model import CNN  # noqa: E402

from utils import Timer, get_device, set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
FILTERS = [32, 64]
FC_DIM = 256
DROPOUT = 0.5
SEED = 42

# ---------------------------------------------------------------------------
# CIFAR-10 normalization constants (per-channel mean and std)
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return train / test DataLoaders.

    Training data is augmented with random horizontal flips and random crops
    with padding, which is standard practice for CIFAR-10.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

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
        images, labels = images.to(device), labels.to(device)

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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a dataset; return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def save_training_curves(
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
    path: str = "training_curves.png",
) -> None:
    """Plot and save loss and accuracy curves side by side."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax_loss.plot(epochs, train_losses, marker="o", label="Train")
    ax_loss.plot(epochs, test_losses, marker="s", label="Test")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("CNN Training & Test Loss (CIFAR-10)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax_acc.plot(epochs, train_accs, marker="o", label="Train")
    ax_acc.plot(epochs, test_accs, marker="s", label="Test")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("CNN Training & Test Accuracy (CIFAR-10)")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {path}")


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model = CNN(
        in_channels=3,
        num_classes=10,
        filters=FILTERS,
        fc_dim=FC_DIM,
        dropout=DROPOUT,
    ).to(device)

    print(f"\n{model}\n")
    print(f"Trainable parameters: {model.count_parameters():,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_losses: list[float] = []
    test_losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []

    with Timer() as timer:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(
                f"Epoch {epoch:>2}/{EPOCHS}  "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}"
            )

    print(f"\nTraining completed in {timer}")

    # ------------------------------------------------------------------
    # Final test result
    # ------------------------------------------------------------------
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    # ------------------------------------------------------------------
    # Save training curves
    # ------------------------------------------------------------------
    save_training_curves(train_losses, test_losses, train_accs, test_accs)


if __name__ == "__main__":
    main()
