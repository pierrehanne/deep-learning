"""Training script for the Perceptron module.

Demonstrates both the from-scratch and PyTorch perceptron on a synthetic
linearly-separable 2D dataset. Prints training progress, final accuracy,
and saves a decision-boundary plot to `decision_boundary.png`.

Usage:
    uv run python 00-perceptron/train.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Import shared utilities from the repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import Perceptron, PerceptronScratch  # noqa: E402

from utils import Timer, get_device, set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_linearly_separable(
    n_samples: int = 200,
    noise: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a 2D linearly-separable binary classification dataset.

    Creates two Gaussian clusters on opposite sides of a diagonal line.
    Labels are in {-1, +1}.

    Args:
        n_samples: Total number of samples (split evenly between classes).
        noise: Standard deviation of the Gaussian noise per cluster.

    Returns:
        x: Tensor of shape (n_samples, 2).
        y: Tensor of shape (n_samples,) with values in {-1, +1}.
    """
    half = n_samples // 2

    # Class +1: centered at (1, 1)
    x_pos = torch.randn(half, 2) * noise + torch.tensor([1.0, 1.0])
    # Class -1: centered at (-1, -1)
    x_neg = torch.randn(half, 2) * noise + torch.tensor([-1.0, -1.0])

    x = torch.cat([x_pos, x_neg], dim=0)
    y = torch.cat([torch.ones(half), -torch.ones(half)], dim=0)

    # Shuffle
    perm = torch.randperm(n_samples)
    return x[perm], y[perm]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_decision_boundary(
    x: torch.Tensor,
    y: torch.Tensor,
    weights_scratch: torch.Tensor,
    bias_scratch: torch.Tensor,
    model_pytorch: Perceptron,
    save_path: str = "decision_boundary.png",
) -> None:
    """Plot data points and decision boundaries for both perceptrons.

    Args:
        x: Input features (n_samples, 2).
        y: Labels in {-1, +1}.
        weights_scratch: Learned weights from the scratch perceptron.
        bias_scratch: Learned bias from the scratch perceptron.
        model_pytorch: Trained PyTorch perceptron module.
        save_path: File path for the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_np = x.numpy()
    y_np = y.numpy()
    x_min, x_max = x_np[:, 0].min() - 0.5, x_np[:, 0].max() + 0.5

    for ax, title in zip(axes, ["From Scratch", "PyTorch nn.Module"], strict=True):
        # Scatter plot
        ax.scatter(
            x_np[y_np == 1, 0],
            x_np[y_np == 1, 1],
            c="steelblue",
            label="Class +1",
            edgecolors="k",
            linewidth=0.5,
            s=40,
        )
        ax.scatter(
            x_np[y_np == -1, 0],
            x_np[y_np == -1, 1],
            c="coral",
            label="Class -1",
            edgecolors="k",
            linewidth=0.5,
            s=40,
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(title)
        ax.legend()

    # Decision boundary: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
    x1_range = torch.linspace(x_min, x_max, 200)

    # --- Scratch boundary ---
    w1_s, w2_s = weights_scratch[0].item(), weights_scratch[1].item()
    b_s = bias_scratch.item()
    if abs(w2_s) > 1e-8:
        x2_scratch = -(w1_s * x1_range + b_s) / w2_s
        axes[0].plot(x1_range.numpy(), x2_scratch.numpy(), "k--", linewidth=2, label="Boundary")
        axes[0].legend()

    # --- PyTorch boundary ---
    w_pt = model_pytorch.linear.weight.detach().cpu().squeeze()
    b_pt = model_pytorch.linear.bias.detach().cpu().item()
    w1_p, w2_p = w_pt[0].item(), w_pt[1].item()
    if abs(w2_p) > 1e-8:
        x2_pytorch = -(w1_p * x1_range + b_pt) / w2_p
        axes[1].plot(x1_range.numpy(), x2_pytorch.numpy(), "k--", linewidth=2, label="Boundary")
        axes[1].legend()

    # Match y-axis limits across subplots
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nDecision boundary plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------


def train_scratch(x: torch.Tensor, y: torch.Tensor) -> PerceptronScratch:
    """Train the from-scratch perceptron and print progress."""
    print("=" * 60)
    print("FROM-SCRATCH PERCEPTRON (manual weight updates)")
    print("=" * 60)

    model = PerceptronScratch(n_features=2, lr=0.1)

    with Timer() as t:
        history = model.train(x, y, n_epochs=100)

    # Print a few milestones
    milestones = [0, len(history) // 2, len(history) - 1]
    for i in sorted(set(milestones)):
        h = history[i]
        print(f"  Epoch {h['epoch']:3d} | Errors: {h['errors']:3d} | Accuracy: {h['accuracy']:.2%}")

    # Final accuracy on full data
    preds = model.predict(x)
    acc = (preds == y).float().mean().item()
    print(f"\n  Final accuracy: {acc:.2%}")
    print(f"  Weights: {model.weights.tolist()}")
    print(f"  Bias:    {model.bias.item():.4f}")
    print(f"  Time:    {t}")

    return model


def train_pytorch(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> Perceptron:
    """Train the PyTorch perceptron and print progress."""
    print()
    print("=" * 60)
    print("PYTORCH PERCEPTRON (nn.Module + autograd)")
    print("=" * 60)

    model = Perceptron(n_features=2).to(device)

    # Map labels from {-1, +1} -> {0, 1} for BCEWithLogitsLoss
    y_bce = ((y + 1) / 2).to(device)
    x_dev = x.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()

    n_epochs = 100

    with Timer() as t:
        for epoch in range(1, n_epochs + 1):
            # Forward pass
            logits = model(x_dev).squeeze(-1)
            loss = criterion(logits, y_bce)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0 or epoch == 1:
                # Compute accuracy using sign of logits (maps to {-1, +1})
                preds = torch.sign(logits.detach())
                y_dev = y.to(device)
                acc = (preds == y_dev).float().mean().item()
                print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {acc:.2%}")

    # Final accuracy — switch to inference mode
    model.eval()
    preds = model.predict(x_dev)
    y_dev = y.to(device)
    acc = (preds == y_dev).float().mean().item()
    w = model.linear.weight.detach().cpu().squeeze().tolist()
    b = model.linear.bias.detach().cpu().item()
    print(f"\n  Final accuracy: {acc:.2%}")
    print(f"  Weights: {w}")
    print(f"  Bias:    {b:.4f}")
    print(f"  Time:    {t}")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Generate synthetic data
    x, y = make_linearly_separable(n_samples=200, noise=0.3)
    print(f"Dataset: {x.shape[0]} samples, {x.shape[1]} features")
    print(f"Class balance: +1={int((y == 1).sum())} / -1={int((y == -1).sum())}\n")

    # Train both models
    scratch_model = train_scratch(x, y)
    pytorch_model = train_pytorch(x, y, device)

    # Plot decision boundaries side by side
    save_path = str(Path(__file__).resolve().parent / "decision_boundary.png")
    pytorch_model = pytorch_model.cpu()
    plot_decision_boundary(
        x,
        y,
        weights_scratch=scratch_model.weights,
        bias_scratch=scratch_model.bias,
        model_pytorch=pytorch_model,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
