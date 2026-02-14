# Deep Learning from Scratch

PyTorch implementations of foundational deep learning models — from perceptrons to recurrent networks. Each module includes clean, modular code, a minimal training example, and a detailed README covering mathematical foundations, intuition, and practical guidance.

Built for **Apple Silicon** (MPS acceleration) with full CPU/CUDA fallback.

## Modules

| # | Module | Key Models | Task |
|---|--------|-----------|------|
| 00 | [Perceptron](00-perceptron/) | Single-layer perceptron (from scratch + PyTorch) | Binary classification |
| 01 | [MLP](01-mlp/) | Multi-layer perceptron | MNIST digit classification |
| 02 | [Autoencoder](02-autoencoder/) | Autoencoder, Word2Vec (Skip-gram, CBOW) | Reconstruction, word embeddings |
| 03 | [CNN](03-cnn/) | Convolutional neural network | CIFAR-10 image classification |
| 04 | [RNN](04-rnn/) | RNN, GRU, LSTM, BiLSTM | Sentiment classification |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run any training script
uv run python 00-perceptron/train.py
uv run python 01-mlp/train.py
uv run python 02-autoencoder/train_autoencoder.py
uv run python 02-autoencoder/train_word2vec.py
uv run python 03-cnn/train.py
uv run python 04-rnn/train.py
```

## Stack

- **Python** 3.12
- **PyTorch** 2.10 (MPS / CUDA / CPU)
- **torchvision** for datasets
- **uv** for dependency management
- **ruff** for linting and formatting

## Lint

```bash
uv run ruff check .
uv run ruff format --check .
```

## Structure

```
.
├── utils.py                  # Shared: device selection, seeding, timer
├── 00-perceptron/
│   ├── model.py              # PerceptronScratch, Perceptron
│   ├── train.py              # Synthetic 2D classification
│   └── README.md
├── 01-mlp/
│   ├── model.py              # MLP (configurable layers/dropout)
│   ├── train.py              # MNIST training
│   └── README.md
├── 02-autoencoder/
│   ├── autoencoder.py        # Autoencoder (symmetric encoder-decoder)
│   ├── word2vec.py           # SkipGram, CBOW with negative sampling
│   ├── train_autoencoder.py  # MNIST reconstruction
│   ├── train_word2vec.py     # Word embedding training
│   └── README.md
├── 03-cnn/
│   ├── model.py              # CNN (conv blocks + classifier)
│   ├── train.py              # CIFAR-10 training
│   └── README.md
├── 04-rnn/
│   ├── model.py              # SimpleRNN, GRU, LSTM, BiLSTM
│   ├── train.py              # Synthetic sentiment analysis
│   └── README.md
├── pyproject.toml
└── uv.lock
```

## License

MIT
