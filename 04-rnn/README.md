# 04 - Recurrent Neural Networks (RNN, GRU, LSTM, BiLSTM)

This module explores the family of **recurrent neural networks** -- architectures
designed to process sequential data by maintaining an internal hidden state that
evolves over time. We implement four variants, train them on a synthetic
sentiment classification task, and compare their performance.

---

## Table of Contents

1. [Model Intuition](#model-intuition)
2. [Vanilla RNN](#vanilla-rnn)
3. [GRU (Gated Recurrent Unit)](#gru-gated-recurrent-unit)
4. [LSTM (Long Short-Term Memory)](#lstm-long-short-term-memory)
5. [BiLSTM (Bidirectional LSTM)](#bilstm-bidirectional-lstm)
6. [Comparison Table](#comparison-table)
7. [When to Use RNNs](#when-to-use-rnns)
8. [Strengths and Weaknesses](#strengths-and-weaknesses)
9. [Fine-Tuning Guidance](#fine-tuning-guidance)
10. [Running the Code](#running-the-code)

---

## Model Intuition

Sequential data -- text, time series, audio, genomic sequences -- has a natural
ordering where the meaning of each element depends on what came before it.
Standard feedforward networks treat inputs as flat, unordered vectors and have no
notion of "memory". Recurrent neural networks address this by processing one
element at a time while passing a **hidden state** forward from step to step:

```
               h_0        h_1        h_2              h_T
                |          |          |                |
                v          v          v                v
  x_1  --->  [RNN] --->  [RNN] --->  [RNN] ---> ... [RNN] ---> output
```

At every time step *t*, the network reads the current input *x_t* and the
previous hidden state *h_{t-1}*, then produces a new hidden state *h_t*. This
hidden state acts as the network's **memory** of everything it has seen so far.

The key challenge is making that memory effective over long sequences. Vanilla
RNNs struggle with this due to vanishing / exploding gradients. **Gating
mechanisms** (GRU, LSTM) solve the problem by learning *when* to remember and
*when* to forget.

---

## Vanilla RNN

### Architecture

```
  x_t ---->[Embedding]---+
                          |
                          v
  h_{t-1} ---------->[ tanh(W_hh * h + W_xh * x + b) ]---> h_t
                                                              |
                                                              v
                                                          [Linear] ---> y
```

### Mathematics

The hidden state at each time step is computed as:

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

The output (for classification) is:

$$
y = W_{hy} \cdot h_T + b_y
$$

where $h_T$ is the final hidden state after processing the entire sequence.

### The Vanishing / Exploding Gradient Problem

During backpropagation through time (BPTT), gradients are multiplied by
$W_{hh}$ at every time step. For a sequence of length $T$:

$$
\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}
$$

- If the largest eigenvalue of $W_{hh}$ is **< 1**, gradients shrink
  exponentially (**vanishing gradients**) -- the network forgets long-range
  dependencies.
- If the largest eigenvalue is **> 1**, gradients grow exponentially
  (**exploding gradients**) -- training becomes unstable.

This makes vanilla RNNs effectively unable to learn dependencies that span more
than ~10-20 time steps.

---

## GRU (Gated Recurrent Unit)

The GRU (Cho et al., 2014) introduces two gates to control information flow,
solving the vanishing gradient problem with fewer parameters than LSTM.

### Architecture

```
  x_t ---->[Embedding]---+-------+-------+
                          |       |       |
                          v       v       v
  h_{t-1} ----+---->[ sigmoid ] [ sigmoid ] [ tanh ]
               |        z_t        r_t       ~h_t
               |         |          |          |
               |         |    (r_t * h_{t-1})--+
               |         |                     |
               +-------->[ (1-z_t)*h_{t-1} + z_t*~h_t ]---> h_t
                                                              |
                                                              v
                                                          [Linear] ---> y
```

### Mathematics

**Update gate** -- decides how much of the old state to keep:

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

**Reset gate** -- decides how much of the old state to use when computing the candidate:

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

**Candidate hidden state** -- a new potential state computed with a reset view of the past:

$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)
$$

**Final hidden state** -- an interpolation between old and candidate:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

### How Gates Solve Vanishing Gradients

When $z_t \approx 0$, the hidden state passes through unchanged
($h_t \approx h_{t-1}$), creating a direct gradient path through time. The
network learns when to update its memory and when to preserve it, allowing
gradients to flow over long distances without vanishing.

### Why GRU Over LSTM?

GRU has **two gates** (vs. LSTM's three) and **no separate cell state**, meaning
~25% fewer parameters. It often performs comparably to LSTM on many tasks and
trains faster, making it a good default choice when computational efficiency
matters.

---

## LSTM (Long Short-Term Memory)

The LSTM (Hochreiter & Schmidhuber, 1997) is the most widely used gated RNN.
It introduces a separate **cell state** $c_t$ -- a "conveyor belt" that carries
information across time steps with minimal transformation.

### Architecture

```
  x_t ---->[Embedding]---+-------+-------+-------+
                          |       |       |       |
                          v       v       v       v
  h_{t-1} ----+---->[ sigmoid ][ sigmoid ][ tanh ][ sigmoid ]
               |        f_t       i_t      ~c_t      o_t
               |         |         |         |         |
               |         v         v         v         |
  c_{t-1} --->[ f_t * c_{t-1} + i_t * ~c_t ]--> c_t   |
                                                  |     |
                                                  v     v
                                           [ o_t * tanh(c_t) ] ---> h_t
                                                                     |
                                                                     v
                                                                 [Linear] ---> y
```

### Mathematics

**Forget gate** -- decides what to discard from the cell state:

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**Input gate** -- decides which new values to store:

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**Candidate cell state** -- new information to potentially add:

$$
\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

**Cell state update** -- the "conveyor belt" is selectively erased and written:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

**Output gate** -- decides what part of the cell state to expose:

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**Hidden state** -- a filtered view of the cell state:

$$
h_t = o_t \odot \tanh(c_t)
$$

### The Cell State as "Conveyor Belt"

The cell state $c_t$ flows through time with only **element-wise** operations
(multiply by forget gate, add new input). Unlike the vanilla RNN where the
hidden state is repeatedly squashed through $\tanh$, the cell state can carry
information unchanged for many time steps. The forget gate only needs to stay
close to 1 for the gradient to pass through -- a much easier learning signal
than maintaining the full weight matrix spectrum required by a vanilla RNN.

This design makes LSTMs capable of learning dependencies spanning hundreds of
time steps.

---

## BiLSTM (Bidirectional LSTM)

A bidirectional LSTM processes the sequence in **both directions** and
concatenates the results, giving the model access to both past and future
context at every time step.

### Architecture

```
  Forward LSTM:
  x_1 --> x_2 --> x_3 --> ... --> x_T
   |       |       |               |
   v       v       v               v
  h_1 --> h_2 --> h_3 --> ... --> h_T  (forward hidden states)
   ->      ->      ->              ->


  Backward LSTM:
  x_1 <-- x_2 <-- x_3 <-- ... <-- x_T
   |       |       |               |
   v       v       v               v
  h_1 <-- h_2 <-- h_3 <-- ... <-- h_T  (backward hidden states)
   <-      <-      <-              <-


  Combined (for classification):

  [h_T(forward) ; h_1(backward)]  ---> [Linear] ---> y
         (2 * hidden_dim)
```

### How It Works

1. A **forward LSTM** reads the sequence from $x_1$ to $x_T$, producing forward
   hidden states $\overrightarrow{h_1}, \ldots, \overrightarrow{h_T}$.
2. A **backward LSTM** reads the sequence from $x_T$ to $x_1$, producing
   backward hidden states $\overleftarrow{h_1}, \ldots, \overleftarrow{h_T}$.
3. For sequence classification, the final hidden states are concatenated:

$$
h = [\overrightarrow{h_T} \; ; \; \overleftarrow{h_1}]
$$

This combined representation has dimensionality $2 \times \text{hidden\_dim}$
and captures patterns from both the beginning and the end of the sequence.

### When Bidirectionality Helps

Bidirectional models are most useful when the **entire sequence is available at
once** (e.g., text classification, named entity recognition, machine
translation encoding). They are **not suitable** for autoregressive tasks like
language modelling or real-time prediction, where future tokens are unavailable.

---

## Comparison Table

| Feature             | Vanilla RNN     | GRU             | LSTM            | BiLSTM            |
|---------------------|-----------------|-----------------|-----------------|-------------------|
| Gates               | None            | 2 (update, reset)| 3 (forget, input, output) | 3 per direction |
| Cell state          | No              | No              | Yes             | Yes               |
| Parameters (relative)| Fewest         | Medium          | Most (unidirectional) | 2x LSTM    |
| Long-range memory   | Poor            | Good            | Very good       | Very good         |
| Bidirectional       | No              | No              | No              | Yes               |
| Training speed      | Fastest         | Fast            | Moderate        | Slowest           |
| Gradient flow       | Vanishing/exploding | Gated (stable) | Gated + cell (very stable) | Same as LSTM |
| Best for            | Very short sequences | Medium sequences, efficiency | Long sequences, general-purpose | Full-sequence tasks |

---

## When to Use RNNs

RNNs (and their gated variants) are a natural fit when:

- **Data is sequential** and order matters: text, time series, audio, sensor
  readings, genomic sequences.
- **Variable-length inputs** are common and you need a model that handles them
  natively without fixed-size windowing.
- **Temporal dependencies** exist: the meaning of an element depends on what
  came before (and after, for BiLSTM).
- **Moderate sequence lengths** (up to a few hundred tokens). For very long
  sequences (thousands of tokens), Transformers with attention are usually
  superior.
- **Computational resources are limited**: RNNs use less memory than
  attention-based models for moderate sequence lengths since they do not need to
  store a full attention matrix.

Common application domains:

- **NLP**: sentiment analysis, named entity recognition, POS tagging, machine
  translation (encoder side).
- **Time series**: forecasting, anomaly detection, signal processing.
- **Speech**: speech recognition, speaker diarization.
- **Biological sequences**: protein structure prediction, gene expression
  modelling.

---

## Strengths and Weaknesses

### Strengths

- **Variable-length sequences**: No fixed input size; sequences of any length
  can be processed.
- **Temporal dependency modelling**: Gated variants (GRU/LSTM) effectively
  capture both short- and long-range dependencies.
- **Parameter efficiency**: Weights are shared across time steps, so the number
  of parameters does not grow with sequence length.
- **Well-understood**: Decades of research, extensive literature, robust
  implementations in every deep learning framework.
- **Streaming / online**: Unidirectional RNNs can process inputs one step at a
  time, enabling real-time applications.

### Weaknesses

- **Sequential processing**: Each time step depends on the previous one, so
  computation cannot be parallelised across the sequence. This makes training
  slower than Transformers, especially on GPUs.
- **Long sequences**: Even LSTMs struggle with sequences beyond a few hundred
  tokens. Transformers with self-attention handle thousands of tokens more
  effectively.
- **No direct access to distant positions**: Information must be carried
  step-by-step through the hidden state. Attention mechanisms access any
  position directly.
- **Training instability**: Vanilla RNNs suffer from vanishing/exploding
  gradients. Even gated variants benefit from gradient clipping and careful
  initialisation.
- **Slower than Transformers**: For most NLP tasks on modern hardware,
  Transformers are faster to train and often more accurate.

---

## Fine-Tuning Guidance

### Hidden Size

- Start with 64--256 for small tasks, 256--512 for medium tasks, 512--1024 for
  large tasks.
- Larger hidden sizes capture more complex patterns but increase computation
  quadratically (weight matrices are $H \times H$).

### Number of Layers

- 1--2 layers are sufficient for most tasks.
- Deeper networks (3+ layers) can model hierarchical patterns but require
  residual connections and more careful tuning.
- Dropout between layers is essential for deeper models (0.2--0.5).

### Dropout

- Apply between recurrent layers (`dropout` parameter in PyTorch's RNN/GRU/LSTM
  when `num_layers > 1`).
- Also apply after the final hidden state before the classification head.
- Typical range: 0.1--0.5. Increase if overfitting, decrease if underfitting.

### Gradient Clipping

- Almost always beneficial. Clip gradients by norm with `max_norm` of 1.0--5.0.
- Prevents exploding gradients during training, especially for vanilla RNNs.
- In PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)`.

### Bidirectional vs. Unidirectional

- Use **bidirectional** when the full sequence is available (classification,
  tagging, encoding for seq2seq).
- Use **unidirectional** for autoregressive generation, real-time processing, or
  when future context is unavailable.
- Bidirectional doubles the representation size and parameter count.

### Teacher Forcing

- For sequence-to-sequence models: feed the ground-truth previous token as input
  during training (teacher forcing) vs. the model's own prediction (free
  running).
- Start with 100% teacher forcing and anneal toward 0% to prevent exposure bias.
- Not applicable for classification tasks (like the one in this module).

### Learning Rate

- Adam optimizer with LR 1e-3 is a reliable starting point.
- Reduce on plateau or use cosine annealing for longer training runs.
- If using SGD, start with LR 1.0 and decay aggressively.

### Embedding Dimension

- For small vocabularies (< 10k): 32--128.
- For large vocabularies (> 50k): 128--300.
- Pre-trained embeddings (GloVe, FastText) can provide a significant boost,
  especially with limited training data.

---

## Running the Code

```bash
# From the repository root
cd 04-rnn

# Train all four models and generate comparison chart
python train.py
```

The script will:

1. Generate a synthetic sentiment dataset (positive/negative sentences).
2. Build a vocabulary and create train/test DataLoaders with padding.
3. Train SimpleRNN, GRU, LSTM, and BiLSTM sequentially (20 epochs each).
4. Print per-epoch loss and accuracy for each model.
5. Save a comparison bar chart to `model_comparison.png`.

### Files

| File       | Description                                      |
|------------|--------------------------------------------------|
| `model.py` | Four `nn.Module` classes: SimpleRNN, GRUClassifier, LSTMClassifier, BiLSTMClassifier |
| `train.py` | Synthetic data generation, training loop, comparison chart |
| `README.md`| This document                                    |
