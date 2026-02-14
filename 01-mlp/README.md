# 01 - Multi-Layer Perceptron (MLP)

A fully-connected feedforward neural network trained on MNIST. This module covers
the mathematical foundations, intuitions, and practical considerations behind the
simplest deep learning architecture.

## Architecture Diagram

```
Input Layer          Hidden Layer 1       Hidden Layer 2       Output Layer
(784 neurons)        (256 neurons)        (128 neurons)        (10 neurons)

  x_1 ----\         /--- h1_1 ---\       /--- h2_1 ---\       /--- y_1
  x_2 -----+--W1--+---- h1_2 ----+--W2-+---- h2_2 ----+--W3-+--- y_2
  x_3 -----+      |     ...      |     |     ...      |     |    ...
  ...      ...     |   h1_256     |     |   h2_128     |     |   y_10
  x_784 ---/       \--- (ReLU) ---/     \--- (ReLU) ---/     \--- (softmax)
                      Dropout(0.2)         Dropout(0.2)

  [Flatten 28x28]   [Linear+Act+Drop]   [Linear+Act+Drop]   [Linear]
```

## Mathematical Foundations

### Forward Propagation

An MLP computes its output by passing data through a sequence of affine
transformations followed by non-linear activation functions. For a network with
$L$ hidden layers:

**Layer 1 (input to first hidden):**

$$z^{(1)} = W^{(1)} x + b^{(1)}$$

$$h^{(1)} = \sigma(z^{(1)})$$

**Layer $l$ (hidden to hidden):**

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$

$$h^{(l)} = \sigma(z^{(l)})$$

**Output layer (last hidden to output):**

$$z^{(L+1)} = W^{(L+1)} h^{(L)} + b^{(L+1)}$$

$$\hat{y} = \text{softmax}(z^{(L+1)})$$

where $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ is the weight matrix,
$b^{(l)} \in \mathbb{R}^{d_l}$ is the bias vector, and $\sigma$ is the
activation function for layer $l$.

### Activation Functions

Activation functions introduce non-linearity, enabling the network to learn
complex mappings. Without them, stacking linear layers collapses to a single
linear transformation.

**ReLU (Rectified Linear Unit)** -- default and most common:

$$\text{ReLU}(z) = \max(0, z)$$

$$\frac{\partial}{\partial z} \text{ReLU}(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

Advantages: computationally cheap, mitigates vanishing gradients for positive
values. Disadvantage: "dying ReLU" problem where neurons output zero for all
inputs.

**Sigmoid:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$$\frac{\partial}{\partial z} \sigma(z) = \sigma(z)(1 - \sigma(z))$$

Output range $(0, 1)$. Suffers from vanishing gradients for large $|z|$.
Useful for binary outputs or gating mechanisms.

**Tanh (Hyperbolic Tangent):**

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

$$\frac{\partial}{\partial z} \tanh(z) = 1 - \tanh^2(z)$$

Output range $(-1, 1)$. Zero-centered (unlike sigmoid), but still suffers from
vanishing gradients at the tails.

### Softmax (Output Layer for Classification)

Converts raw logits into a probability distribution over $K$ classes:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Combined with cross-entropy loss:

$$\mathcal{L} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

For one-hot encoded labels this simplifies to $\mathcal{L} = -\log(\hat{y}_c)$
where $c$ is the correct class.

### Backpropagation and the Chain Rule

Training minimizes the loss $\mathcal{L}$ by computing gradients of the loss
with respect to every parameter using the chain rule of calculus and then
updating parameters via gradient descent.

**Output layer gradients:**

$$\frac{\partial \mathcal{L}}{\partial W^{(L+1)}} = \frac{\partial \mathcal{L}}{\partial z^{(L+1)}} \cdot (h^{(L)})^T$$

**Hidden layer gradients (backpropagated error):**

$$\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot (h^{(l-1)})^T$$

where $\odot$ denotes element-wise multiplication and $\sigma'$ is the
derivative of the activation function.

The gradients flow backward through the network -- hence "backpropagation."
Each layer's gradient depends on the gradient from the layer above, scaled by
the local derivative. This is why activation function derivatives matter:
if $\sigma'$ is very small (vanishing gradients), learning slows; if very
large (exploding gradients), training becomes unstable.

### Universal Approximation Theorem

A feedforward network with a single hidden layer containing a finite number of
neurons can approximate any continuous function on a compact subset of
$\mathbb{R}^n$ to arbitrary accuracy, given a suitable activation function
(Hornik et al., 1989).

$$\forall \epsilon > 0, \exists N \in \mathbb{N} \text{ such that } \left| f(x) - \sum_{i=1}^{N} v_i \, \sigma(w_i^T x + b_i) \right| < \epsilon$$

In practice, deeper networks (more layers, fewer neurons per layer) are often
more parameter-efficient than very wide shallow networks.

## Model Intuition

### How Stacking Layers Creates Non-Linear Decision Boundaries

Each hidden layer applies a linear transformation followed by a non-linear
activation. The first layer learns simple features (edges, intensity patterns
in images). Subsequent layers compose these into increasingly abstract
representations:

- **Layer 1:** detects local patterns (strokes, curves)
- **Layer 2:** combines patterns into parts (loops, intersections)
- **Output:** maps parts to class identities (digits 0--9)

A single linear classifier can only draw hyperplanes. By stacking non-linear
layers, the network can carve out arbitrarily complex decision regions in the
input space.

### Why Depth Helps

While a single hidden layer is theoretically sufficient (universal
approximation), deeper networks learn **hierarchical representations** more
efficiently. Empirically:

- Depth provides exponentially more representational power per parameter
- Shallow networks may need exponentially many neurons to represent the same
  function a deep network learns with modest width
- Deeper networks tend to generalize better when properly regularized

### What Neurons "Learn"

Each neuron computes $\sigma(w^T x + b)$ -- a soft decision boundary in input
space. In early layers, neurons act as feature detectors. In later layers, they
combine features into higher-order concepts. The weight vector $w$ determines
*what* the neuron responds to; the bias $b$ shifts the activation threshold.

## When to Use an MLP

| Use Case | Why MLP Works |
|---|---|
| **Tabular data** | Handles heterogeneous features naturally |
| **Simple classification / regression** | Easy baseline before trying complex models |
| **Feature extraction** | Hidden layers learn useful representations |
| **Baseline neural network** | Quick to implement, well understood |
| **Small-to-medium datasets** | Fewer parameters than CNNs / Transformers |

MLPs are generally the **first architecture to try** when moving from
traditional ML (logistic regression, random forests) to neural networks. If
your data has spatial structure (images), use a CNN. If it has sequential
structure (text, time series), use an RNN or Transformer.

## Strengths

- **Universal approximation:** can represent any continuous function given
  enough capacity
- **Non-linear modeling:** handles complex relationships that linear models
  cannot capture
- **Flexible architecture:** easy to adjust width, depth, activations, and
  regularization
- **Well understood:** decades of research, reliable training procedures,
  extensive tooling
- **Fast training:** simple computation graph, efficient on modern hardware
- **Good on tabular data:** competitive with or superior to tree-based methods
  when properly tuned

## Weaknesses

- **No spatial awareness:** treats input as a flat vector -- ignores pixel
  neighborhoods, spatial hierarchies (use CNNs instead)
- **No temporal awareness:** cannot model sequences or dependencies over time
  (use RNNs or Transformers instead)
- **Prone to overfitting:** especially on small datasets with large networks;
  requires careful regularization (dropout, weight decay)
- **Requires feature engineering:** for structured data (images, text), raw
  features are suboptimal -- specialized architectures extract better features
  automatically
- **Fully connected = many parameters:** parameter count grows as
  $O(d_l \times d_{l-1})$ per layer, which can be wasteful when input
  dimensions are large
- **Sensitive to input scale:** performance degrades without proper
  normalization of input features

## Fine-Tuning Guidance

### Hidden Layer Sizing

- Start with 1--2 hidden layers; add more only if underfitting
- Use a **funnel shape**: decreasing width across layers
  (e.g., 512 -> 256 -> 128) to progressively compress representations
- Rule of thumb: first hidden layer between 1x and 4x the input dimension;
  subsequent layers halve the width
- Monitor validation loss -- if it plateaus, the network may lack capacity

### Dropout Rates

| Scenario | Recommended Dropout |
|---|---|
| Small dataset, large model | 0.3 -- 0.5 |
| Medium dataset | 0.1 -- 0.3 |
| Large dataset | 0.0 -- 0.1 |
| After input layer | 0.0 -- 0.2 (lighter) |
| Between hidden layers | 0.2 -- 0.5 |

Dropout is only active during training. It randomly zeros neurons, forcing the
network to learn redundant representations and reducing co-adaptation.

### Learning Rate Schedules

- **Constant:** simple baseline; use with Adam which adapts per-parameter
- **Step decay:** reduce LR by a factor every $N$ epochs
  (`StepLR(optimizer, step_size=10, gamma=0.1)`)
- **Cosine annealing:** smoothly decays LR following a cosine curve
  (`CosineAnnealingLR(optimizer, T_max=epochs)`)
- **Reduce on plateau:** lower LR when validation loss stops improving
  (`ReduceLROnPlateau(optimizer, patience=5)`)
- **Warmup:** start with a small LR and ramp up over the first few epochs;
  helps stabilize early training

Typical starting learning rates: Adam 1e-3, SGD 1e-2 to 1e-1.

### Batch Normalization

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

- Normalizes activations within each mini-batch to zero mean and unit variance
- Learnable scale ($\gamma$) and shift ($\beta$) parameters preserve
  expressiveness
- Placed **after the linear layer and before the activation** (common
  convention) or after the activation (also works)
- Benefits: faster convergence, allows higher learning rates, mild
  regularization effect
- Caution: behavior differs between training (batch stats) and inference
  (running stats)

### Weight Initialization

Poor initialization can cause vanishing or exploding activations. Two standard
strategies:

**Xavier / Glorot initialization** (good for sigmoid, tanh):

$$W \sim \mathcal{U}\left[-\frac{\sqrt{6}}{\sqrt{d_{\text{in}} + d_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{d_{\text{in}} + d_{\text{out}}}}\right]$$

Keeps the variance of activations constant across layers when using symmetric
activations.

**He / Kaiming initialization** (good for ReLU):

$$W \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}}}\right)$$

Accounts for ReLU zeroing half of the activations. PyTorch uses Kaiming
initialization by default for `nn.Linear` layers.

## Running the Code

```bash
# From the repository root
cd 01-mlp
uv run python train.py
```

This will:
1. Download MNIST (first run only, ~11 MB)
2. Train a 784 -> 256 -> 128 -> 10 MLP for 10 epochs
3. Print per-epoch training/test loss and accuracy
4. Save `training_curve.png` with the loss curve

## File Structure

```
01-mlp/
  model.py          # MLP class (nn.Module)
  train.py          # Training script (MNIST)
  README.md         # This file
  training_curve.png  # Generated after training
```
