# 00 — Perceptron

The **perceptron** is the simplest possible neural network: a single artificial neuron that
performs binary classification by learning a linear decision boundary. Invented by Frank
Rosenblatt in 1958, it is the foundational building block upon which all modern deep learning
is built.

---

## Architecture

```
        Inputs          Weights          Summation       Activation
       --------        ---------        -----------      ----------

       x_1 ----[ w_1 ]---\
                           \
       x_2 ----[ w_2 ]-----+----> z = w . x + b ----> y = sign(z)
                           /
       ...               /
                         /
       x_n ----[ w_n ]--/
                           ^
                           |
                         [ b ]
                         bias
```

**One neuron, one output.** Each input feature is multiplied by a learnable weight, the
products are summed with a bias term, and the sign of the result determines the predicted class.

---

## Mathematical Foundations

### The Perceptron Model

The perceptron computes a binary prediction from an input vector
$\mathbf{x} \in \mathbb{R}^n$:

$$
\hat{y} = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)
$$

where:

- $\mathbf{w} \in \mathbb{R}^n$ is the weight vector
- $b \in \mathbb{R}$ is the bias (threshold)
- $\text{sign}(z) = \begin{cases} +1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}$

The decision boundary is the hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$, which
divides the input space into two half-spaces corresponding to the two classes.

### The Perceptron Learning Rule

Given a misclassified sample $(\mathbf{x}_i, y_i)$ where $\hat{y}_i \neq y_i$, the weights
and bias are updated:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \, y_i \, \mathbf{x}_i
$$

$$
b \leftarrow b + \eta \, y_i
$$

where $\eta > 0$ is the learning rate. The update nudges the decision boundary so that
$\mathbf{x}_i$ is more likely to be classified correctly on the next pass.

**Intuition:** If a positive sample ($y_i = +1$) is misclassified, we *add* $\mathbf{x}_i$
to the weight vector, rotating the boundary toward correctly classifying it. If a negative
sample ($y_i = -1$) is misclassified, we *subtract* $\mathbf{x}_i$.

### The Perceptron Convergence Theorem

> If the training data is **linearly separable**, the perceptron learning algorithm is
> guaranteed to converge in a finite number of updates.

More precisely, if there exists a weight vector $\mathbf{w}^*$ with $\|\mathbf{w}^*\| = 1$
and a margin $\gamma > 0$ such that $y_i (\mathbf{w}^* \cdot \mathbf{x}_i) \geq \gamma$
for all training samples, then the number of weight updates is bounded by:

$$
k \leq \left(\frac{R}{\gamma}\right)^2
$$

where $R = \max_i \|\mathbf{x}_i\|$ is the radius of the data.

---

## Model Intuition

### Geometric View

The perceptron finds a **separating hyperplane** in the input space. In 2D, this is a line;
in 3D, a plane; in $n$ dimensions, an $(n-1)$-dimensional hyperplane. Everything on one side
is classified as $+1$, everything on the other as $-1$.

```
    x_2
     ^
     |    + + +
     |  + + + +     <- Class +1 (above the line)
     | + + + +
     |_ _ _ _ _ _ _ _ _  <-- decision boundary: w . x + b = 0
     |  - - - -
     |    - - - -   <- Class -1 (below the line)
     |      - -
     +-------------------> x_1
```

### Biological Analogy

The perceptron loosely models a biological neuron:

| Biological Neuron     | Perceptron              |
|-----------------------|-------------------------|
| Dendrites (inputs)    | Input features $x_i$    |
| Synaptic strengths    | Weights $w_i$           |
| Cell body (soma)      | Weighted sum $z$        |
| Firing threshold      | Bias $b$                |
| Axon (output signal)  | Prediction $\hat{y}$    |

A biological neuron "fires" when its inputs exceed a threshold — the perceptron outputs $+1$
when $\mathbf{w} \cdot \mathbf{x} + b \geq 0$.

---

## When to Use

- **Linearly separable binary classification** — the perceptron is ideal when you know (or
  suspect) that a simple linear boundary can separate your two classes.
- **Educational purposes** — understanding the perceptron is a prerequisite for understanding
  multi-layer perceptrons, backpropagation, and deep networks.
- **Baseline model** — use as a sanity-check baseline before trying more complex architectures.
- **Online learning** — the perceptron naturally supports incremental, single-sample updates.

---

## Strengths

| Strength | Details |
|----------|---------|
| **Simplicity** | One of the easiest ML models to implement and understand. |
| **Guaranteed convergence** | Will always find a solution if data is linearly separable (convergence theorem). |
| **Interpretability** | Weights directly indicate feature importance and direction. |
| **Speed** | Extremely fast to train — each update is $O(n)$ for $n$ features. |
| **No hyperparameter tuning** | Learning rate affects speed but not the final solution (for separable data). |
| **Online-capable** | Can learn from streaming data one sample at a time. |

---

## Weaknesses

| Weakness | Details |
|----------|---------|
| **Cannot solve XOR** | Fails on any non-linearly-separable problem (Minsky & Papert, 1969). |
| **Binary output only** | Produces hard $\{-1, +1\}$ labels, not probabilities. |
| **No hidden representation** | Cannot learn feature transformations — only a linear boundary. |
| **Sensitive to feature scaling** | Unscaled features can cause slow or erratic convergence. |
| **Non-unique solution** | The final boundary depends on initialization and data ordering. |
| **No convergence guarantee for non-separable data** | May oscillate forever if no linear boundary exists. |

### The XOR Problem

The XOR function cannot be represented by any single hyperplane:

```
  x_2
   1 |  -       +
     |
   0 |  +       -
     +--+-------+--> x_1
        0       1
```

No single line can separate the `+` from the `-` points. This limitation motivated the
development of **multi-layer perceptrons** (MLPs) with hidden layers.

---

## Fine-Tuning Guidance

### Learning Rate Selection

The learning rate $\eta$ controls step size during weight updates:

| $\eta$ value | Effect |
|:---:|--------|
| Too large ($>1$) | Weights overshoot — oscillations, slow or no convergence. |
| Moderate ($0.01$–$0.1$) | Good default range for normalized data. |
| Too small ($<0.001$) | Convergence is guaranteed but very slow. |

For linearly separable data, the perceptron converges regardless of learning rate — but a
well-chosen $\eta$ converges faster.

### Feature Normalization

Always normalize your features before training a perceptron. Without normalization,
features with large magnitude dominate the decision boundary.

Common strategies:

- **Zero-mean, unit-variance** (standardization): $x' = (x - \mu) / \sigma$
- **Min-max scaling** to $[0, 1]$ or $[-1, 1]$

### When to Move to an MLP

If your perceptron fails to converge or achieves poor accuracy, the data is likely **not
linearly separable**. Next steps:

1. **Add a hidden layer** — even one hidden layer with nonlinear activations (ReLU, sigmoid)
   can solve XOR and many nonlinear problems.
2. **Use a proper loss** — switch from the perceptron rule to gradient-based optimization
   with cross-entropy loss.
3. **See** [`01-mlp/`](../01-mlp/) for the natural next step.

---

## Files in This Module

| File | Description |
|------|-------------|
| `model.py` | `PerceptronScratch` (manual updates) and `Perceptron` (nn.Module) |
| `train.py` | Training script with synthetic data, accuracy logging, and plotting |
| `decision_boundary.png` | Generated plot comparing both implementations |

## Running

```bash
uv run python 00-perceptron/train.py
```

---

## References

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and
  Organization in the Brain.* Psychological Review, 65(6), 386–408.
- Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry.*
  MIT Press.
- Novikoff, A. B. J. (1963). *On Convergence Proofs for Perceptrons.* Symposium on
  Mathematical Theory of Automata.
