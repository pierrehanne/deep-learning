# 03 - Convolutional Neural Network (CNN)

A CNN for image classification trained on CIFAR-10, demonstrating hierarchical
feature extraction through learnable convolutional filters.

```
Architecture

Input (3x32x32)
      |
      v
+-----------------+
| Conv2d 3->32    |  3x3 kernel, padding=1
| BatchNorm2d     |
| ReLU            |
| MaxPool2d 2x2   |  -> 32 x 16 x 16
+-----------------+
      |
      v
+-----------------+
| Conv2d 32->64   |  3x3 kernel, padding=1
| BatchNorm2d     |
| ReLU            |
| MaxPool2d 2x2   |  -> 64 x 8 x 8
+-----------------+
      |
      v
  Flatten          -> 64 * 8 * 8 = 4096
      |
      v
+-----------------+
| Linear 4096->256|
| ReLU            |
| Dropout(0.5)    |
+-----------------+
      |
      v
+-----------------+
| Linear 256->10  |  -> logits (10 classes)
+-----------------+
```

## Quick Start

```bash
cd 03-cnn
python train.py
```

This downloads CIFAR-10 automatically, trains for 10 epochs, and saves
`training_curves.png` with loss and accuracy plots.

---

## Mathematical Foundations

### The Convolution Operation

Despite the name, CNNs actually compute **discrete cross-correlation** rather
than true convolution (the difference is that the kernel is not flipped). For
a 2D input $I$ and kernel $K$ of size $k \times k$:

$$
(I * K)[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i + m,\; j + n] \cdot K[m, n]
$$

Each output element is the dot product of the kernel with the local patch of
the input it overlaps.

### Feature Maps

A convolutional layer with $C_\text{out}$ filters produces $C_\text{out}$
**feature maps**. Each filter has shape $(C_\text{in}, k, k)$ and scans the
entire input spatially, producing one 2D feature map. The full output tensor
has shape $(B, C_\text{out}, H_\text{out}, W_\text{out})$.

### Output Size Formula

Given an input with spatial dimension $W$, kernel size $K$, padding $P$, and
stride $S$, the output spatial dimension is:

$$
W_\text{out} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

For this model (input 32, kernel 3, padding 1, stride 1):

$$
W_\text{out} = \left\lfloor \frac{32 - 3 + 2}{1} \right\rfloor + 1 = 32
$$

After 2x2 max-pooling with stride 2: $32 / 2 = 16$.

### Receptive Field

The **receptive field** of a neuron is the region of the input image that
influences its activation. After each conv + pool block, the receptive field
grows. For stacked 3x3 convolutions:

- 1 layer: 3x3 receptive field
- 2 layers: 5x5 effective receptive field
- 3 layers: 7x7 effective receptive field

This is why small 3x3 kernels stacked together can capture large-scale
patterns while using fewer parameters than a single large kernel.

### Pooling

**Max pooling** takes the maximum value in each local window:

$$
\text{MaxPool}(X)[i, j] = \max_{(m, n) \in \mathcal{R}_{i,j}} X[m, n]
$$

where $\mathcal{R}_{i,j}$ is the pooling region centered at $(i, j)$. Pooling
reduces spatial dimensions, provides a degree of translation invariance, and
increases the effective receptive field.

---

## Model Intuition

### Hierarchical Feature Learning

CNNs learn features in a hierarchy from simple to complex:

1. **Early layers** (first conv block) learn low-level features: edges,
   corners, color gradients
2. **Middle layers** combine low-level features into textures, patterns,
   and parts of objects
3. **Later layers** assemble these into high-level representations:
   object shapes, faces, semantic concepts

This mirrors the hierarchical processing in the biological visual cortex.

### Weight Sharing and Translation Invariance

A single convolutional filter is applied at every spatial position of the
input. This **weight sharing** has two consequences:

- **Parameter efficiency**: A 3x3 filter with 64 output channels on a 64-channel
  input has $3 \times 3 \times 64 \times 64 = 36{,}864$ parameters, regardless
  of whether the input is 32x32 or 1024x1024.
- **Translation equivariance**: If an object shifts in the input, its feature
  map shifts by the same amount. Combined with pooling, this yields approximate
  **translation invariance** --- the network recognizes the object regardless of
  where it appears.

### Why CNNs Work for Images

Images have strong **spatial locality** (nearby pixels are related) and
**stationarity** (the same patterns appear in different locations). CNNs
exploit both properties through local receptive fields and weight sharing,
making them dramatically more efficient than fully connected networks for
visual data.

---

## When to Use CNNs

| Use Case | Notes |
|---|---|
| **Image classification** | The canonical CNN application (e.g. CIFAR-10, ImageNet) |
| **Object detection** | CNN backbone + detection head (YOLO, Faster R-CNN) |
| **Semantic segmentation** | Encoder-decoder architectures (U-Net, DeepLab) |
| **Grid-structured data** | Audio spectrograms, time-frequency representations |
| **Feature extraction** | Pretrained CNN as a frozen feature extractor for downstream tasks |

---

## Strengths

- **Parameter efficiency**: Weight sharing means far fewer parameters than an
  equivalent fully connected network. A 3-layer CNN for 32x32 images uses
  ~100K parameters; a comparable MLP would need millions.
- **Translation invariance**: Recognizes patterns regardless of position,
  essential for visual recognition.
- **Hierarchical features**: Automatically learns a meaningful feature
  hierarchy without manual feature engineering.
- **State-of-the-art for vision**: Decades of proven results on image
  benchmarks, with well-understood training recipes.

## Weaknesses

- **Not rotation/scale invariant by default**: A cat rotated 90 degrees may
  not be recognized unless the training data includes rotated examples or
  explicit augmentation is used. Data augmentation or specialized
  architectures (e.g. group-equivariant CNNs) are needed.
- **Requires lots of data**: Training from scratch demands large labeled
  datasets; transfer learning mitigates this.
- **Computationally expensive**: Convolutions at high resolution are
  memory- and compute-intensive, especially during training.
- **Pooling loses spatial information**: Max/average pooling discards precise
  spatial positions, which matters for tasks requiring fine localization
  (addressed by architectures like U-Net or dilated convolutions).

---

## Fine-Tuning Guidance

### Filter Sizes

Use **3x3 kernels** as the default. Two stacked 3x3 layers have the same
receptive field as one 5x5 layer but with fewer parameters and more
non-linearity. This is the standard from VGGNet onwards.

### Number of Filters

A common pattern is to **double the filters each layer** (e.g. 32, 64, 128, 256).
As spatial dimensions halve through pooling, doubling the channel count
keeps the computational cost roughly constant per layer.

### Batch Normalization

Always include **BatchNorm after each convolution** (before or after ReLU ---
both work, but post-conv/pre-activation is most common). BatchNorm:

- Stabilizes training by normalizing activations
- Allows higher learning rates
- Acts as mild regularization

### Data Augmentation

Critical for preventing overfitting on small datasets like CIFAR-10:

- **RandomHorizontalFlip**: Doubles effective dataset size for symmetric objects
- **RandomCrop with padding**: Provides translation augmentation
- **ColorJitter**: Varies brightness, contrast, saturation
- **Cutout / Random Erasing**: Forces the network to use global context

### Transfer Learning

For most practical tasks, **start from a pretrained model** (e.g. ResNet
trained on ImageNet) rather than training from scratch:

1. Replace the final classification layer to match your number of classes
2. Freeze early layers (low-level features transfer well)
3. Fine-tune later layers with a small learning rate
4. Gradually unfreeze more layers if data permits

### Learning Rate Warmup

Start with a small learning rate and linearly increase it over the first
few epochs (or first ~5% of training steps). This prevents early divergence
when BatchNorm statistics and model weights are still poorly initialized.
After warmup, use cosine annealing or step decay.

---

## References

- LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998) --- the original LeNet paper
- Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015) --- VGGNet and the case for 3x3 filters
- He et al., "Deep Residual Learning for Image Recognition" (2016) --- ResNet and skip connections
- Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (2015)
