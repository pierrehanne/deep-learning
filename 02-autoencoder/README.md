# 02 — Autoencoders & Word2Vec

Two foundational unsupervised learning architectures that learn compressed
representations of data without labels.

```
Part 1: Autoencoders      — learn to compress and reconstruct data
Part 2: Word2Vec           — learn dense word embeddings from text
```

---

## Part 1 — Autoencoders

### What Is an Autoencoder?

An autoencoder is a neural network trained to copy its input to its output
through a compressed intermediate representation (the **bottleneck**). By forcing
the data through a lower-dimensional space, the network learns to extract the
most important features.

```
         ENCODER                    DECODER
  Input ---------> Latent z ---------> Reconstruction
  x (784)          (64)                x_hat (784)

  +---------+     +------+     +---------+
  | 784     |---->| 256  |     | 256     |
  |         |     |      |---->|         |---->  x_hat
  |         |     +------+     |         |
  +---------+        |         +---------+
                  +------+
                  |  64  |  <-- bottleneck (latent space)
                  +------+
```

### Mathematical Foundations

An autoencoder consists of two functions:

**Encoder** — maps input to latent space:

$$z = f_\theta(x) = \sigma(W_e x + b_e)$$

**Decoder** — maps latent space back to input space:

$$\hat{x} = g_\phi(z) = \sigma(W_d z + b_d)$$

**Reconstruction loss** — the network minimizes the difference between input
and output:

$$\mathcal{L}(\theta, \phi) = \frac{1}{N} \sum_{i=1}^{N} \| x_i - \hat{x}_i \|^2$$

For inputs normalized to $[0, 1]$, binary cross-entropy can also be used:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i) \right]$$

### The Bottleneck Principle

The bottleneck forces the network to learn a compressed representation. If the
latent dimension equals the input dimension, the network can learn the identity
function and nothing useful is captured.

```
Dimension:  784 -----> 256 -----> 64 -----> 256 -----> 784
                                  ^^
                            bottleneck: must capture
                            the essence of the input
                            in only 64 numbers
```

The smaller the bottleneck, the more the network must compress. Too small and
it loses important information; too large and it does not learn useful features.

### Connection to PCA

A **linear autoencoder** (no activation functions) learns the same subspace as
PCA. Specifically, if an autoencoder has one hidden layer with linear
activations and is trained with MSE loss, the learned weights span the same
subspace as the top-$k$ principal components:

$$\min_{W_e, W_d} \| X - X W_e W_d \|^2_F$$

This is equivalent to the PCA objective. Nonlinear activations allow
autoencoders to capture structure that PCA cannot.

### Variants

| Variant | Key Idea |
|---------|----------|
| **Denoising** | Add noise to input, train to reconstruct clean version. Forces the network to learn robust features rather than memorizing. |
| **Sparse** | Add a sparsity penalty on the latent activations ($L_1$ or KL divergence). Encourages the network to use only a few active units. |
| **Variational (VAE)** | Latent space is a probability distribution $q(z|x)$. Enables generation of new samples by sampling from the latent space. Loss adds KL divergence: $\mathcal{L} = \text{reconstruction} + D_{KL}(q(z|x) \| p(z))$. |
| **Contractive** | Penalizes the Frobenius norm of the encoder Jacobian, making the representation robust to small input perturbations. |

---

## Part 2 — Word2Vec

### What Is Word2Vec?

Word2Vec learns dense vector representations (embeddings) for words from
unlabeled text. Words that appear in similar contexts get similar vectors.

```
Vocabulary: [the, cat, sat, on, mat, dog, ran, ...]

                    Embedding Space (2D projection)

          cat *
                    * dog

     sat *      * ran

            * the
       * on
            * mat
```

### Skip-Gram

Given a center word, predict surrounding context words.

```
Sentence:    "the  cat  sat  on  the  mat"
                    ^^^
              center word = "sat"

Window = 2:  context = {cat, on}  (also possibly {the, the})

    center: "sat"  --->  predict: "cat", "on", "the", "the"
```

**Objective** — maximize the probability of context words given the center word:

$$J = \frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \log P(w_{t+j} \mid w_t)$$

where $T$ is the corpus length and $c$ is the window size.

With the softmax formulation:

$$P(w_O \mid w_I) = \frac{\exp(v'_{w_O} \cdot v_{w_I})}{\sum_{w=1}^{V} \exp(v'_w \cdot v_{w_I})}$$

where $v$ and $v'$ are the center and context embedding vectors.

### CBOW (Continuous Bag of Words)

Given surrounding context words, predict the center word.

```
Sentence:    "the  cat  sat  on  the  mat"

Window = 2:  context = {the, cat, on, the}  --->  predict: "sat"

    context words       average        predict center
    [the, cat, on, the] -------> v_ctx -------> "sat"
```

**Objective** — maximize the probability of the center word given its context:

$$P(w_t \mid w_{t-c}, \ldots, w_{t+c}) = \frac{\exp(v_{w_t} \cdot \bar{v}_{\text{ctx}})}{\sum_{w=1}^{V} \exp(v_w \cdot \bar{v}_{\text{ctx}})}$$

where $\bar{v}_{\text{ctx}} = \frac{1}{2c} \sum_{j} v'_{w_{t+j}}$ is the mean of the context embeddings.

### Negative Sampling

The full softmax requires summing over the entire vocabulary $V$ for every
training step — this is $O(V)$ per example and infeasible for large
vocabularies (often 100K+ words).

**Negative sampling** replaces the expensive softmax with a binary
classification task: distinguish real (center, context) pairs from randomly
sampled "negative" pairs.

For a positive pair $(w, c)$ and $k$ negative samples $\{n_1, \ldots, n_k\}$:

$$\mathcal{L} = -\log \sigma(v'_c \cdot v_w) - \sum_{i=1}^{k} \log \sigma(-v'_{n_i} \cdot v_w)$$

where $\sigma$ is the sigmoid function. This reduces the cost from $O(V)$ to
$O(k)$ per training example, where typically $k = 5\text{-}20$.

Negative words are sampled from a noise distribution, typically the unigram
distribution raised to the 3/4 power:

$$P_n(w) = \frac{f(w)^{3/4}}{\sum_{w'} f(w')^{3/4}}$$

This sub-linear scaling gives rare words a slightly higher chance of being
selected as negatives.

### The Embedding Space

After training, the learned vectors capture semantic relationships:

```
vec("king") - vec("man") + vec("woman") ~ vec("queen")
```

Words cluster by meaning:

```
Animals:    cat, dog, bear, deer, fish      (nearby in vector space)
Nature:     river, forest, mountain, valley  (nearby in vector space)
Actions:    runs, walks, jumps, swims        (nearby in vector space)
```

### Connection to Autoencoders

Word2Vec can be viewed as **implicit matrix factorization**. The skip-gram
objective with negative sampling approximately factorizes a shifted PMI
(Pointwise Mutual Information) matrix:

$$W \cdot C^\top \approx M$$

where $M_{ij} = \text{PMI}(w_i, c_j) - \log k$ and $k$ is the number of
negative samples.

This connects to autoencoders: both learn low-rank representations of
high-dimensional data. An autoencoder compresses pixels; Word2Vec compresses
co-occurrence statistics. Both use a bottleneck to force useful representations.

```
Autoencoder:    pixels  -->  latent z  -->  pixels
Word2Vec:       one-hot -->  embedding -->  context prediction
                             (both learn compressed representations)
```

---

## When to Use

| Use Case | Recommended Approach |
|----------|---------------------|
| Dimensionality reduction | Autoencoder (nonlinear alternative to PCA) |
| Feature learning / pretraining | Autoencoder as a pretrained encoder for downstream tasks |
| Anomaly detection | Autoencoder (high reconstruction error = anomaly) |
| Image denoising | Denoising autoencoder |
| Word embeddings | Word2Vec (Skip-Gram or CBOW) |
| Transfer learning in NLP | Pretrained Word2Vec embeddings as input features |
| Finding word similarities | Cosine similarity on Word2Vec embeddings |

## Strengths

- **Unsupervised**: no labels required, can leverage large amounts of unlabeled data
- **Learns useful representations**: captures structure that downstream tasks can exploit
- **Flexible architecture**: encoder/decoder can be any differentiable function (MLP, CNN, RNN)
- **Scalable**: negative sampling makes Word2Vec practical for very large vocabularies
- **Composable**: learned representations transfer well to other tasks

## Weaknesses

- **Reconstruction is not always useful**: minimizing reconstruction loss does not guarantee the learned features are useful for downstream tasks
- **Training instability**: autoencoders can learn trivial solutions (near-identity) if the bottleneck is too large
- **Hyperparameter sensitive**: bottleneck size, learning rate, and architecture choices strongly affect quality
- **No probabilistic interpretation** (vanilla autoencoder): unlike VAEs, vanilla autoencoders do not model a distribution over the latent space
- **Word2Vec limitations**: does not handle polysemy (one vector per word), out-of-vocabulary words require retraining

## Fine-Tuning Guidance

### Autoencoder Hyperparameters

| Parameter | Guidance |
|-----------|----------|
| **Bottleneck size** | Start with input_dim / 10. Too small = blurry reconstructions. Too large = no compression. |
| **Hidden layers** | 2-3 layers is usually sufficient. Gradually reduce dimensions (e.g., 784 -> 256 -> 64). |
| **Learning rate** | Adam with 1e-3 is a good default. Reduce if training is unstable. |
| **Activation** | ReLU for hidden layers. Sigmoid output for [0,1] inputs; linear for unbounded. |
| **Loss function** | MSE for continuous data. BCE for binary/normalized data. |

### Word2Vec Hyperparameters

| Parameter | Guidance |
|-----------|----------|
| **Embedding dimension** | 50-300. Larger = more expressive but slower and needs more data. |
| **Window size** | 2-10. Smaller = more syntactic similarity. Larger = more semantic similarity. |
| **Negative samples** | 5-20. More negatives = better approximation but slower training. 5 works well in practice. |
| **Min word frequency** | 2-5. Filters noise words that appear too rarely to learn good embeddings. |
| **Learning rate** | 0.01-0.025 for SGD. Use linear decay for large corpora. |
| **Skip-Gram vs CBOW** | Skip-Gram works better on small data and rare words. CBOW is faster and works better on frequent words. |

---

## Files

| File | Description |
|------|-------------|
| `autoencoder.py` | Autoencoder model with configurable encoder/decoder |
| `word2vec.py` | Skip-Gram and CBOW models with negative sampling |
| `train_autoencoder.py` | Train on MNIST, save reconstruction comparison |
| `train_word2vec.py` | Train on small corpus, show nearest neighbors |

## Usage

```bash
# Train autoencoder on MNIST
python 02-autoencoder/train_autoencoder.py

# Train Word2Vec on sample corpus
python 02-autoencoder/train_word2vec.py
```

## References

- Rumelhart, Hinton, Williams (1986). "Learning internal representations by error propagation"
- Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Levy & Goldberg (2014). "Neural Word Embedding as Implicit Matrix Factorization"
