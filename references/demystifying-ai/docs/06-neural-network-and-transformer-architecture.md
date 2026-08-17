# From Artificial Neurons to Decoder-Only Transformers

[<- Demystifying AI visual primer](../README.md) · [Parameters and activations](03-parameters-and-activations.md) · [Self-attention](04-self-attention.md)

This chapter connects two levels of the same system. The first diagram builds a
neural network from arithmetic, parameters, and training loops. The second uses
those mechanisms inside a **decoder-only, causal, pre-normalized Transformer**
representative of GPT-style language models. It is one Transformer design, not a
claim that every Transformer has exactly this structure.

<div class="zoomable-diagram" tabindex="0" role="region" aria-label="Scrollable deep-neural-network architecture and training-flow diagram">
  <a href="../images/deep_neural_network_training_flows.png" title="Open the full-resolution diagram">
    <img src="../images/deep_neural_network_training_flows.png" alt="Deep neural network from artificial neurons and dense layers through forward propagation, loss, backpropagation, parameter updates, batches, and epochs">
  </a>
</div>

*Diagram 1 — General neural-network mechanics. [Open the full-resolution image](../images/deep_neural_network_training_flows.png) for browser zoom, or scroll the viewport above.*

<div class="zoomable-diagram" tabindex="0" role="region" aria-label="Scrollable decoder-only Transformer architecture and training-flow diagram">
  <a href="../images/decoder_only_transformer_architecture_training_flows.png" title="Open the full-resolution diagram">
    <img src="../images/decoder_only_transformer_architecture_training_flows.png" alt="Decoder-only causal pre-normalized Transformer showing tokenization, embeddings, repeated blocks, self-attention, feed-forward networks, residual streams, language-model head, next-token loss, and training flows">
  </a>
</div>

*Diagram 2 — GPT-style assembly of the same mechanics. [Open the full-resolution image](../images/decoder_only_transformer_architecture_training_flows.png) for browser zoom, or scroll the viewport above.*

## 1. The two levels of understanding

There are two different goals:

1. **Mechanical understanding:** enumerate every tensor, shape, arithmetic
   operation, dependency, and stored intermediate. This level can be exact.
2. **Representational interpretation:** explain what millions or billions of
   learned values collectively encode. Individual features can sometimes be
   analyzed, but emergent representations are only partially interpretable.

Nothing in either forward or backward propagation is magic. The incomplete part
is our semantic interpretation of the solution learned by all those operations.

## 2. Artificial neuron and classical perceptron

The lower-left panel of Diagram 1 defines one artificial neuron:

```text
z = sum_i(w_i * x_i) + b
y = phi(z)
```

- `x_i`: input value or activation; not a parameter.
- `w_i`: trainable weight attached to this neuron.
- `b`: trainable bias.
- `z`: pre-activation weighted sum.
- `phi`: activation function.
- `y`: neuron output activation.

A classical perceptron traditionally uses a threshold or step function. Modern
neurons often use ReLU, sigmoid, tanh, GELU, or SiLU. Nonlinearity matters:

```text
(xW1 + b1)W2 + b2 = x(W1W2) + (b1W2 + b2)
```

Therefore, composing affine layers without nonlinear activations is still only
one affine transformation, regardless of how many layers are written in code.

## 3. Dense or fully connected layer

A dense layer runs many neurons in parallel. Every output neuron receives all
inputs, but owns a different weight vector and bias:

```text
y = xW + b

x: (1, 4)
W: (4, 3)
b: (3,)
y: (1, 3)
```

The three columns of `W` belong to three output neurons. The layer contains
`4 * 3 + 3 = 15` parameters: 12 weights and 3 biases. Here, “outputs” means
units in this particular layer, not necessarily final model outputs. The work is
approximately `4 * 3` multiply-accumulate contributions, not `4^3`.

## 4. What flows and what stays local

Diagram 1 uses distinct arrows deliberately:

- input data and activations flow forward;
- `W` and `b` stay attached to their owning layer;
- a scalar loss is computed from the final prediction and target;
- gradients propagate backward;
- the optimizer uses gradients to update local parameters.

For two layers:

```text
a1 = phi(xW1 + b1)
y_hat = a1W2 + b2
L = loss(y_hat, y)
```

`W1` never travels to layer 2, but it influences `L` indirectly because changing
`W1` changes `a1`, which changes every later result.

## 5. Forward propagation

For a batch of `B` samples, let the first layer have `D` inputs and `H` outputs:

| Stage | Input | Operation and local parameters | Output | Saved for backward? |
|---|---|---|---|---|
| Input | raw features `(B, D)` | optional preprocessing; no model parameter | `x: (B, D)` | commonly |
| Dense 1 | `x: (B, D)` | `z1 = xW1 + b1`, `W1: (D,H)`, `b1: (H,)` | `z1: (B,H)` | yes |
| Activation | `z1: (B,H)` | `a1 = phi(z1)` | `a1: (B,H)` | yes or recomputed |
| Later layers | prior activation | repeated local transforms | new activations | as needed |
| Prediction | final activation | output projection | `y_hat` | usually |
| Loss | prediction + target | loss reduction | scalar `L` | graph metadata |

An **activation** is a computed intermediate for particular input data. It is not
a learned parameter. Frameworks may save activations or recompute them during
backward; that is a runtime memory/performance choice.

## 6. Loss function

A loss converts prediction error into a scalar objective. In classification, a
model produces **logits** (unnormalized class scores), softmax maps them to
probabilities, and cross-entropy penalizes low probability on the correct class.

| Term | Meaning |
|---|---|
| Prediction | model output or selected class/token |
| Correct label | target supplied by the dataset |
| Loss | differentiable scalar optimized during training |
| Accuracy | fraction predicted correctly; usually an evaluation metric |

Accuracy is not generally differentiable, so the optimizer follows loss
gradients rather than “accuracy gradients.”

## 7. Backpropagation and gradients

A gradient measures sensitivity: `dL/dparameter` asks how an infinitesimal
parameter change changes loss. For the two-layer example:

```text
dL/dW1 =
  dL/dy_hat * dy_hat/da1 * da1/dz1 * dz1/dW1
```

Backpropagation traverses the computation graph in reverse topological order,
combines upstream sensitivity with each operation's local derivative, and sums
gradient contributions where a value fans out to multiple consumers. It
**computes gradients**; it does not update parameters.

## 8. Optimizer and learning

Basic stochastic gradient descent applies:

```text
W <- W - learning_rate * dL/dW
```

The learning rate controls step size. Momentum maintains a moving direction to
smooth updates. Adam maintains first- and second-moment estimates for each
parameter and scales updates adaptively. Those moment tensors are **optimizer
state**. One update does not make a parameter “learned”; learning is the
accumulated effect of many data-driven updates.

## 9. Batches, steps and epochs

- **Sample:** one training example.
- **Batch/mini-batch:** samples processed together.
- **Training step:** normally forward, loss, backward, and one optimizer update.
- **Epoch:** one complete pass through the training dataset.

```text
Training
└── Epochs
    └── Batches
        └── Forward -> Loss -> Backward -> Update
```

With 150 samples and batch size 30, one epoch has 5 batches and normally 5
updates. If divisibility fails, the final batch may be smaller unless the loader
drops it. Gradient accumulation is an explicit exception: several mini-batches
can contribute to one update.

## 10. Width, depth and parameter count

- **Width:** units or channels in a layer.
- **Depth:** sequential learned transformations.
- **Parameter count:** learned scalar values.
- **Activation memory:** temporary forward values needed by later work/backward.

A single layer can be extremely wide without being deep. More depth changes the
composition of functions and can express hierarchical transformations, but also
lengthens gradient paths and makes optimization harder.

## 11. Residual or skip connections

Diagram 1 shows a residual path:

```text
output = input + transformation(input)
```

The input remains available as a **residual stream**, the transform can learn a
modification rather than a total replacement, and gradients gain a short path
through addition. Shapes must be compatible or a projection must align them.
Residual connections help substantially; they do not eliminate every stability
or optimization problem.

## 12. Normalization

Dataset feature scaling preprocesses external features using dataset statistics.
LayerNorm or RMSNorm operates inside the network on current activations.
Normalization stabilizes internal magnitudes; LayerNorm commonly has learned
scale and bias, while RMSNorm commonly has learned scale and may omit bias. These
small learned tensors are local model parameters.

## 13. Transition from generic neural networks to Transformers

Transformers are neural networks. Diagram 2 reuses dense projections,
activations, normalization, residual connections, losses, backpropagation,
optimizers, batches, and epochs from Diagram 1. The defining addition is
attention-based **token mixing**. Dense layers still perform most projections,
but attention decides how token positions exchange information.

## 14. Input text, tokenization and embeddings

Diagram 2 begins:

```text
input text -> tokenizer -> token IDs -> embedding lookup
           -> positional information -> X
```

Let `B` be batch size, `T` sequence length, and `d_model` residual-stream width:

```text
token IDs: (B, T)               # integer table indices
X:         (B, T, d_model)      # floating-point vectors
```

Token embeddings are learned table rows. Positional information supplies order,
through learned vectors, fixed functions, rotations such as RoPE, or another
design. Token IDs themselves are not semantic vectors.

## 15. Decoder-only and causal

The depicted model is **decoder-only**: one causal stack predicts subsequent
tokens. Position `t` may attend only to positions `<= t`; a causal mask blocks
future positions, preventing training-time answer leakage.

- Encoder-only models build bidirectional representations.
- Encoder-decoder models encode an input then decode an output with cross-attention.
- Decoder-only models autoregressively continue a sequence.

This chapter stays focused on the third family.

## 16. Transformer depth

```text
X0 -> Block1 -> Block2 -> ... -> BlockN
```

Stacking `N` blocks produces model depth. Activations cross block boundaries.
Each block normally owns separate parameters unless the architecture explicitly
ties parameters across blocks.

## 17. Pre-normalized Transformer block

Follow the enlarged block in Diagram 2:

1. The residual stream enters.
2. Normalization produces attention input.
3. Causal multi-head self-attention mixes tokens.
4. An output projection returns to `d_model`.
5. Residual addition updates the stream.
6. A second normalization produces MLP input.
7. The feed-forward network transforms each position.
8. A second residual addition updates the stream.

It is **pre-norm** because normalization precedes each sublayer. Post-norm and
other variants exist.

## 18. Q, K and V projections

Given `X: (B,T,d_model)`:

```text
Q = XWQ
K = XWK
V = XWV
```

These are dense layers applied at every token position. Intuitively, queries
describe what a token seeks, keys what positions offer for matching, and values
what information can be collected. Exactly, they are learned matrix projections;
the intuition must not replace the tensor operations.

## 19. Multi-head causal self-attention

For each head:

```text
Attention(Q,K,V) = softmax((Q K^T / sqrt(d_k)) + causal_mask) V
```

1. `QK^T` computes pairwise token-position scores.
2. Division by `sqrt(d_k)` controls score magnitude.
3. The causal mask makes future scores unavailable.
4. Softmax normalizes allowed scores into weights.
5. Multiplication by `V` forms weighted mixtures.

```text
Q, K, V per head: (B, T, d_head)
attention scores:  (B, heads, T, T)
attention output:  (B, T, d_model)
```

The score tensor grows quadratically with `T`, so it can dominate activation
memory for long sequences.

## 20. Attention heads

Commonly, `d_head = d_model / number_of_heads`. Each head uses a different
learned projection space. Head outputs concatenate along the feature dimension
and pass through `WO`. Heads can specialize, but not every head has one stable,
human-readable meaning.

## 21. Feed-forward network

The Transformer MLP applies independently to every token position while sharing
the same weights across positions:

```text
h = activation(xW_up + b_up)
y = hW_down + b_down
```

Attention mixes information across positions. The MLP transforms each resulting
position. Q/K/V, output, up, and down projections are all dense layers; the
attention weighting operation itself is not a dense layer.

## 22. Parameter ownership in a Transformer block

| Owner | Typical local parameters |
|---|---|
| Attention | `WQ`, `WK`, `WV`, `WO` (and architecture-dependent biases) |
| MLP | `W_up`, `b_up`, `W_down`, `b_down` |
| Normalization | learned scale and, depending on type, bias |

Every repeated block normally owns its own copies. Parameters stay local while
activations move forward and parameter gradients move backward.

## 23. Final normalization, LM head and logits

Diagram 2 ends:

```text
block output -> final normalization -> LM head -> logits
```

```text
logits: (B, T, vocabulary_size)
```

Each logit is an unnormalized score for one vocabulary item. Softmax produces
probabilities. During inference, argmax or a sampling strategy selects a token.

## 24. Next-token training

Teacher forcing shifts targets by one position:

```text
input:  [token1, token2, token3]
target: [token2, token3, token4]
```

The model predicts the next token at all positions in parallel during training;
the causal mask prevents each position from seeing its target or later tokens.
Cross-entropy compares every predicted distribution with its correct next token.

## 25. Training versus inference

| Training | Inference |
|---|---|
| forward pass and cross-entropy | forward computation |
| backpropagation | no backward |
| optimizer updates | frozen parameters |
| batches over many steps/epochs | autoregressive token generation |
| stores training activations/gradients/state | may cache K/V from earlier tokens |

K/V caching avoids recomputing earlier key and value representations during
generation. Diagram 2 primarily describes training.

## 26. What makes a Transformer trainable at depth

Self-attention alone does not enable depth. Practical training also depends on
residual paths, normalization, careful initialization, nonlinear activations,
stable optimizers, and sometimes gradient clipping. Large models additionally
require distributed computation and communication. These measures manage, but
do not abolish, vanishing or exploding gradients.

## 27. Parameters versus activations versus gradients versus optimizer state

| Category | Definition | Lifetime | Example | Learned? | Checkpoint? | Memory implication |
|---|---|---|---|---|---|---|
| Parameters | persistent model values | across inputs | `WQ`, dense weights | yes | yes | required for train and infer |
| Activations | computed forward values | batch/sequence | embeddings, Q/K/V, block outputs | no | normally no | large training cost; grows with batch/sequence |
| Gradients | loss sensitivities | backward/update interval | `dL/dWQ` | no | normally no | roughly parameter-shaped during training |
| Optimizer state | update statistics | across steps | Adam first/second moments | no, but evolves | often yes for resume | often multiple parameter-sized tensors |

Training therefore usually uses much more memory than inference: it retains
activations, gradients, and optimizer state in addition to parameters.

### Compile time versus runtime

Source parsing, graph capture, kernel selection, and machine-code generation may
happen at **compile time** (or just-in-time compilation). Inputs, activations,
loss, backward, and optimizer updates happen at **runtime**. Compiled kernels are
instructions; parameters are runtime data. Compilation can specialize execution
without learning model weights.

## 28. Common misconceptions

| Misconception | Correction |
|---|---|
| Parameters flow through the network. | Activations flow; parameters stay with their owners. |
| A parameter is learned after one pass. | Learning accumulates across many updates. |
| One epoch means one optimizer update. | An epoch normally contains many batch updates. |
| Every DataFrame column is automatically a feature. | The input pipeline explicitly selects and encodes features. |
| A dense layer is nonlinear by itself. | `xW+b` is affine; a nonlinear activation must follow. |
| Attention is just another dense layer. | Dense projections create Q/K/V; score-based token mixing is distinct. |
| Transformers created deep learning. | Deep neural networks predate Transformers. |
| Every Transformer is decoder-only. | Encoder-only and encoder-decoder families also exist. |
| Softmax probabilities are parameters. | They are activations computed from logits. |
| Accuracy and loss are the same. | Loss is optimized; accuracy is usually an evaluation metric. |
| More layers always improve a model. | Depth adds capacity and optimization cost; results depend on design/data/training. |
| Every attention head has a simple role. | Head behavior can be distributed, contextual, and hard to interpret. |

## 29. Reading the two diagrams together

1. Start at the artificial neuron in Diagram 1.
2. Expand it into a dense layer.
3. Follow activations through the generic network.
4. Follow gradients back from loss.
5. Locate batches, steps, and epochs.
6. Move to token embeddings in Diagram 2.
7. Enter one Transformer block.
8. Separate dense projections from token mixing.
9. Follow the residual stream through repeated blocks.
10. End at logits, next-token loss, gradients, and local parameter updates.

## 30. Mastery exercises

1. **Count parameters:** For a `5 -> 7` dense layer, count weights and biases.
   *Check:* `5*7+7=42`.
2. **Derive shapes:** For `x:(8,5)` and `W:(5,7)`, derive `y`. *Check:* `(8,7)`.
3. **One neuron:** With `x=[2,-1]`, `w=[3,4]`, `b=1`, ReLU, compute output.
   *Check:* `z=3`, output `3`.
4. **Tiny forward pass:** Evaluate two affine/ReLU stages by hand and compare with
   a NumPy implementation.
5. **Simple gradient:** For `L=(wx-y)^2`, derive `dL/dw`.
   *Check:* `2(wx-y)x`; compare with finite differences.
6. **Batch and epoch:** For 103 samples and batch size 20, count batches. *Check:*
   6 with a final batch of 3, normally 6 updates.
7. **Q/K/V:** With `B=2,T=16,d_model=64,heads=4`, give each head's Q/K/V shape.
   *Check:* `(2,16,16)`.
8. **Scores:** For exercise 7, give score shape. *Check:* `(2,4,16,16)`.
9. **Classify values:** Label token IDs, `WQ`, Q, `dL/dWQ`, and Adam moments as
   input, parameter, activation, gradient, or optimizer state.
10. **Train vs infer:** List which tensors disappear when switching to inference.
11. **Implement NumPy neuron/dense:** verify output shapes and the 15-parameter
    `4 -> 3` example.
12. **Implement causal attention:** construct scores, upper-triangular mask,
    stable softmax, and `weights @ V` in NumPy; assert every masked probability is
    zero and compare against PyTorch
    `scaled_dot_product_attention(..., is_causal=True)` in float64.

For numerical gradient validation, use centered finite differences
`(L(w+eps)-L(w-eps))/(2*eps)` and expect error to shrink until floating-point
roundoff dominates.

## 31. Architecture checklist

- [ ] I can state what flows forward and what flows backward.
- [ ] I can identify what remains local and what the optimizer updates.
- [ ] I know when updates occur and what defines an epoch.
- [ ] I can construct a dense layer from neurons and count its parameters.
- [ ] I can identify the dense projections inside a Transformer.
- [ ] I can explain why attention, rather than an MLP, mixes tokens.
- [ ] I can explain causal masking without saying “the model just knows.”
- [ ] I can trace the residual stream and define Transformer depth.
- [ ] I can explain why training memory exceeds inference memory.
- [ ] I can distinguish parameters, activations, gradients, optimizer state,
      compile-time work, and runtime work.
