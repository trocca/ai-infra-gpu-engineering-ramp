# 4. Self-attention

[<- Demystifying AI visual primer](../README.md) · [Reference shelf](../../README.md)

## Why “self”?

Self-attention means that Query, Key, and Value all come from the same input sequence.

Given an input representation matrix \(X\):

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

The word “self” refers to the source sequence, not to a token looking only at itself.

## What attention does

Each token updates its representation by assigning weights to tokens in the same sequence.

Example:

```text
"The dog eats the meat because it is hungry."
```

The representation of `it` can assign high attention to `dog` and low attention to unrelated tokens.

## Query, Key, and Value

A useful analogy:

- **Query**: What am I looking for?
- **Key**: What information do I represent?
- **Value**: What content should I contribute if selected?

## Why Query and Key have multiple components

A token is represented by a vector rather than one scalar.

For one attention head:

\[
q_i\in\mathbb{R}^{d_k}
\]

\[
k_j\in\mathbb{R}^{d_k}
\]

Their compatibility is measured with a dot product:

\[
q_i\cdot k_j
\]

Both vectors need the same number of components.

## Why \(K^T\)?

Suppose:

\[
Q\in\mathbb{R}^{N\times d_k}
\]

\[
K\in\mathbb{R}^{N\times d_k}
\]

Then:

\[
K^T\in\mathbb{R}^{d_k\times N}
\]

and:

\[
QK^T\in\mathbb{R}^{N\times N}
\]

The resulting matrix contains one score for every token-to-token pair.

The superscript \(T\) means **transpose**, not exponentiation.

## Full formula

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

### Step 1: scores

\[
S=QK^T
\]

### Step 2: scaling

\[
\frac{S}{\sqrt{d_k}}
\]

This helps keep values numerically stable.

### Step 3: softmax

Each row is converted into positive attention weights that sum to 1.

### Step 4: weighted Value combination

\[
A V
\]

where \(A\) is the attention-weight matrix.

Each output token becomes a weighted mixture of Value vectors from the sequence.

## Multi-head attention

Rather than computing one attention pattern, the model computes several heads in parallel.

Each head has its own learned projections:

\[
W_Q^{(h)},\ W_K^{(h)},\ W_V^{(h)}
\]

The head outputs are concatenated and projected back to \(d_{\text{model}}\).

## Is self-attention “inside one neuron”?

No.

A single neuron performs a weighted sum followed by an optional nonlinearity.

Self-attention is a coordinated matrix computation involving many learned units and many temporary activations.

![Self-attention flow](../images/self_attention_complete_flow_en.png)
