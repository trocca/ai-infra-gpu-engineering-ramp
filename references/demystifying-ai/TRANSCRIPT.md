# Conversation Transcript — Demystifying AI

[<- Demystifying AI visual primer](README.md) · [Reference shelf](../README.md)

This file preserves the conceptual progression of the discussion in condensed question-and-answer form.

## exaFLOPS

**Question:** What is an exaFLOP?

**Answer:** One exaFLOPS means \(10^{18}\) floating-point operations per second. The precision format matters, because FP64, FP32, FP16, FP8, and FP4 throughput can differ dramatically.

## FP4 and FP8

**Question:** What are FP4 and FP8?

**Answer:** They are low-precision numeric formats used to reduce storage, bandwidth, and compute cost. FP8 uses 8 bits per value; FP4 uses 4 bits and requires more aggressive quantization techniques.

## Structured sparsity

**Question:** Is structured sparsity basically compression?

**Answer:** Yes, but it is compression designed for direct hardware execution. With 2:4 sparsity, only two values in each group of four remain non-zero.

## FlashAttention

**Question:** What is FlashAttention?

**Answer:** It is a more memory-efficient way to compute ordinary attention. It tiles the computation and avoids writing the entire \(N\times N\) attention matrix to GPU memory.

## Classical attention

**Question:** What are classical attention and self-attention?

**Answer:** Self-attention lets each token update its representation by assigning weights to tokens in the same sequence.

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

## Transpose

**Question:** What does the superscript \(T\) mean?

**Answer:** \(T\) denotes matrix transpose. It exchanges rows and columns so that \(QK^T\) produces a token-by-token score matrix.

## Vector components

**Question:** Why do Query and Key vectors have \(d\) components?

**Answer:** A token needs a multi-dimensional representation. The dot product compares corresponding Query and Key components to produce a similarity score.

## Self-attention logic

**Question:** Why is it called self-attention?

**Answer:** Because Query, Key, and Value all originate from the same sequence. “Self” does not mean that a token attends only to itself.

## Embeddings

**Question:** What is an embedding?

**Answer:** An embedding is the dense numerical vector used by the neural network to represent a token.

```text
Text token → token ID → embedding vector
```

## Embedding lookup

**Question:** Does `embedding_table[4317]` mean a vector of 4317 elements?

**Answer:** No. `4317` is the row index. The returned row has \(d_{\text{model}}\) elements.

## Fixed width

**Question:** Is \(d_{\text{model}}\) a fixed number of columns?

**Answer:** Yes, within a given model. It is the width of token embeddings and hidden representations.

## Parameters

**Question:** How do parameters relate to all this?

**Answer:** The embedding table and projection matrices are collections of learned parameters. Query, Key, Value, and attention scores are temporary activations computed from those parameters and the current input.

## Real numbers notation

**Question:** Does \(\mathbb{R}\) mean real numbers?

**Answer:** Yes. For example:

\[
E\in\mathbb{R}^{V\times d_{\text{model}}}
\]

means that \(E\) is a \(V\)-by-\(d_{\text{model}}\) matrix whose entries are real numbers.
