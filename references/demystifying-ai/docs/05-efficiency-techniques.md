# 5. FlashAttention, sparsity, and efficiency

[<- Demystifying AI visual primer](../README.md) · [Reference shelf](../../README.md)

## Classical attention memory problem

For \(N\) tokens, the attention-score matrix has shape:

\[
N\times N
\]

Therefore its size grows quadratically:

\[
O(N^2)
\]

The classical implementation may write large intermediate matrices to GPU high-bandwidth memory and read them back multiple times.

## FlashAttention

FlashAttention computes the same mathematical attention result while organizing the work in smaller blocks.

It:

1. loads blocks of \(Q\), \(K\), and \(V\) into fast on-chip memory
2. computes partial scores
3. updates the softmax online
4. updates the output incrementally
5. avoids materializing the complete attention matrix in main GPU memory

FlashAttention is therefore primarily an **I/O-aware algorithm**.

It reduces:

- HBM reads and writes
- intermediate-memory usage
- execution time
- energy spent moving data

It generally does **not** eliminate the quadratic number of token comparisons.

## Structured sparsity

Structured sparsity removes values using a hardware-friendly pattern.

Example: 2:4 sparsity

\[
[0.8,\ 0,\ -0.3,\ 0]
\]

Two of every four weights are non-zero.

This can reduce:

- stored values
- memory traffic
- multiply operations

## Is sparsity compression?

Yes, but it is compute-aware compression.

Unlike ZIP compression, supported hardware can often operate directly on the sparse representation without first reconstructing the dense matrix.

## Combining techniques

Starting from FP16 dense weights:

- FP8 halves the bits per value
- FP4 quarters the bits per value
- 2:4 sparsity retains half the values

Conceptually:

\[
\text{FP4 + 2:4 sparsity}
\]

can reduce raw weight storage toward one eighth of FP16 dense storage, before metadata and scale overhead.
