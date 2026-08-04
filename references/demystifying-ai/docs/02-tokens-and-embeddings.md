# 2. Tokens, token IDs, and embeddings

[<- Demystifying AI visual primer](../README.md) · [Reference shelf](../../README.md)

## A token is not necessarily a word

A token may be:

- a whole word
- part of a word
- punctuation
- whitespace combined with text
- a byte or byte sequence
- a special control symbol

Example:

```text
"The dog runs."
        ↓ tokenizer
["The", " dog", " runs", "."]
```

## Token ID

Each token in a tokenizer vocabulary has an integer ID.

```text
"dog" → 4317
```

The number `4317` is only an index. It does not mean “dog” mathematically and has no semantic magnitude.

## Embedding table

The model stores a trainable matrix:

\[
E\in\mathbb{R}^{V\times d_{\text{model}}}
\]

where:

- \(V\) = vocabulary size
- \(d_{\text{model}}\) = embedding dimension

If:

\[
V=50{,}000,\qquad d_{\text{model}}=512
\]

then:

\[
E\in\mathbb{R}^{50{,}000\times512}
\]

The table has:

- 50,000 rows
- 512 columns

## Lookup

```python
embedding = embedding_table[4317]
```

This means:

> select row 4317

It does **not** mean “create a vector with 4317 elements.”

The returned vector has exactly \(d_{\text{model}}\) components:

\[
E[4317]\in\mathbb{R}^{512}
\]

## Is \(d_{\text{model}}\) fixed?

For a given model, yes.

Every token embedding and every hidden state in that model normally has the same width:

\[
d_{\text{model}}
\]

Different models may choose different values.

## Embedding as the AI-visible representation

The neural network does not process the string `"dog"` directly. It processes a vector such as:

\[
[0.72,-0.18,0.23,\ldots,0.44]
\]

The stages are:

```text
Human-readable token
        ↓
Token ID
        ↓
Embedding vector
```

## Initial versus contextualized representation

The same token ID retrieves the same initial embedding within a fixed model.

However, after Transformer layers process the sequence, its representation becomes context-dependent.

Example:

```text
"I ate a peach."
"Fishing is prohibited."
```

A token with the same surface form may acquire different contextual representations because it attends to different surrounding tokens.
