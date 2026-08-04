# Project handoff prompt for Codex or Claude Code

[<- Demystifying AI visual primer](../README.md) · [Reference shelf](../../README.md)

You are continuing a project named **Demystifying AI**.

The repository contains a beginner-friendly but technically accurate guide to Transformer architecture. Read `README.md`, all files under `docs/`, and the diagrams under `images/`.

Goals:

1. Preserve the current conceptual progression:
   - FLOPS and numeric precision
   - structured sparsity
   - FlashAttention
   - tokens and token IDs
   - embeddings
   - parameters versus activations
   - self-attention
2. Keep explanations visual, direct, and mathematically correct.
3. Avoid assuming that a token is always a whole word.
4. Distinguish carefully between:
   - token ID
   - embedding vector
   - model parameter
   - temporary activation
   - \(d_{\text{model}}\)
   - \(d_k\)
5. Build a polished educational website from the Markdown content.
6. Use the existing PNG diagrams as assets.
7. Add interactive examples only where they improve understanding.
8. Prefer a clean dark/light responsive design with readable mathematical notation.
9. Do not silently change technical claims; flag ambiguities or corrections.
10. Create a clear navigation path from beginner concepts to advanced efficiency techniques.

Suggested first task:

> Inspect the project, propose an information architecture for a small documentation website, and implement the initial version without removing any source Markdown files.
