# Week 5 Notes — Architecture, Prompting, Data Prep

Fill these in as you study. Keep bullets short; if it doesn't fit on a flashcard, it's two bullets.

## Transformer internals
### Decoder-only block anatomy
-
-
### Attention math (Q/K/V, scaling, causal mask)
-
-
### MHA vs MQA vs GQA
-
-
### Modern deltas vs 2017 paper (pre-norm, RMSNorm, SwiGLU)
-
-

## KV cache
### Why it exists / what it stores
-
-
### Size formula + worked example (Llama-3-8B @ 8k)
-
-
### Link to paged attention / vLLM (week 7 preview)
-

## Positional encodings
### Sinusoidal / learned absolute
-
### RoPE (mechanism, context extension: NTK, YaRN)
-
-
### ALiBi
-

## Tokenization
### BPE / byte-level BPE
-
-
### SentencePiece / Unigram
-
### Vocab size trade-offs; multilingual & code effects
-
-

## Prompt engineering
### Zero-shot vs few-shot (ICL) — when each wins
-
-
### CoT, zero-shot CoT, self-consistency
-
-
### System prompts & chat templates — why templates matter
-
-
### Sampling: temperature, top-k, top-p, penalties
-
-
### Structured / constrained output (JSON mode, guided decoding)
-

## RAG
### Pipeline stages (ingest → chunk → embed → retrieve → rerank → generate)
-
-
### RAG vs fine-tuning decision rule
-
### Chunking & overlap; embedding vs generator models
-
-
### Reranking (cross-encoder); lost-in-the-middle
-
### NVIDIA mapping: NeMo Retriever, NIM embed/rerank microservices
-

## Data preparation
### Curation pipeline stages, in order
-
-
### Dedup: exact vs fuzzy (MinHash-LSH) vs semantic — and why
-
-
### Quality filtering: heuristic vs model-based
-
### PII redaction, toxicity filtering
-
### NeMo Curator — capabilities to name-drop
-
-
### SFT data formats; quality > quantity (LIMA)
-
### Tokenizer training: when and what breaks
-

## Misses from self-check (log here)
-
