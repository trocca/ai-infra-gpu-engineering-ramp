# Week 5 Plan (Aug 10–14, 2026) — LLM Architecture, Prompt Engineering, Data Preparation

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

Exam coverage this week: **LLM Architecture 6% + Prompt Engineering 13% + Data Preparation 9% = 28%**

Each day ≈ 2 h: ~75 min study, ~30 min hands-on/notes, ~15 min flashcards (add your own cards to `flashcards.csv` as you go).

## Prerequisites before Monday

- Companion lesson: [Week 05 companion — transformer math, PyTorch architecture, and tokenizer support](../../../companion-lessons/week-05.md).
- Source reading: [HF Ultra-Scale Playbook — transformer memory and single-GPU training](../../../references/hf-ultrascale-playbook.md#week-5---transformer-memory-and-single-gpu-training).
- Math support: Q/K/V projection shapes, `softmax(QK^T/sqrt(d))V`, cross-entropy, and the KV-cache memory formula.
- Programming support: `torch.nn.Module`, tokenizer-to-token-ID flow, train vs eval mode, and causal masking.
- Gate: compute attention tensor shapes for `B=2`, `T=128`, `C=768`, `H=12` and identify the test that protects attention correctness.

---

## Day 1 (Mon) — Transformer internals
**Domain: LLM Architecture (6%)**

- (40 min) Re-read "Attention Is All You Need" with modern eyes; sketch a decoder-only block from memory: embeddings → [attention → MLP] × N → LM head. Note what modern LLMs changed: pre-norm (RMSNorm), SwiGLU MLP, no encoder.
- (35 min) Attention math: Q/K/V projections, scaled dot-product (`softmax(QKᵀ/√d_k)V`), why the √d_k scaling, causal masking. Multi-head vs **MQA** vs **GQA** — memorize *why* GQA exists (shrinks KV cache, near-MHA quality; used by Llama-3, Qwen2.5).
- (30 min) Hands-on: in a notebook, implement single-head scaled dot-product attention in ~15 lines of PyTorch on random tensors; verify output shape and causal mask behavior.
- (15 min) Flashcards.
- Resources: Jay Alammar "The Illustrated Transformer"; Lilian Weng "The Transformer Family v2"; nanoGPT `model.py` (read it — it's ~300 lines and exam-perfect).

## Day 2 (Tue) — KV cache, positional encodings, tokenization
**Domain: LLM Architecture (6%), sets up GPU Acceleration (14%) for week 7**

- (35 min) KV cache: why decode re-uses cached K/V, cache size formula `2 × layers × kv_heads × head_dim × seq_len × batch × bytes/param` — compute it for Llama-3-8B at 8k context and understand why it dominates memory at high batch. This is the *reason* paged attention exists (week 7) and why you saw KV-cache pressure in your vLLM demo.
- (30 min) Positional encodings: absolute sinusoidal/learned vs **RoPE** (rotates Q/K pairs; relative by construction; enables NTK/YaRN context extension) vs **ALiBi** (linear attention bias). Know which models use what (RoPE ≈ everything modern).
- (30 min) Tokenization: BPE and byte-level BPE, SentencePiece/Unigram, vocab-size trade-offs, why tokenizer choice affects multilingual/code performance and effective context. Play with https://tiktokenizer.vercel.app or `AutoTokenizer` on tricky strings (numbers, code, non-English).
- (10 min) Skim: prefill vs decode phases — one is compute-bound, one memory-bound. Just plant the seed; week 7 harvests it.
- (15 min) Flashcards.

## Day 3 (Wed) — Prompt engineering core
**Domain: Prompt Engineering (13%)**

- (35 min) Zero-shot vs few-shot (in-context learning): when few-shot helps, example ordering/formatting sensitivity, label bias. Chain-of-Thought: what it is, when it helps (multi-step reasoning), zero-shot CoT ("think step by step"), self-consistency (sample k chains, majority vote).
- (30 min) System prompts and chat templates: role separation, why the *template* (special tokens) matters as much as the words — a fine-tuned chat model prompted without its template degrades badly. Look at a real chat template in `tokenizer_config.json` (Qwen2.5, Llama-3).
- (30 min) Sampling parameters: temperature, top-k, top-p, repetition/frequency penalty, greedy vs sampling vs beam. Know the exam-style trade-offs: temp 0 for extraction/determinism, higher temp + top-p for creative tasks. Structured output: JSON mode, constrained/guided decoding (vLLM `guided_json`, outlines).
- (10 min) Prompt-injection preview: system-prompt override attempts, why delimiters and instruction hierarchy matter (full treatment week 8 day 3).
- (15 min) Flashcards.
- Resources: NVIDIA DLI prompt-engineering short course; OpenAI/Anthropic prompting guides (concepts transfer; exam is vendor-neutral on technique).

## Day 4 (Thu) — RAG fundamentals
**Domain: Prompt Engineering (13%) — RAG sits in this domain on this exam**

- (35 min) RAG pipeline end-to-end: ingest → chunk → embed → index (vector DB) → retrieve (dense/hybrid) → rerank → stuff context → generate. Why RAG vs fine-tuning: RAG for *knowledge* (fresh, attributable, revocable), FT for *behavior/format/style* — classic exam discriminator.
- (30 min) Details that get tested: chunking strategies (fixed, recursive, semantic; overlap), embedding models vs generator models are different models, cosine similarity, top-k retrieval, reranking (cross-encoder) as precision stage, "lost in the middle" long-context failure, hallucination mitigation via grounding + citation.
- (30 min) NVIDIA stack mapping: NeMo Retriever / NIM embedding + reranking microservices; know they exist and what layer each covers. Skim a NIM RAG reference architecture diagram.
- (10 min) Evaluation preview: RAG evaluated on faithfulness/groundedness + answer relevance + retrieval recall (ties into week 6 day 4).
- (15 min) Flashcards.
- Cross-ref demo repo: your NIM deployment demo is exactly the serving layer of this pipeline; be able to narrate "retriever NIM + LLM NIM on K8s" as one story.

## Day 5 (Fri) — Data preparation + week wrap
**Domain: Data Preparation (9%)**

- (40 min) Pretraining data pipeline: acquisition → language ID → heuristic quality filters (length, symbol ratio, boilerplate) → model-based quality filtering (classifier scores) → **dedup** (exact hash + fuzzy MinHash-LSH; semantic dedup with embeddings) → PII redaction → toxicity filtering → tokenization/packing. Why dedup matters: memorization, eval contamination, wasted compute.
- (30 min) **NeMo Curator**: know its role (GPU-accelerated, scales with Dask/RAPIDS) and its named capabilities: exact/fuzzy/semantic dedup, quality classifiers, PII redaction, synthetic-data pipelines. This is the NVIDIA-flavored answer to most data-prep questions.
- (20 min) Fine-tuning data & tokenizer topics: instruction-dataset formats (prompt/response pairs, chat JSONL), data quality > quantity for SFT (LIMA idea), training a tokenizer on in-domain corpora (when: new language/domain like DNA or code; cost: breaks compatibility with pretrained weights unless you retrain/extend embeddings).
- (15 min) Run `week-5/self-check.md` cold; log misses in notes.
- (15 min) Flashcards + fill exit criteria below.

## Exit criteria (Friday)

- [ ] I can sketch a decoder-only transformer block from memory and explain each component's job
- [ ] I can explain MHA vs MQA vs GQA and why GQA shrinks the KV cache
- [ ] I can compute a KV-cache size for a given model/context and explain why it drives inference memory
- [ ] I can explain RoPE in two sentences and name one context-extension technique built on it
- [ ] I can explain BPE and give one trade-off of small vs large vocab
- [ ] I can pick zero-shot / few-shot / CoT / self-consistency for a given scenario and justify it
- [ ] I can explain temperature vs top-p and choose sampling settings for extraction vs creative tasks
- [ ] I can draw a full RAG pipeline and say when to use RAG vs fine-tuning
- [ ] I can describe a pretraining data-curation pipeline and what NeMo Curator does at each stage
- [ ] I can explain exact vs fuzzy (MinHash) vs semantic dedup and why dedup matters
- [ ] Self-check score ≥ 80% (retake Monday if below)
