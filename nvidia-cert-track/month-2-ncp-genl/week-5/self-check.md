# Week 5 Self-Check — Architecture, Prompting, Data Prep

Answer out loud or on paper *before* opening the details block. Score yourself; ≥80% to pass the week.

**1. Why is the dot product in attention scaled by √d_k?**

<details><summary>Answer</summary>
Without scaling, dot products grow with dimension d_k, pushing softmax into saturated regions with near-zero gradients and overly peaked attention. Dividing by √d_k keeps logits at roughly unit variance so softmax stays well-conditioned.
</details>

**2. In a decoder-only LLM, what exactly does the causal mask prevent, and where is it applied?**

<details><summary>Answer</summary>
It prevents each token from attending to future positions. It is applied to the attention score matrix (QKᵀ/√d_k) before softmax, by setting masked positions to −∞ so they get zero attention weight.
</details>

**3. Explain the difference between MHA, MQA, and GQA, and the main *deployment* reason GQA is popular.**

<details><summary>Answer</summary>
MHA: every head has its own K and V. MQA: all heads share a single K/V head. GQA: groups of query heads share K/V heads (a middle ground). Deployment reason: KV-cache size scales with the number of KV heads, so GQA cuts KV-cache memory several-fold with minimal quality loss, allowing bigger batches / longer contexts at inference.
</details>

**4. Roughly how big is the KV cache for a 32-layer model with 8 KV heads, head_dim 128, at 8192 context, batch 1, FP16?**

<details><summary>Answer</summary>
2 (K and V) × 32 layers × 8 heads × 128 dim × 8192 tokens × 2 bytes ≈ 1.07 GB per sequence. Key point: it scales linearly with sequence length and batch size, and dominates memory at high concurrency.
</details>

**5. What is RoPE and why does it enable context-length extension tricks?**

<details><summary>Answer</summary>
Rotary Position Embedding rotates Q and K vectors by position-dependent angles, so attention scores depend on *relative* position. Because position is encoded as rotation frequencies, you can rescale those frequencies (position interpolation, NTK-aware scaling, YaRN) to stretch beyond the trained context with little or no fine-tuning.
</details>

**6. Pre-norm vs post-norm: which do modern LLMs use and why?**

<details><summary>Answer</summary>
Pre-norm (normalization before attention/MLP, typically RMSNorm). It keeps gradients stable in very deep stacks, making training converge reliably without careful warmup; the original post-norm transformer is harder to train at depth.
</details>

**7. Why is byte-level BPE preferred over word-level vocabularies for LLMs?**

<details><summary>Answer</summary>
No out-of-vocabulary tokens — any string decomposes into bytes, so the model can represent arbitrary text (code, typos, any language). It balances vocab size vs sequence length: frequent substrings become single tokens while rare strings fall back to bytes.
</details>

**8. A prompt performs well in testing, but in production the model ignores the system instructions when users paste long documents. Name the failure and two mitigations.**

<details><summary>Answer</summary>
Instruction dilution / lost-in-the-middle (and possibly indirect prompt injection from pasted content). Mitigations: restate critical instructions near the end of the prompt or in the correct system role; clearly delimit untrusted content; shorten/summarize the pasted context; use a model or serving path that enforces system-prompt priority; add guardrails validation on output.
</details>

**9. When does few-shot prompting beat zero-shot, and what is a known sensitivity of few-shot?**

<details><summary>Answer</summary>
Few-shot wins when the task format is unusual or ambiguous (the examples teach format and label space) and for pattern-matching tasks the model wasn't instruction-tuned on. Sensitivity: performance varies with example choice, order, and format (recency and majority-label bias).
</details>

**10. What is self-consistency and when is it worth its cost?**

<details><summary>Answer</summary>
Sample multiple chain-of-thought completions at temperature > 0 and take a majority vote on the final answer. Worth it for high-stakes multi-step reasoning (math, logic) where accuracy justifies k× inference cost; pointless for extraction or short factual lookups.
</details>

**11. Your extraction pipeline needs deterministic, parseable output. Which sampling settings and decoding features do you use?**

<details><summary>Answer</summary>
Temperature 0 (greedy) — or very low temp — plus constrained/guided decoding (JSON schema / grammar, e.g. vLLM guided_json) to guarantee syntactic validity. Few-shot examples of the exact output format also help.
</details>

**12. RAG vs fine-tuning: a customer wants the model to answer from a knowledge base updated daily AND to always respond in their house style. What do you recommend?**

<details><summary>Answer</summary>
Both, split by purpose: RAG for the daily-changing knowledge (fresh, attributable, no retraining), and a light fine-tune (e.g. LoRA) or a strong system prompt for the persistent style/format behavior. Fine-tuning alone is wrong for daily-changing facts; RAG alone may not hold style reliably.
</details>

**13. Why add a reranker after vector retrieval instead of just increasing top-k?**

<details><summary>Answer</summary>
Bi-encoder vector search is fast but coarse (query and document embedded independently). A cross-encoder reranker scores query+document jointly, giving much better precision on the handful of chunks that actually enter the prompt. Raising top-k alone adds noise, cost, and lost-in-the-middle risk.
</details>

**14. Name the stages of a pretraining data-curation pipeline in a sensible order.**

<details><summary>Answer</summary>
Acquisition/extraction (e.g. HTML → text) → language identification → heuristic quality filtering (length, symbol ratios, boilerplate) → model-based quality filtering → deduplication (exact, then fuzzy/semantic) → PII redaction and toxicity/unsafe-content filtering → tokenization and sequence packing. (Exact ordering of filter vs dedup can vary; dedup before final filtering is common to save compute.)
</details>

**15. Exact vs fuzzy vs semantic dedup — what does each catch, and what technique implements fuzzy dedup at scale?**

<details><summary>Answer</summary>
Exact: identical documents (hashing). Fuzzy: near-duplicates with small edits — implemented at scale with MinHash signatures + Locality-Sensitive Hashing (LSH) to approximate Jaccard similarity without all-pairs comparison. Semantic: same meaning, different wording — embedding clustering. NeMo Curator provides all three, GPU-accelerated.
</details>

**16. Why does duplicated training data hurt an LLM beyond wasted compute?**

<details><summary>Answer</summary>
It causes memorization/regurgitation of repeated text (privacy and copyright risk), skews the data distribution toward duplicated content, and inflates benchmark scores through train/test contamination when eval sets leak into training data.
</details>

**17. When would you train a new tokenizer instead of reusing the base model's, and what's the cost?**

<details><summary>Answer</summary>
When the target domain's text is badly served by the existing vocab — new languages/scripts, DNA/protein sequences, heavy domain jargon — causing token-inefficient encoding. Cost: the pretrained embedding and LM-head weights no longer match the vocabulary, so you must extend/reinitialize embeddings and retrain (or continue-pretrain); you can't just swap tokenizers under a trained model.
</details>

**18. What is NeMo Curator and why does NVIDIA pitch it over CPU-based pipelines?**

<details><summary>Answer</summary>
NVIDIA's data-curation framework for LLM training corpora: extraction, language ID, heuristic + classifier filtering, exact/fuzzy/semantic dedup, PII redaction, synthetic-data pipelines. Pitched because it runs GPU-accelerated (RAPIDS/Dask, scales to multi-node), turning weeks of CPU curation into hours — dedup especially benefits.
</details>

**19. A chat model gives strangely bad answers when called through a raw completion API with plain concatenated text. Most likely cause?**

<details><summary>Answer</summary>
The chat template wasn't applied — the model was trained to see special role tokens (e.g. <|im_start|>system … ) and without them it's out of distribution. Fix: apply the model's chat template (tokenizer.apply_chat_template or a chat endpoint).
</details>

**20. Why are prefill and decode phases of inference fundamentally different workloads?**

<details><summary>Answer</summary>
Prefill processes the whole prompt in parallel — large matrix multiplies, compute-bound, high Tensor Core utilization. Decode generates one token at a time per sequence, re-reading the entire KV cache and weights each step — memory-bandwidth-bound with low arithmetic intensity. This asymmetry motivates continuous batching, paged KV cache, and disaggregated prefill/decode serving (as in Dynamo).
</details>
