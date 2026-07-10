# Week 6 Plan (Aug 17–21, 2026) — Fine-Tuning + Evaluation

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

Exam coverage this week: **Fine-Tuning 13% + Evaluation 7% = 20%**, plus the LoRA lab (also feeds Model Optimization intuition).

## Prerequisites before Monday

- Companion lesson: [Week 06 companion — LoRA rank math, QLoRA, and evaluation support](../../../companion-lessons/week-06.md).
- Math support: rank, adapter parameter counts, `W' = W + (alpha/r) * B @ A`, and the capacity/update-scale trade-off.
- Programming support: freezing parameters, optimizer parameter groups, adapter state dicts, PEFT parity tests, and evaluation caveats.
- Gate: compute LoRA parameter counts for one `4096 x 4096` projection at `r=8`, `16`, and `64`.

---

## Day 1 (Mon) — Full fine-tuning vs PEFT; LoRA mechanics
**Domain: Fine-Tuning (13%)**

- (35 min) Full FT: what it costs (optimizer states — AdamW keeps ~2 extra FP32 copies, so ~16 bytes/param mixed-precision → a 7B model needs ~112 GB+ just for training state), catastrophic forgetting, when it's still right (large data, deep behavior change, you own the whole stack).
- (40 min) **LoRA math** — know this cold: freeze W, learn ΔW = B·A with A ∈ ℝ^{r×d}, B ∈ ℝ^{d×r} (B init to zero so training starts at identity), effective update scaled by **α/r**. Trainable params for one d×d projection ≈ 2·d·r vs d². Typical: r = 8–64, α = 2r, target modules = attention projections (q,k,v,o) ± MLP. Adapters are mergeable (W′ = W + (α/r)·BA) → zero inference overhead after merge.
- (20 min) Other PEFT for name recognition: prompt tuning / p-tuning (NVIDIA's NeMo heritage), prefix tuning, IA³, adapters. Know LoRA dominates in practice and *why* (quality, mergeability, tooling).
- (15 min) Flashcards.
- Resources: LoRA paper §4; HF PEFT conceptual docs; NeMo PEFT docs.

## Day 2 (Tue) — QLoRA, hyperparameters, tooling
**Domain: Fine-Tuning (13%)**

- (35 min) **QLoRA**: base weights quantized to **NF4** (4-bit NormalFloat, information-theoretically motivated for normal-ish weight distributions), **double quantization** (quantize the quantization constants), **paged optimizers** (spill optimizer state to CPU on OOM spikes); LoRA adapters stay in BF16 and gradients flow through the frozen 4-bit weights. Net effect: fine-tune a 7–8B model on a single 24 GB GPU.
- (30 min) Hyperparameters that get tested: learning rate ~1e-4–3e-4 for LoRA (10× higher than full FT's ~1e-5–2e-5), rank/alpha trade-offs, LoRA dropout, epochs (1–3 typical for SFT; more → overfit/forgetting), effective batch via gradient accumulation, warmup + cosine decay, BF16 vs FP16 (BF16 preferred: same range as FP32, no loss-scaling needed).
- (30 min) Tooling map: HF **PEFT** (adapters) + **TRL** (SFTTrainer, DPOTrainer) + bitsandbytes (NF4); NVIDIA side: **NeMo** framework fine-tuning (NeMo 2.0 recipes, NeMo-Aligner for RLHF/DPO). Skim `labs/lab-finetune-lora.md` so Friday's lab is execution, not discovery.
- (15 min) Flashcards.

## Day 3 (Wed) — Instruction tuning, RLHF, DPO
**Domain: Fine-Tuning (13%)**

- (30 min) SFT / instruction tuning: chat-format datasets (JSONL of role/content turns), loss masking on prompt tokens (train only on assistant responses), chat template consistency between train and inference, data quality > volume (LIMA: ~1k excellent examples can align a strong base model).
- (40 min) **RLHF pipeline** (InstructGPT): SFT → train reward model on human preference pairs → PPO optimization against RM with a KL penalty to the SFT policy (prevents reward hacking / drift). Know each stage's artifact and failure mode (RM overoptimization = reward hacking).
- (30 min) **DPO**: skips the reward model and RL loop — directly optimizes the policy on preference pairs with a classification-style loss (implicit reward = log-ratio vs reference model, β controls KL strength). Trade-off: simpler/cheaper/stabler than PPO, but offline (no exploration). Name-check successors: KTO (unpaired thumbs up/down), ORPO (no reference model), GRPO (used for reasoning RL). NVIDIA tooling: NeMo-Aligner implements SFT/RM/PPO/DPO.
- (20 min) Decision drill: for five scenarios (style transfer, domain knowledge, safety alignment, format compliance, preference polish) pick full FT / LoRA / RAG / DPO and justify. This is exactly how the exam asks it.

## Day 4 (Thu) — Evaluation
**Domain: Evaluation (7%)**

- (30 min) **Perplexity**: exp of average negative log-likelihood; measures language-modeling fit, not task ability; only comparable with the same tokenizer; drops after domain FT but says nothing about instruction quality.
- (35 min) Benchmarks — know what each measures: **MMLU** (broad knowledge, multiple choice), **HellaSwag** (commonsense completion), **GSM8K/MATH** (math reasoning), **HumanEval/MBPP** (code, pass@k), **TruthfulQA** (falsehood resistance), **MT-Bench** (multi-turn, judged), **Chatbot Arena** (human pairwise Elo), **IFEval** (instruction following). Contamination caveat: public benchmarks leak into training data.
- (30 min) **LLM-as-judge**: pairwise vs single-answer grading with rubrics; known biases — position bias (swap answer order and average), verbosity bias, self-preference bias; mitigate with rubrics, order swapping, strong judge model. Task metrics: ROUGE/BLEU (n-gram overlap — weak for open generation), exact match/F1 (extraction/QA), pass@k (code), plus RAG triad (faithfulness, answer relevance, context relevance). Tooling: lm-evaluation-harness, NeMo Evaluator.
- (15 min) Flashcards.

## Day 5 (Fri) — LoRA lab + week wrap
**Domains: Fine-Tuning (13%) hands-on**

- (~90 min) Run `labs/lab-finetune-lora.md` end-to-end on a rented single GPU (L4/A10G/4090 class, ~$0.30–0.80/hr). Deliverables: adapter size noted, before/after generations compared, merged model saved.
- (20 min) Run `week-6/self-check.md`; log misses in notes.
- (10 min) Exit criteria below.

## Exit criteria (Friday)

- [ ] I can write the LoRA update equation and explain r, α, target modules, and why B starts at zero
- [ ] I can explain why LoRA fine-tuning fits on one 24 GB GPU when full FT of the same model needs >100 GB (optimizer states!)
- [ ] I can explain QLoRA's three tricks: NF4, double quantization, paged optimizers
- [ ] I can state typical LoRA hyperparameters (lr, r, α, epochs) and what happens when each is wrong
- [ ] I can describe the three RLHF stages and each one's failure mode
- [ ] I can explain DPO vs PPO-RLHF in two sentences and pick between them for a scenario
- [ ] I can choose between full FT / LoRA / RAG / DPO for a given business scenario and defend it
- [ ] I can define perplexity precisely and name its two big limitations
- [ ] I can match 6+ benchmarks to what they measure
- [ ] I can name three LLM-as-judge biases and a mitigation for each
- [ ] Lab complete: I have an adapter, a merged model, and observed behavior change
- [ ] Self-check score ≥ 80%
