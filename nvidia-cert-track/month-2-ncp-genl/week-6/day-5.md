# Week 6 · Day 5 — LoRA lab, self-check, and ship day

[← Master Plan](../../../MASTER-PLAN.md) · [Week 6 overview](plan.md) · [← previous day](day-4.md) · [next day →](../week-7/day-1.md)

Friday, Aug 21 2026. The week where you *implemented* LoRA ends with you *operating* it like a working engineer: run the cert lab on a rented GPU, pass the closed-book self-check, then prove your from-scratch implementation is equivalent to HuggingFace PEFT and ship the writeup.

## Study block (2 h)

**Exam domain: Fine-Tuning (13%), hands-on** — per the [week plan](plan.md), today's study slot is the scheduled run of the cert lab, then the week's scoring ritual.

### Part 1 — Run the LoRA lab end-to-end (~90 min)

Run [labs/lab-finetune-lora.md](../labs/lab-finetune-lora.md) on a rented single GPU (L4 / A10G / 4090 class, ~$0.30–0.80/hr — under $2 total). You skimmed it Tuesday; today is execution. The lab is the exam's fine-tuning domain compressed into six numbered steps — as you run it, narrate each observation in exam language:

- **Baseline generations first** (`before.txt`) — no comparison, no lab. The before/after diff is the deliverable.
- **`trainable ≈ 1.18%`** printed by the trainer — that's Monday's arithmetic (2·d·r per projection, r=16, seven target modules) confirmed by someone else's code.
- **lr 2e-4, cosine, warmup 3%, grad accumulation → effective batch 16** — Tuesday's hyperparameter table, live.
- **Adapter ≈ 70–75 MB vs ~3 GB base** — the multi-tenant serving story in `ls -lh` form.
- **After-generations shift toward Alpaca's terse style but knowledge is unchanged** — the RAG-vs-FT discriminator made visible (week-5 day-4).
- **Merge → ~3.1 GB dense model, zero inference overhead** — W′ = W + (α/r)·BA, third time you've touched it this week (theory Monday, your `merge_()` yesterday, PEFT's `merge_and_unload()` today).

Deliverables to note down: adapter size, before/after comparison saved, merged model produced. Then **clean up and terminate the instance** — cost control is part of the skill. If time is tight, the QLoRA variant in the lab is optional: you already produced that comparison properly on Wednesday.

### Part 2 — Closed-book self-check + scoring (~30 min)

1. Run [self-check.md](self-check.md) **closed-book**. Score it; target ≥ 80%, below → Monday-morning retake before week 7 begins.
2. Tick the exit criteria in [plan.md](plan.md) — the list is aggressive this week (LoRA equation cold, QLoRA's three tricks, RLHF stages + failure modes, perplexity + limitations, 6+ benchmarks, 3 judge biases, lab complete). Unticked boxes get a named weekend slot.
3. Fill the **week 6 row in [PROGRESS.md](../../PROGRESS.md)**: topics done, labs done (`lab-finetune-lora` + the from-scratch project), self-check %, confidence /5.

### Mini-review — the five ideas that must survive the weekend

1. **ΔW = (α/r)·B·A**, B zero-init, target the attention projections; merge = fold into W, zero overhead.
2. **Full FT ≈ 16 bytes/param of training state**; LoRA pays it on ~1% of params — that's the whole 24 GB story.
3. **QLoRA = NF4 + double quantization + paged optimizers**; gradients flow *through* frozen 4-bit, never *into* it.
4. **SFT imitates, preference optimization ranks**; PPO needs an RM + KL anchor (else reward hacking); DPO gets there with a classification loss on pairs.
5. **PPL = exp(mean NLL)** — same tokenizer only, fit not ability; benchmarks match names to skills; judges have position/verbosity/self-preference biases.

### Read next (weekend, optional)

- Your own `RESULTS.md` from this afternoon — reading your writeup as a stranger is the best review.
- Raschka, *LLM evaluation* article — consolidates Thursday.
- Week 7 preview, 10 minutes max: skim [../week-7/plan.md](../week-7/plan.md) — quantization + GPU acceleration (31%!); your NF4 experience is the on-ramp.

### Quick check (week-closing composite)

1. Reproduce the lab's headline numbers from memory: trainable %, adapter size vs base, merged-model size.
2. Why is the lab's lr (2e-4) safe for LoRA but reckless for full FT?
3. The lab's after-generations changed style, not knowledge. Which week-5 concept does this demonstrate, and what would you use to change *knowledge*?
4. You forgot to terminate the cloud instance until Sunday. Which exam domain does this violate? (Trick question — answer anyway.)

<details><summary>Answers</summary>

1. Trainable ≈ **1.18%** (~18.5M of 1.56B); adapter ≈ **70–75 MB** vs ~**3 GB** base; merged ≈ **3.1 GB** (adapter dissolves into weights).
2. LoRA trains a tiny zero-initialized subspace — large steps are safe and needed; at full-FT scale, 2e-4 across 1.5B pretrained weights causes instability/catastrophic drift. Rule: LoRA lr ≈ 10× full-FT lr.
3. The **RAG-vs-FT discriminator**: fine-tuning shifts *behavior/format*; knowledge changes need **RAG** (fresh, attributable) or much heavier continued pretraining.
4. None — but it violates your wallet. Cost hygiene (terminate, clean caches) is the operational habit the lab grades implicitly.

</details>

## Build block (4 h) — ship day

**Study→build echo, week-closing edition:** this morning you ran LoRA through HF PEFT in the cloud; this afternoon you prove your own implementation is *indistinguishable from it* — same config → identical trainable-param count, overlapping loss curves. Study taught the standard; build met it. That parity claim is the strongest line on the whole project's README.

[Project brief](../../../gpu-engineering-lab/02-llm-engineering/week-06-lora-from-scratch/README.md) — Day 5: parity vs PEFT + writeup (bench → README numbers → push).

**Objective:** run your exact Day-2 config through HF PEFT (`LoraConfig` + `get_peft_model`), same data/seed/schedule. Verify trainable-param count matches yours **exactly** and loss curves overlap to eyeball tolerance. Then write `RESULTS.md` — what r bought you, VRAM/time comparison tables, eval table, PEFT parity note, before/after generations — and push.

**Definition of done:**
- `make test` fully green, including PEFT parity tests
- Loss-curve overlay plot: yours vs PEFT, same axes, committed
- `RESULTS.md` complete: bf16-vs-QLoRA table, eval JSON summary, qualitative generations table, parity note
- Root README results row updated; **pushed to GitHub**
- (From this morning: lab artifacts noted, cloud instance terminated)

**One hint:** if param counts differ by exactly one layer's worth, check `modules_to_save` / whether PEFT wrapped `lm_head` or embeddings under your target list — and confirm `bias="none"` on both sides. Param-count parity is a *config* discrepancy 95% of the time, not a math bug.

## Close the day (15 min)

- **Anki:** the five weekend-survivor ideas above as five cards, plus lab numbers (1.18%, 70 MB, 3.1 GB). Run the full week-5 + week-6 deck once — week 7 (31% of the exam) needs a clear head, not review debt.
- **notes.md:** two lines — self-check score + weakest topic; the parity verdict (param count exact? curves overlap?).
- **Blockers:** anything unshipped gets a named weekend hour or an explicit drop. Confirm the week-7 GPU plan (quantize-serve + DDP labs both need cloud time) so Monday starts with study, not logistics.
