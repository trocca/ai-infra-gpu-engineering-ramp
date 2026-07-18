# Week 8 Plan (Aug 31–Sep 4, 2026) — Deployment, Monitoring, Safety, Review, Mock — EXAM WEEK

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

Exam coverage this week: **Model Deployment 9% + Production Monitoring & Reliability 7% + Safety/Ethics/Compliance 5% = 21%**, plus consolidation of everything. Exam sits Fri Sep 4 / Sat Sep 5 — confirm your booking today (Monday) if you haven't.

## Prerequisites before Monday

- Companion lesson: [Week 08 companion — serving math, queueing, Rust async, and safety support](../../../companion-lessons/week-08.md).
- Source reading: [HF Ultra-Scale Playbook — serving memory and precision carryover](../../../references/hf-ultrascale-playbook.md#week-8---serving-memory-and-precision-carryover).
- Math support: TTFT, ITL, throughput, p95 latency, Little's Law, and KV-cache memory budgeting.
- Programming/systems support: `async`/`await`, `tokio`, `axum`, continuous batching, paged KV, NIM/vLLM/Triton role boundaries, and safety controls.
- Gate: define TTFT/ITL/throughput without notes and identify what `ferrum-serve` will compare against vLLM.

---

## Day 1 (Mon) — Deployment + quantize/serve lab
**Domain: Model Deployment (9%)**

- (30 min) Serving stack map — be able to draw this: model checkpoint → inference engine (**vLLM** / **TensorRT-LLM** / SGLang) → server layer (**Triton Inference Server**: multi-framework, ensembles, model repository, dynamic + in-flight batching via the TRT-LLM backend) → packaged product (**NIM**: prebuilt containers, one OpenAI-compatible API, auto-selected optimized profiles per GPU, enterprise support) → orchestration (K8s: NIM Operator, autoscaling, and — from your demo repo — MIG/time-slicing/MPS for GPU sharing, DRA ResourceClaims for allocation, KAI for scheduling).
- (25 min) Serving topologies: single-GPU; TP across GPUs in one node (when model > 1 GPU); replicas behind a load balancer (throughput scale-out); **disaggregated prefill/decode** (separate GPU pools, KV transferred between them — **NVIDIA Dynamo**'s core idea, plus KV-aware routing); multi-model (adapters: multi-LoRA serving on one base model). Match topology → workload: latency-sensitive chat vs offline batch vs many fine-tuned tenant variants.
- (~55 min) Run `labs/lab-quantize-serve.md`: vLLM FP16 baseline vs AWQ, benchmark TTFT/throughput. Paste numbers into notes.
- (10 min) Flashcards.
- Cross-ref demo repo: this whole day is your demo narrative — vLLM/Dynamo/NIM, MIG vs MPS, DRA. Rehearse answers as booth pitches; they double as exam answers.

## Day 2 (Tue) — Monitoring & reliability
**Domain: Production Monitoring & Reliability (7%)**

- (35 min) LLM-specific metrics — definitions cold: **TTFT** (time to first token — prefill + queueing), **ITL/TPOT** (inter-token latency — decode speed), **e2e latency**, **throughput** (tokens/s, requests/s), **goodput** (throughput meeting SLOs). Percentiles: report P50/P95/P99, never means — tail latency is what users feel and SLOs bind. Little's law intuition: concurrency = arrival rate × latency.
- (30 min) GPU/system metrics via **DCGM** (+ dcgm-exporter → Prometheus/Grafana on K8s): GPU utilization (and why it lies — SM occupancy vs "busy"), memory used, SM clock/throttling, temperature, power, NVLink errors, **XID errors** (Xid = GPU fault codes → node health, remediation/draining). Serving-engine metrics: queue depth, KV-cache utilization, preemption/eviction counts, request timeouts.
- (30 min) Reliability patterns: health/readiness probes for model servers (model loaded ≠ container up), **canary** deployments (small traffic slice to new model/config, compare metrics, auto-rollback), shadow traffic, A/B testing; **drift**: input/prompt distribution drift, output-quality drift — detect with embedding-distribution monitoring and periodic golden-set evals in CI/CD; token-cost monitoring.
- (15 min) Flashcards.

## Day 3 (Wed) — Safety, ethics, compliance
**Domain: Safety/Ethics/Compliance (5%)**

- (30 min) Threats: **prompt injection** (direct jailbreaks; indirect via retrieved/pasted content — the RAG attack), data exfiltration via outputs, harmful-content generation, hallucination presented as fact. Defenses in layers: system-prompt hardening < input/output filtering < **guardrails**.
- (30 min) **NeMo Guardrails**: programmable rails (Colang) around any LLM — input rails (jailbreak/topic checks), dialog rails (stay on allowed flows), retrieval rails, output rails (moderation, fact-check hooks); works with content-safety models (e.g. Llama Guard–class classifiers). Know it's a *framework around* the model, complementary to alignment (RLHF/DPO) which is *in* the model.
- (30 min) Compliance topics: **PII** handling across the lifecycle (curation-time redaction — NeMo Curator; inference-time detection/masking; logging policies), model/data **licensing** (open-weights ≠ open-source; Llama community license restrictions vs Apache-2.0 models like many Qwen/Mistral variants; dataset provenance and copyright), bias/fairness evaluation, transparency artifacts (model cards), regulatory awareness (EU AI Act risk tiers, GDPR data rights vs trained weights).
- (30 min) Start review: re-run all three earlier self-checks quickly (aim 90%+ now); list weak domains for tomorrow.

## Day 4 (Thu) — Full review day
**All domains, weighted by weakness × exam weight**

- (30 min) Flashcards: full deck (`flashcards.csv` in Anki), flag every hesitation.
- (45 min) Attack the top-3 weak spots found yesterday — reread the relevant notes.md sections and re-derive from memory (LoRA equation, KV-cache math, parallelism table, quantization decision matrix, RAG pipeline, RLHF pipeline, metric definitions).
- (30 min) Speed drill: skim all four plan.md exit-criteria lists; anything you can't do from memory gets 10 focused minutes.
- (15 min) Logistics: proctor system test, ID ready, room plan, exam-timing plan (60–70 Q / 120 min → flag-and-skip anything > 3 min).

## Day 5 (Fri) — Mock exam + final gaps (or exam day)

- (75 min) `mock-exam.md`: 60 questions, strictly timed at 75 minutes (harder than the real ratio on purpose), no notes.
- (35 min) Score against the key. For every miss AND every lucky guess, write one line in notes.md: the concept + the correct discriminator.
- (10 min) Stop. Sleep beats cramming. Target ≥ 75% mock → comfortable pass margin; 65–75% → re-drill the missed domains tomorrow morning before the exam; < 65% → consider rescheduling (you can usually move a booking ≥ 24 h out).

## Exit criteria (before walking into the exam)

- [ ] I can draw the NVIDIA serving stack (engine → Triton → NIM → K8s) and place vLLM/TRT-LLM/Dynamo on it
- [ ] I can pick a serving topology for chat / batch / multi-tenant-LoRA scenarios
- [ ] I can define TTFT, ITL, throughput, goodput and tie each to prefill/decode and to an SLO
- [ ] I can explain why P99 matters more than mean latency, and what DCGM/XID monitoring is for
- [ ] I can design a canary rollout for a new model version and say what metrics gate promotion
- [ ] I can explain direct vs indirect prompt injection and layer the defenses, ending at NeMo Guardrails
- [ ] I can distinguish alignment (in-model) from guardrails (around-model)
- [ ] I can speak to PII handling at curation vs inference time, and open-weights licensing gotchas
- [ ] Mock exam ≥ 75% under time pressure
- [ ] All four weekly self-checks ≥ 90% on re-run
