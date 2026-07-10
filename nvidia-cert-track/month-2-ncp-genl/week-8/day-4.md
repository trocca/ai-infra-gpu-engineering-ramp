# Week 8 · Day 4 — Weakest-domain re-drill + proctoring dress rehearsal

[← Master Plan](../../../MASTER-PLAN.md) · [Week 8 overview](plan.md) · [← previous day](day-3.md) · [next day →](day-5.md)

## Study block (2 h)

Two jobs today: convert yesterday's miss map into repaired knowledge, and remove every possible logistics failure mode for tomorrow. No new topics — the day before an exam is for consolidation, not acquisition.

### Part 1 — Re-drill the top-3 weak domains (~75 min, 25 min each)

Work the three domains you ranked yesterday by **miss rate × weight**. For each: reread the mapped material below, then **re-derive the core artifact from memory** on blank paper — writing it out is the test; reading is not.

| Domain | Where the material lives | The from-memory artifact |
|---|---|---|
| Model Optimization (17%) | [week-7 day 1](../week-7/day-1.md), [day 2](../week-7/day-2.md), notes | quantize/dequant equations; format→hardware table; the four-axes decision matrix |
| GPU Acceleration (14%) | [week-7 day 3](../week-7/day-3.md), [day 4](../week-7/day-4.md), [day 5](../week-7/day-5.md) | prefill-vs-decode table; FlashAttention 3-liner; parallelism→collectives table |
| Prompt Engineering (13%) | week-5 days 3–4, notes | zero/few-shot vs CoT vs ReAct decision list; sampling-params table |
| Fine-Tuning (13%) | week-6 days 1–3 + LoRA lab | the LoRA equation + rank/alpha effects; SFT→RLHF/DPO pipeline sketch |
| Data Preparation (9%) | week-5 day 5 | curation pipeline order (dedupe/filter/PII/blend) — NeMo Curator stages |
| Model Deployment (9%) | [day 1](day-1.md) | the five-layer stack drawing + five topologies with match-ups |
| Monitoring (7%) | [day 2](day-2.md) | metric definitions + canary rollout design |
| Evaluation (7%) | week-6 day 4 | benchmark names→what they measure; LLM-as-judge caveats |
| LLM Architecture (6%) | week-5 days 1–2 | attention/KV-cache memory math; decoder-only data flow |
| Safety/Compliance (5%) | [day 2](day-2.md) | threat list + rails mapping; licensing one-liners |

Then a **speed pass (~20 min)**: skim all four weekly plan exit-criteria lists (weeks [5](../week-5/plan.md)·[6](../week-6/plan.md)·[7](../week-7/plan.md)·[8](plan.md)). Anything you can't do from memory gets 10 focused minutes, max — timebox ruthlessly.

Finish with flashcards: full deck, flag every hesitation; flagged cards get one more pass tonight or tomorrow morning — nothing else does.

### Part 2 — Proctoring dress rehearsal (~25 min) — [booking checklist](../../tools/booking-checklist.md)

Run the full checklist **on the exact machine, network, and room you'll use tomorrow**:

- [ ] **Proctor system test** passed: webcam, mic, screen share, bandwidth — today, not tomorrow morning.
- [ ] Government photo ID at the desk, name matching the candidate account **exactly**.
- [ ] One monitor only — external monitors physically unplugged. All apps closed, notifications/OS updates disabled, no VMs running.
- [ ] Room: alone, door closed, desk cleared of papers/devices/headphones — you'll pan the webcam at check-in.
- [ ] Wired Ethernet if at all possible; phone hotspot identified as backup.
- [ ] Check-in time computed: exam slot minus 30 minutes, alarm set.
- [ ] Reschedule cutoff confirmed (from your booking email) — after tonight you're committed either way.

### Tonight

One page only: your miss-discriminator lines from yesterday. Then stop. **Sleep beats cramming** — a rested pass through 60 questions is worth more than two exhausted extra hours of review.

## Build block (4 h)

**ferrum-serve day 4: load test vs vLLM — the honest number.** [Project brief](../../../gpu-engineering-lab/02-llm-engineering/week-08-mini-inference-server/README.md)

- `bench/loadtest.py` is COMPLETE (deliberately Python — the referee's language is irrelevant; both servers get the identical client speaking HTTP/SSE).
- Run the **closed-loop matrix** (concurrency 1→16) and **open-loop** (Poisson arrivals) against ferrum-serve, then against `vllm serve` per `bench/compare_vllm.md` — same model, same GPU, same client, same sampling. Fix the power profile; record clocks; ≥ 50 requests per point after warmup; medians AND p95.
- Collect the **Rust-only metrics table** (compare_vllm.md §7): stripped binary size, docker image size, cold-start-to-first-token — this feeds week 11's K8s story.
- **Definition of done:** TTFT/ITL/throughput-vs-concurrency curves for both servers, JSON committed; ≥ 8 concurrent streams sustained without error; free blocks back to 100% after every workload (`/stats`).
- Hint: expect to lose to vLLM substantially — a gather-based engine losing to years-tuned paged-attention CUDA kernels is the *expected result*; capture the open-loop queueing collapse point too, it's the most instructive plot you'll make.
- Keep it strictly to 4 h today. If the matrix isn't done, cut concurrency points, not sleep — plots can be regenerated Saturday; tomorrow morning cannot.

## Close the day (15 min)

- Anki: flagged cards only — one final pass.
- One line in [notes.md](notes.md): re-drill verdict per weak domain (repaired / still shaky).
- Early night. Tomorrow: [exam-day protocol](day-5.md).
