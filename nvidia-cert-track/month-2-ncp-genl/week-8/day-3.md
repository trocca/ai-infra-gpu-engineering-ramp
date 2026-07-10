# Week 8 · Day 3 — Mock exam day: sit it, score it, map it

[← Master Plan](../../../MASTER-PLAN.md) · [Week 8 overview](plan.md) · [← previous day](day-2.md) · [next day →](day-4.md)

## Study block (2 h + marking)

No new material today. Today produces the single most valuable artifact of exam week: a **per-domain miss map** that tells tomorrow's re-drill exactly where to aim.

### Sitting protocol — [mock-exam.md](../mock-exam.md)

- **Timed: 120 minutes, hard stop** — the real exam's ratio (60–70 Q / 120 min). The mock's own header offers a punishing 75-minute variant; if you finish with time to spare, note your actual elapsed time — that's your pacing calibration. Do NOT pause the clock.
- **Exam conditions**: no notes, no flashcards, phone out of the room, answers circled on paper. Simulate Friday: one screen, water bottle, door closed.
- **Pacing plan (rehearse it now, use it Friday)**: 60 questions / 120 min = 2 min/question, but budget ~1.5 to bank a review buffer. Any question eating **> 3 minutes: flag, pick your best elimination guess, move on**. Never leave blanks — there's no penalty for wrong answers. Last 15 minutes: revisit flags only.
- **First pass discipline**: read the *scenario constraint* before the options (memory ceiling? latency? no training budget? compliance?). Most NCP-GENL questions are constraint-matching; the constraint eliminates two options instantly.

### Marking protocol (~40 min, immediately after — accuracy fades fast)

1. Score against the key. The sections mirror the official weights, so compute a **per-domain score**:

| # | Domain | Weight | Qs | Score | Miss × weight |
|---|--------|-------:|---:|-------|---------------|
| 1 | Model Optimization | 17% | 10 | /10 | |
| 2 | GPU Acceleration & Optimization | 14% | 8 | /8 | |
| 3 | Prompt Engineering | 13% | 8 | /8 | |
| 4 | Fine-Tuning | 13% | 8 | /8 | |
| 5 | Data Preparation | 9% | 5 | /5 | |
| 6 | Model Deployment | 9% | 5 | /5 | |
| 7 | Production Monitoring & Reliability | 7% | 4 | /4 | |
| 8 | Evaluation | 7% | 4 | /4 | |
| 9 | LLM Architecture | 6% | 4 | /4 | |
| 10 | Safety, Ethics & Compliance | 5% | 4 | /4 | |

Copy this table into [notes.md](notes.md) and fill it in. **Priority = miss rate × exam weight**: two misses in Model Optimization (17%) outrank two misses in Safety (5%).

2. For **every miss AND every lucky guess**, write one line in notes.md: *the concept + the correct discriminator* (e.g. "W4A16 ≠ prefill speedup — weight-only doesn't cut FLOPs"). Lucky guesses count as misses for re-drill purposes; they will not be lucky twice.
3. Rank your **three weakest domains** by the priority column — that ranking IS tomorrow's agenda.

### Decision thresholds

- **≥ 75%** overall → comfortable pass margin; tomorrow is polish + logistics.
- **65–75%** → pass is plausible but not safe; tomorrow's re-drill is load-bearing — give it the full block and steal from the build if needed.
- **< 65%** → tonight, seriously consider moving the booking (usually free ≥ 24–72 h out — check your confirmation email; policy notes in the [booking checklist](../../tools/booking-checklist.md)). A postponed exam costs a week; a failed NCP-GENL costs $200 and morale.

Whatever the score: no re-drilling tonight. Log it, close the notebook.

## Build block (4 h)

**ferrum-serve day 3: engine + server — streaming, cancellation, graceful shutdown.** The system comes alive today. [Project brief](../../../gpu-engineering-lab/02-llm-engineering/week-08-mini-inference-server/README.md)

- `engine.rs`: `PagedKvCache` (write/gather), the engine-thread loop (`spawn_engine`), prefill + batched decode via Tuesday's scheduler.
- `server.rs`: `/v1/completions` with SSE (OpenAI chunk shape), `/stats`, `shutdown_signal` + graceful drain.
- Smoke ritual: `make smoke`; 10 concurrent `curl -N` streams while `/stats` shows blocks recycling; **Ctrl-C one stream and watch its blocks return**; `kill -TERM` mid-stream → clean drain, exit 0.
- **Correctness gate (non-negotiable):** temperature-0 server output == Monday's `generate_single` output, token for token, at any concurrency.
- Hint: cancellation is ownership, not polling — client disconnects → axum drops the SSE stream → your `mpsc` receiver drops → the engine's `send()` errs → free the sequence's blocks in that error arm. If you find yourself writing an `is_disconnected()` check, you've re-imported the Python design.

## Close the day (15 min)

- Anki: add a card for each miss-discriminator line you wrote today (these are the highest-value cards you'll ever make).
- One line in [notes.md](notes.md): mock score, three weakest domains, and the pacing stat (how many questions were still unanswered at the 60-min mark).
- Blockers — and book nothing for tomorrow evening: Thursday ends with the proctoring system check and an early night.
