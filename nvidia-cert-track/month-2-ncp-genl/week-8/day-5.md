# Week 8 · Day 5 — EXAM DAY: NCP-GENL (Friday, Sep 4)

[← Master Plan](../../../MASTER-PLAN.md) · [Week 8 overview](plan.md) · [← previous day](day-4.md) · [next day →](../../month-3-ncp-aio/week-9/day-1.md)

## Study block (exam-day protocol — not a normal study day)

Today you sit **NCP-GENL**. Eight weeks of work land here; the protocol below exists so zero decisions are made under stress.

### Morning — light review only (60–90 min, then STOP)

- Flagged Anki cards from yesterday — one pass, no new cards.
- Your one-page discriminator sheet (the mock-miss lines). Read it twice.
- Optionally, from memory on blank paper, the three highest-weight artifacts: the quantization decision matrix, the prefill/decode table, the serving-stack drawing.
- **Do not** open papers, docs, or new practice questions. Anything you'd learn now costs more in confidence than it earns in points.

### Pre-exam (T−45 min)

- [ ] Bathroom, water, snack. Phone silenced and out of the room.
- [ ] Desk cleared; second monitor unplugged; notes/papers out of the room (webcam room-pan is coming).
- [ ] Every app closed; notifications and OS updates disabled; wired network if available.
- [ ] ID on the desk. **Check in at T−30** per the [booking checklist](../../tools/booking-checklist.md).

### During — the pacing contract you rehearsed Wednesday

- 60–70 questions / 120 minutes → cruise at ~1.5 min/question to bank a review buffer.
- Read the **constraint** first (memory ceiling / latency / no-training-budget / compliance) — it eliminates options before you've finished the stem.
- **> 3 minutes on a question: flag, best elimination guess, move.** No blanks, ever — no wrong-answer penalty.
- Final 15 minutes: flagged questions only. Don't re-litigate confident answers; first instincts on known material are usually right.
- One bad question means nothing. The pass margin was built over eight weeks; let a hard item go.

### After — same day, while fresh

1. **Log the result in [PROGRESS.md](../../PROGRESS.md)**: the month-2 exam-result line (score, date) + the week-8 row (mock %, confidence /5).
2. Claim the **Credly badge** when the email lands; add it to LinkedIn — for an evangelist/pre-sales trajectory, badge visibility is half the value. Calendar the 2-year expiry with a 3-month-early reminder.
3. Write five lines in [notes.md](notes.md): which domains felt over/under-weighted vs your prep, and what you'd tell week-5-you. This feeds the month-3 (NCP-AIO) plan directly — that exam is $500 and hands-on; intelligence about how NVIDIA writes questions is valuable.
4. If it didn't go your way: log it just the same, take the weekend completely off, then re-book with the miss map as your syllabus. The infrastructure month starts Monday regardless — [week 9, day 1](../../month-3-ncp-aio/week-9/day-1.md).

Also today: quick pass over the [week-8 self-check](self-check.md) is *not* required — the real exam superseded it; mark the exit criteria in [plan.md](plan.md) from memory of the mock + exam instead.

## Build block (LIGHT — 30–60 min, deliberately)

**ferrum-serve day 5: publish.** The lab brief marks Friday light on purpose — the exam owns today. [Project brief](../../../gpu-engineering-lab/02-llm-engineering/week-08-mini-inference-server/README.md)

- Generate final plots from yesterday's JSON; write `RESULTS.md`: the **honest %-of-vLLM number**, the **gather-vs-true-paged-attention caveat** stated explicitly, the "what vLLM does that I don't" list (paged-attention kernels, CUDA graphs, chunked prefill, prefix caching, spec decode, TP), and the Rust metrics table (binary size, image size, cold-start).
- Add the month-2 results row to the root README. Push.
- **Definition of done:** `cargo test` green, clippy clean, `Cargo.lock` committed, RESULTS.md + plots pushed — and the exam got its hours.
- Hint: write the %-of-vLLM sentence first and the caveat sentence immediately after it — those two lines are the interview artifact; everything else is supporting material.
- Timebox hard. Anything unfinished moves to the weekend, guilt-free. This is a *capstone shipped during exam week* — say that sentence in interviews.

## Close the day (15 min)

- No Anki tonight. The deck earned a rest day too.
- One line in [notes.md](notes.md): exam outcome + the single strongest and weakest moment in there.
- Close out month 2. Two certs' worth of momentum, a Triton kernel suite, and a Rust inference engine in four weeks — Monday begins NCP-AIO and the infrastructure month: [week 9, day 1](../../month-3-ncp-aio/week-9/day-1.md).
