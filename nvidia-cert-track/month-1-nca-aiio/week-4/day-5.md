# Week 4 · Day 5 — EXAM DAY: NCA-AIIO

[← Master Plan](../../../MASTER-PLAN.md) · [Week 4 overview](plan.md) · [← previous day](day-4.md) · [next day →](../../month-2-ncp-genl/week-5/day-1.md)

## Study block (2 h)

Today the study block is a protocol, not a lesson. **No new material this morning.** Anything you'd learn today would cost more in confidence than it gains in points.

### Morning warm-up (40 min, calm)

- (20 min) **Flashcards only** — normal review pass, all domains. Familiar cards, familiar rhythm. If a card feels shaky, note it and move on; do not open source docs.
- (20 min) Read your own [notes.md](notes.md) summaries — one calm pass across the three domains. You wrote these; they're the highest-density review that exists for your brain. Then close everything.

### Pre-exam logistics checklist (run it, don't trust memory)

- [ ] Log in **15 minutes early** to the proctoring portal.
- [ ] **ID** on the desk, matching your portal name (verified Wednesday/Thursday).
- [ ] Exam machine: yesterday's system check passed; charger in; notifications/updates off; VPN off if it interferes.
- [ ] Room: clean desk, second monitor unplugged, **phone out of the room**, door closed, household warned.
- [ ] Water within reach; bathroom before check-in (proctored = no breaks in a 60-minute exam).

### In the exam: pacing and question strategy

- **Pacing math**: 50 questions / 60 minutes ≈ **70 seconds each**. First pass: answer *everything*, flag doubts. Second pass: flagged questions only. **Never leave blanks** — there's no penalty for guessing.
- **Question strategy**: this exam rewards judgment. Eliminate the two obviously-wrong options first; choose between the final two using the **scenario constraint** — the "must," "lowest latency," "multi-tenant," "no code changes," "without rewriting the application" phrase is the key that selects the answer.
- Trust first instincts on flagged questions unless you can articulate *why* the other option is right — changing answers on vibes loses points.
- One hard question is worth exactly as much as one easy question. If 70 seconds pass without traction, best guess, flag, move.

### After the exam

1. **Breathe.** Result appears via the certification portal.
2. **Log the result in [PROGRESS.md](../../PROGRESS.md)** — date, score/outcome, and two lines: which domain felt hardest, and what you'd tell week-1-you.
3. **Claim the badge** when the credential email arrives (Credly-style digital badge) — add it to LinkedIn while the momentum is real.
4. If it didn't go your way: the retake policy means this is a delay, not a verdict. Write down every topic you remember struggling with *today* (it evaporates by tomorrow), map them to domains, book the retake, and fold the gap list into next week's plan.
5. Either way — **Month 1 is done.** Monday starts NCP-GenL (Month 2): [week 5, day 1](../../month-2-ncp-genl/week-5/day-1.md).

### Quick check (recite before check-in — closed book, out loud)

1. What's the pacing plan?
<details><summary>Answer</summary>~70 seconds per question. First pass answers all 50 and flags doubts; second pass revisits flags only; no blanks ever — guessing is free.</details>

2. What breaks a tie between the last two plausible options?
<details><summary>Answer</summary>The scenario constraint — the "must / lowest latency / multi-tenant / no code changes" phrase. The distractor satisfies the topic; the answer satisfies the constraint.</details>

3. What are the two post-exam actions in this repo/portal?
<details><summary>Answer</summary>Log the result (date, outcome, reflections) in PROGRESS.md, and claim the digital badge from the certification portal when the email arrives.</details>

## Build block (4 h — deliberately light today)

**rusty-kernels Day 5 — package polish + publish.** Low-intensity, high-satisfaction work that fits exam day. [Project brief](../../../gpu-engineering-lab/01-foundations/week-04-pytorch-custom-ops/README.md)

- `maturin build --release` → wheel; `pip install` into a **fresh venv** and run the tests (the "works from wheel" check).
- Copy `.github-workflow-snippet.yml` to `.github/workflows/ci.yml` (lint + CPU import smoke + Rust fmt/clippy/test; GPU tests skip in CI).
- RESULTS.md: speedup tables and charts, the Welford note, the binding-decision paragraph, limitations, "What didn't work". Push.
- Write the **Phase 1 retrospective paragraph** in the repo root README — four weeks, from first kernel to a pip-installable Rust extension.
- Definition of done: wheel installs clean, CI green on GitHub, RESULTS.md complete, retrospective pushed.
- Hint: schedule this block *after* the exam, not before — the morning belongs to the warm-up protocol and a calm head.

## Close the day (15 min)

- Anki: skip tonight. You've earned it.
- Final line in [notes.md](notes.md): exam outcome, hardest domain, and one sentence to week-1-you.
- Blockers: none. Log the result in [PROGRESS.md](../../PROGRESS.md), claim the badge, close the laptop. Month 2 (NCP-GenL) starts Monday: [week 5, day 1](../../month-2-ncp-genl/week-5/day-1.md).
