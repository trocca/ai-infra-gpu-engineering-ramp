# Week 12 · Day 3 — Mock day: the full exam simulation

[← Master Plan](../../../MASTER-PLAN.md) · [Week 12 overview](plan.md) · [← previous day](day-2.md) · [next day →](day-4.md)

---

## Study block (2 h)

**All four domains — today measures, it doesn't teach.** The mock decides (per the master
plan's exam gate) whether Friday happens: **sit the real exam only at ≥ 80% here** — that's
≥ 24/30 MCQ *and* credible passes on the lab scenarios. Decide today, not Friday morning.

### 1. Set the conditions (0:00–0:05)

Exam-realistic or the score is fiction: closed book, no second monitor, phone out of the room,
plain terminal available for scratch work, one 120-minute timer. The real thing is 30 MCQ +
3 hands-on labs in 120 minutes, online proctored — simulate the pressure, not just the questions.

### 2. Sit [mock-exam.md](../mock-exam.md) — 120 minutes (0:05–2:05)

Budget it exactly like Friday:

- **MCQ section: 45 minutes for 30 questions** (90 s each). Fast first pass — bank the easy
  ~60%, flag and skip anything that costs more than 90 seconds. Second pass for the flags.
  Never sink 5 minutes into one question; a flagged MCQ is worth the same as an answered one
  until time runs out.
- **3 written lab scenarios: 25 minutes each** (75 min). On paper, write what you would *type*,
  in order, including the verification command after each step — the scenarios are graded on
  the diagnostic sequence, not prose. Read each scenario fully before writing line one.

Domain map for scoring: Q1–9 Installation & Deployment, Q10–16 Administration, Q17–23 Workload
Management, Q24–30 Troubleshooting & Optimization.

### 3. Score and dissect (immediately after — eats into the build block, allowed)

- Mark with the answer key + lab scoring guide at the bottom of [mock-exam.md](../mock-exam.md).
  No partial credit generosity on MCQs; labs per the guide's step rubric.
- Compute per-domain scores and write all four into [notes.md](notes.md) — **the weakest domain
  becomes tomorrow's re-drill target.**
- For every miss, write one line: *what I answered, why the right answer is right, which
  symptom/command I confused.* Each becomes a flashcard tonight — the miss-card loop is where
  the mock pays rent.
- Gate check against the [week plan](plan.md) exit criteria: ≥ 24/30 in ≤ 45 min on MCQs, all
  three labs solvable on paper. Below the bar → tonight you decide what Thursday re-drills, and
  you consider the reschedule policy honestly (rescheduling beats burning a $500 attempt —
  [booking checklist](../../tools/booking-checklist.md)).

### Quick check (meta — answer after scoring)

**1. Which domain scored lowest, and which *specific* labs/flashcards cover it?**
<details><summary>Answer</summary>Your answer, from notes.md. Map: Install/Deploy → lab-gpu-operator + driver/toolkit cards; Administration → lab-slurm-basics + BCM/cmsh cards; Workload Mgmt → lab-runai-kai + allocation table + NGC cards; Troubleshooting → lab-troubleshoot + Xid/dcgmi/NCCL cards. Tomorrow re-drills exactly that row.</details>

**2. Did the MCQ pass fit in 45 minutes — and if not, where did the time leak?**
<details><summary>Answer</summary>Count questions that took >90 s. The fix is mechanical, not knowledge: flag-and-move on the first read, trust the second pass. Rehearse that motion tomorrow, not more content.</details>

**3. On the lab scenarios: did every step you wrote include its verification command?**
<details><summary>Answer</summary>The rubric rewards diagnose→fix→<em>verify</em> chains. If your answers jump fix-to-fix without checks, that's the single highest-value habit to correct before Friday — one diagnostic between any two actions.</details>

---

## Build block (4 h) — capstone Day 3: repo polish

Objective (Day 3 of the [capstone brief](../../../gpu-engineering-lab/03-scale-and-serve/week-12-capstone/README.md)),
local — a 5-minute reviewer must see what you built, measured, and reported honestly:

- Every week's README (1–12): results table filled with real numbers, zero "TODO" in published text.
- Root README: the 5-minute portfolio tour — what/phase map/three headline numbers/how to reproduce/honest limitations.
- **DoD:** mermaid architecture diagrams wherever missing; demo GIF or asciinema of `make pipeline` + a curl against the served model.
- **DoD:** yesterday's end-to-end numbers table rendered in the capstone README.
- Hint: read the root README pretending you're the hiring manager with 5 minutes — anything you skip past, cut or move down.

---

## Close the day (15 min)

- [ ] Anki: every mock miss is now a card (symptom → cause phrasing). Clear all due reviews — tomorrow's deck must be light.
- [ ] One line in [notes.md](notes.md): mock score by domain + go/no-go decision for Friday.
- [ ] Blockers: name tomorrow's weakest-domain re-drill target explicitly.
- [ ] Local day — no instance check. Sleep matters more than one more review pass from tonight on.
