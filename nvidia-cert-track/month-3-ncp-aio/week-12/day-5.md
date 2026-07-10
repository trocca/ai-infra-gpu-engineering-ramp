# Week 12 · Day 5 — EXAM DAY: NCP-AIO

[← Master Plan](../../../MASTER-PLAN.md) · [Week 12 overview](plan.md) · [← previous day](day-4.md) · [Campaign checklist →](../../../MASTER-PLAN.md)

---

## Study block (2 h) — exam protocol

**Today (Fri Oct 2) you sit NCP-AIO: 30 MCQ + 3 hands-on labs, 120 minutes, online proctored.**
Twelve weeks of prep are in the bank; today's only job is clean execution.

### 1. Morning skim — 30–45 minutes MAX, no new material

Skim, don't study: flashcard *misses* only, then the one-page recall set —

- the allocation-strategy table (MIG / time-slicing / MPS / fractions / DRA);
- the Xid table (13/31/43 app · 48/63/64/74/79/94/95 hardware) + drain→`dcgmi diag -r 3`→RMA;
- the 4-step container-can't-see-GPU sequence; the fabric manager signature;
- syntax banks: `nvidia-smi mig -cgi/-lgi`, `sacctmgr`, `dcgmi diag -r <1-4>`, `nvidia-ctk
  runtime configure`, cmsh mode names + `commit`;
- the three NGC auth paths.

Close the notes when the timer rings. More cramming past this point trades calm for noise.

### 2. Logistics (from yesterday's [checklist](../../tools/booking-checklist.md) — already done, just verify)

ID on desk · room alone, door closed, desk cleared · one monitor, externals unplugged · apps
closed, notifications and OS updates off · wired connection · **check in ~30 min early**.
System test passed yesterday — don't re-tinker with anything today.

### 3. Exam strategy — the 120-minute budget

**The 120 minutes, drawn — labs get 75; MCQs must live inside 45 or they eat lab points.**

```
0'                 45'                                        115'    120'
|---- 30 MCQs -----|-------------- 3 hands-on labs ------------|-------|
  45 min              25 min x 3 = 75 min                        ~5 min
  90 s per question   read fully -> act -> verify, every step    flagged
  bank easy, flag rest                                           MCQs
```

- **Labs are where the points hide: ~25 min × 3 = 75 min. MCQs get ~45 min (90 s each).**
- If the interface allows, a fast MCQ pass first: bank the easy ~60%, flag the rest, move on.
  Never 5 minutes on one question — a flag costs nothing, a stall costs a lab step.
- In each lab: read the FULL task before typing; verify every step's output before the next;
  when something fails, **one diagnostic** (`kubectl describe` / `journalctl` / `nvidia-smi mig
  -lgi` / `sacct`), then adapt — never thrash fix-after-fix without looking.
- Leave ~5 minutes to revisit flagged MCQs. An educated guess beats a blank.
- Something feels broken in the lab environment itself? Tell the proctor immediately — that's
  what they're for.

### 4. After the exam — close the campaign

- [ ] Log result + date in [PROGRESS.md](../../PROGRESS.md) — the final row of the 12-week table.
- [ ] Claim the digital badge (Credly email) and add it to LinkedIn next to NCA-AIIO and
      NCP-GENL; note the 2-year expiry in your calendar with a 3-month reminder.
- [ ] Same day, while fresh: write which domains felt under/over-weighted in your prep into
      [notes.md](notes.md) — future-you's syllabus review.
- [ ] Publish the LinkedIn/blog post drafted on Day 4.
- [ ] Walk the **[Definition of undeniable checklist](../../../MASTER-PLAN.md)** in the master
      plan: 3 badges live · 12 lab READMEs with measured numbers · demo repo re-verified with
      the ferrum-serve scene · the recorded end-to-end run · pitch updated · interview drill
      done twice without notes. Tick what's done; whatever's open becomes next week's short list.

### Quick check (post-exam debrief — answer tonight)

**1. Which lab exercise cost the most time, and would your 25-min drill protocol have covered it?**
<details><summary>Answer</summary>Your answer, in notes.md. If yes: the drill system works, keep it for the next cert. If no: name the gap (topic vs time-management) — that distinction decides what changes next time.</details>

**2. Did the 45/75 split survive contact with the real exam?**
<details><summary>Answer</summary>Compare planned vs actual. The usual failure is MCQ overrun stealing lab time; if that happened, the fix stays mechanical: harder flagging, earlier first pass.</details>

**3. Pass or not — what is true today that wasn't true 12 weeks ago?**
<details><summary>Answer</summary>Write it as capabilities, not feelings: the stacks you can rebuild from git, the failure signatures you recognize on sight, the numbers you measured yourself. That list is the campaign's real artifact — the badge just certifies it.</details>

---

## Build block (light — exam week) — capstone Day 5: ship

Objective (Day 5 of the [capstone brief](../../../gpu-engineering-lab/03-scale-and-serve/week-12-capstone/README.md)),
morning-light or post-exam, energy permitting:

- **DoD:** CI green end to end; tag `v1.0`; demo GIF in the root README renders.
- **DoD:** the 5-minute repo tour rehearsed out loud once more, timed under 6 minutes.
- **DoD:** phase-3 cost tallied in the root cost log (target ≤ $50).
- Hint: tag before you tinker — `v1.0` marks what the exam-week version actually was.
- Then close the laptop and go pass NCP-AIO. Everything else keeps.

---

## Close the day (15 min)

- [ ] PROGRESS.md final row written; badge claim email checked.
- [ ] One line in [notes.md](notes.md): the debrief trio above.
- [ ] No cloud instances anywhere: week-11 node deleted, capstone session terminated, cost log final.
- [ ] Campaign closed. Whatever the result email says — you built the thing. Go celebrate, then
      pick up the [master plan checklist](../../../MASTER-PLAN.md) on Monday.
