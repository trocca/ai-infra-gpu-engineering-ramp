# Week 12 · Day 4 — Weakest-domain re-drill + proctoring system check

[← Master Plan](../../../MASTER-PLAN.md) · [Week 12 overview](plan.md) · [← previous day](day-3.md) · [next day →](day-5.md)

---

## Study block (2 h)

**Target: yesterday's weakest domain.** No new material — the day before an exam is for
converting known-shaky into known-solid, and for making sure the *machine* passes its exam too.

### 1. Weakest-domain re-drill (0:00–1:10)

Take the lowest-scoring domain from yesterday's [notes.md](notes.md) entry and hit its exact
artifacts:

- **Installation & Deployment weak** → re-run [lab-gpu-operator](../labs/lab-gpu-operator.md)
  once more under the 25-min clock (or whiteboard the full component chain + failure modes if
  no node today), and redo [lab-troubleshoot](../labs/lab-troubleshoot.md) (a)+(b) from the
  symptom strings alone.
- **Administration weak** → [lab-slurm-basics](../labs/lab-slurm-basics.md) accounting section
  cold, plus the cmsh bank: mode names, `commit` discipline, `imageupdate --dry-run`,
  healthcheck triage.
- **Workload Management weak** → [lab-runai-kai](../labs/lab-runai-kai.md) queue/preemption flow
  on paper, the allocation-strategy table blind, sbatch + K8s Job manifests against the
  5-minute timer, three NGC auth paths from memory.
- **Troubleshooting weak** → [lab-mig-config](../labs/lab-mig-config.md) command sequence
  hand-written again, then recite: 4-step container sequence, Xid table, dcgmi ladder, FM
  signature, NCCL `via` lines.

Then the recite list from the [week plan](plan.md) exit criteria, out loud, no notes: GPU
Operator component order · Slurm upgrade order · A100 MIG profiles · Xid 48/63/74/79 · dcgmi
diag levels · fabric manager signature · cmsh mode names. Anything that stumbles → flashcard →
re-recite in 30 minutes.

### 2. Re-walk the three mock lab scenarios (1:10–1:30)

On paper, from the scenario text only (no peeking at your marked answers): the ordered command
list with a verification step after every action. Yesterday you graded them; today they should
flow without pauses. This is the closest rehearsal of Friday's highest-value 75 minutes.

### 3. Proctoring system check — do it NOW, not tomorrow (1:30–2:00)

Work through the [booking checklist](../../tools/booking-checklist.md) proctoring section on
**the exact machine + network you'll use Friday**:

- [ ] Run the proctor's official system test: webcam, mic, screen share, bandwidth.
- [ ] One monitor only — externals *unplugged*, not just off. No VMs. Close/disable everything
      that can pop a notification (OS updates too).
- [ ] Government photo ID at hand; name matches the candidate account exactly.
- [ ] Room plan: alone, door closed, desk cleared — you'll pan the webcam at check-in.
- [ ] Wired Ethernet if at all possible.

**Why this matters double for NCP-AIO:** the 3 hands-on labs run in a browser-based lab
environment *during the proctored session* — a flaky connection doesn't just risk a dropped
webcam feed, it drops your terminal mid-lab while the clock runs. Test the system today so
tomorrow's only variable is you.

Confirm the appointment time, check-in window (~30 min early), and the reschedule cutoff.

### Quick check

**1. Recite the alert-to-RMA playbook and the 4-step container-can't-see-GPU sequence, back to back.**
<details><summary>Answer</summary>Playbook: alert → dmesg grep xid → app vs HW → drain → dcgmi diag -r 3 → resume or RMA. Container: runtime wiring (runtimeClassName/default) → toolkit configured + daemon restarted → device plugin/node capacity → operator validator pods. If either needed a pause, once more in 30 minutes.</details>

**2. Why is the proctoring system test more critical for NCP-AIO than for a pure-MCQ exam?**
<details><summary>Answer</summary>The three hands-on labs run in a browser-based lab environment inside the proctored session — connection or screen-share failures interrupt live terminal work against the clock, not just a question form. Hence: wired connection, tested machine, single monitor, today.</details>

**3. What's tonight's cutoff rule?**
<details><summary>Answer</summary>Flashcards due today cleared, then stop — no new material after dinner. Exit criterion says sleep ≥ 7 h; a rested pass at 90% recall beats a tired pass at 100%.</details>

---

## Build block (4 h) — capstone Day 4: report, post, and the recorded dress rehearsal

Objective (Day 4 of the [capstone brief](../../../gpu-engineering-lab/03-scale-and-serve/week-12-capstone/README.md)), local:

- Write `REPORT.md` from `REPORT-template.md`: what I built, the numbers, **what didn't work**
  (specific failures = credibility), and "what I'd do with 8×H100".
- Draft the LinkedIn/blog post (300–500 words, one plot, repo link) — publish *after* the exam.
- **SHOW touchpoint (master plan): the recorded dress rehearsal.** One take, screen-recorded:
  the [demo](../../../k8s-ai-stack-demo/scripts/demo.sh) storyline (recording from the week-11
  run is fine for the cluster scenes) → the 5-minute repo tour from the capstone README script →
  a [mock-interview Q&A](../../../03_mock_interview_qa.md) drill, answered out loud, no notes.
  This recording is the master plan's "one recorded end-to-end run" checkbox.
- **DoD:** REPORT.md complete; post drafted; rehearsal recording saved and watchable (you watched ≥ the first 2 minutes back).
- Hint: record the tour BEFORE you feel ready — the second take after watching yourself is the one that lands under 6 minutes.

---

## Close the day (15 min)

- [ ] Anki: clear all due cards; tomorrow morning is skim-only.
- [ ] One line in [notes.md](notes.md): re-drill result + system check PASSED (or what you fixed).
- [ ] Lay out tomorrow: ID, water, room booked with the household, check-in time on the calendar.
- [ ] Local day — no instance check. **Sleep ≥ 7 h is an exit criterion, treat it like one.**
