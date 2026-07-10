# Week 4 · Day 4 — Targeted re-drill + logistics check

[← Master Plan](../../../MASTER-PLAN.md) · [Week 4 overview](plan.md) · [← previous day](day-3.md) · [next day →](day-5.md)

## Study block (2 h)

Last study day before the exam. The agenda was written yesterday: your miss list and domain tally. Work worst domain first, and be mechanical about it — re-drilling has a method.

### Leech pass (15 min)

Run flashcards, but today mark **leeches** — cards you've missed 3+ times. Don't just re-fail them: **rewrite** each leech. Leeches usually mean the card is bad (two facts in one card, ambiguous prompt), not that you're bad. Split them, add a concrete scenario to the prompt, or convert term→definition into scenario→choice.

### How to re-drill, by domain (60 min, worst first)

For each domain, the drill loop is the same: *re-read the relevant notes section → redo that week's self-check questions you got wrong → say the exit-criteria line out loud → make/fix the flashcard.* What differs is where the material lives:

- **Domain 1 — Essential AI Knowledge (38%)** → [week-1 notes](../week-1/notes.md) + [week-1 self-check](../week-1/self-check.md). Highest-yield re-drills: training vs inference workload characteristics; the software stack one-liners (CUDA / cuDNN / TensorRT / NGC / AI Enterprise — one sentence each, no overlap); GPU-vs-CPU "why parallel wins"; the AI lifecycle stages. Flashcard subset: the stack-component and lifecycle cards.
- **Domain 2 — AI Infrastructure (40%)** → [week-2 notes](../week-2/notes.md) + [week-3 notes](../week-3/notes.md), both self-checks ([wk2](../week-2/self-check.md) / [wk3](../week-3/self-check.md)). Highest-yield: the decision pairs — IB vs RoCE/Spectrum-X, NVLink vs PCIe, MIG-capable GPU tiers, on-prem vs cloud drivers — plus the numbers (10.2 kW, 40–120 kW racks, ~40 kW air limit, PUE direction, NDR 400G). Flashcard subset: every "customer wants X → recommend Y" card.
- **Domain 3 — AI Operations (22%)** → this week's [notes.md](notes.md), Day 1–2 sections. Highest-yield: the GPU-in-K8s chain, GPU Operator one-liner, Slurm vocabulary, DCGM-vs-nvidia-smi, ECC single/double, the MIG/time-slicing/vGPU triple. Flashcard subset: the ops one-liners.

Rule of thumb for time-splitting: number of misses × domain weight. Five Domain-2 misses outrank five Domain-3 misses.

### Fluency speed-run (30 min)

Read all three weeks' **exit-criteria checklists** ([wk1](../week-1/plan.md), [wk2](../week-2/plan.md), [wk3](../week-3/plan.md), [wk4](plan.md)) and answer each one *out loud, closed book*. Fluent = exam-ready; hesitant = re-read that notes section now. Speaking, not re-reading, is the test — recognition feels like knowledge but isn't.

### Logistics + proctoring system check (15 min — do not skip)

Work through the [booking checklist](../../tools/booking-checklist.md) end to end on the **machine and room you'll use Friday**:

- Confirm the appointment time in the portal; verify your **ID exactly matches your portal name**.
- Run the proctor's **system check** on the exam machine: webcam, mic, screen sharing, bandwidth, and the required browser/app installed and logged in.
- Plan the room: clean desk, no second monitor (or unplugged), phone out of the room, no one walking through, chargers plugged in.
- Know the check-in time: you'll log in **15 minutes early** Friday.

A failed system check tonight is an inconvenience; discovered Friday, it's a rescheduled exam.

### Quick check

1. What's the rule for prioritizing re-drill time across domains?
<details><summary>Answer</summary>Misses × domain weight, worst first. Domain 2 carries 40% (~20 questions), Domain 1 38%, Domain 3 22% — equal miss counts are not equal point risks.</details>

2. What do you do with a leech card besides reviewing it again?
<details><summary>Answer</summary>Rewrite it: split multi-fact cards, disambiguate the prompt, or convert it to a scenario→choice format. Repeated failure usually indicts the card's design, not your memory.</details>

3. Why is the exit-criteria speed-run done out loud?
<details><summary>Answer</summary>Recognition masquerades as knowledge when re-reading. Producing the answer verbally, closed book, is retrieval practice — the same act the exam demands — and instantly exposes what's only familiar rather than known.</details>

4. Name four things the proctoring system check must verify tonight.
<details><summary>Answer</summary>Webcam, microphone, screen-sharing capability, and bandwidth/software readiness (required browser or app installed and working) — on the exact machine and network you'll use Friday, plus ID-vs-portal-name match on the account.</details>

## Build block (4 h)

**rusty-kernels Day 4 — benchmarks.** [Project brief](../../../gpu-engineering-lab/01-foundations/week-04-pytorch-custom-ops/README.md)

- `python bench/bench_ops.py`: rows {512…16384} × hidden {768…8192}, fp32 + fp16, vs (a) eager and (b) `torch.compile`.
- Expect to beat eager clearly on memory-bound shapes and to roughly tie `torch.compile` — report honestly; explaining *why* compile ties you is the better interview story.
- Measure the v0 sync overhead at small shapes; attempt the stream-ordered upgrade only if the numbers justify it.
- `ncu` one kernel vs the eager sequence to show the DRAM-traffic delta.
- Definition of done: benchmark JSON + charts committed with both baselines; sync-overhead numbers recorded.
- Hint: exclude torch.compile's compilation time via warmup iterations and say so in the caption — otherwise your comparison is accidentally rigged.

## Close the day (15 min)

- Anki: quick normal pass only — the heavy lifting happened this morning.
- One "hardest thing today" line in [notes.md](notes.md) — plus "system check: PASS" once it's true.
- Blockers: none allowed to carry into tomorrow. Lay out ID, water, and the room tonight. Early night — sleep beats study from here.
