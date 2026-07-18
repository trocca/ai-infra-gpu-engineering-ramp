# Week 12 Plan — Troubleshooting deep-dive + lab drills + EXAM

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

**Dates:** Mon 2026-09-28 → Fri 2026-10-02 · ~2h/day · **Exam: end of this week (Fri 2026-10-02)**
**Domains:** Troubleshooting & Optimization (23%) Mon–Tue; Wed–Thu are timed drills across ALL
domains (this is where the 3 hands-on lab exercises are won); Friday = exam.

## Prerequisites before Monday

- Companion lesson: [Week 12 companion — troubleshooting, capstone integration, and portfolio defense](../../../companion-lessons/week-12.md).
- Source reading: [HF Ultra-Scale Playbook — capstone defense and scale-up story](../../../references/hf-ultrascale-playbook.md#week-12---capstone-defense-and-scale-up-story).
- Math/support: utilization ratios, scaling efficiency, queue saturation, and before/after evidence.
- Systems/story support: triage by layer, timed drill discipline, capstone script reproducibility, and the 90-second repo tour.
- Gate: list the five most likely failure layers for a GPU training job and decide what gets cut if exam readiness and capstone polish collide.

---

## Day 1 (Mon) — Storage performance + NCCL/network debugging

- 0:00–0:45 — Storage performance: symptoms of storage-bound training (low GPU util, high
  dataloader wait), `fio` benchmarking (`fio --name=read --rw=read --bs=1M --size=10G
  --numjobs=4 --iodepth=32 --direct=1`), sequential vs random IO, local NVMe scratch vs
  parallel FS (Lustre/GPFS) vs NFS bottlenecks, page cache effects, GPUDirect Storage concept.
  Fixes: stage data locally, more dataloader workers, larger reads, sharded formats
  (webdataset/tfrecord).
- 0:45–1:30 — NCCL debugging (leans on your demo repo's NCCL transports work):
  `NCCL_DEBUG=INFO` / `NCCL_DEBUG_SUBSYS=INIT,NET` reading — which transport got picked
  (NVLink/P2P vs SHM vs NET/IB vs NET/Socket); `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE`,
  `NCCL_P2P_DISABLE` effects; hangs from asymmetric env or firewalled ports; `nccl-tests`
  (`all_reduce_perf -b 8 -e 1G -f 2 -g <ngpus>`) as the canonical fabric benchmark; bus BW
  interpretation.
- 1:30–2:00 — **Do:** lab-troubleshoot drill (c) — NCCL env misconfigurations on the lab VM.

## Day 2 (Tue) — DCGM diagnostics + Xid errors

- 0:00–0:45 — DCGM: `dcgmi discovery -l`, `dcgmi diag -r <1|2|3|4>` (1 ≈ seconds smoke test,
  2 ≈ a few min incl. PCIe + brief stress, 3 ≈ 30+ min full incl. memory/stress, 4 extended),
  what each level adds, `dcgmi health`, `dcgmi dmon`. Active vs passive health. When the
  answer is "run dcgmi diag -r 3 before RMA".
- 0:45–1:30 — Xid errors: where they appear (`dmesg`, nvidia-smi, DCGM
  `DCGM_FI_DEV_XID_ERRORS`), the ones worth memorizing:
  13 (app: graphics engine exception), 31 (app: GPU memory page fault), 43 (app abort),
  48 (double-bit ECC), 63/64 (row remap pending/failure), 74 (NVLink error),
  79 (GPU fallen off the bus — hardware/power/PCIe), 94/95 (contained/uncontained ECC).
  App-caused vs hardware-caused triage. ECC + row remapping + when to drain and RMA.
- 1:30–2:00 — **Do:** lab-troubleshoot drill (d) — dcgmi diag runs, find Xids in dmesg.

## Day 3 (Wed) — TIMED LAB DRILL DAY 1 (exam simulation)

Re-run labs cold, against the clock, on a fresh VM where possible. 25-minute hard cap each:

- Drill 1 (25 min): lab-gpu-operator from bare k3s → test CUDA pod Running. No notes.
- Drill 2 (25 min): lab-mig-config command sequence — if no A100 rented today, write every
  command + expected output by hand, then check against the lab file.
- Drill 3 (25 min): lab-troubleshoot drill (a)+(b) — break it yourself with the break script,
  then fix without notes.
- 0:15 debrief per drill: what did I look up? Put it on a flashcard.

## Day 4 (Thu) — TIMED LAB DRILL DAY 2 + mock exam

- Drill 4 (25 min): lab-slurm-basics — cold: bring up container cluster, add QOS, sbatch GPU
  job, show accounting.
- Drill 5 (25 min): lab-runai-kai — cold: queues + two gang jobs + demonstrate preemption.
- 0:50 — **`mock-exam.md` MCQ section, timed: 45 minutes for 30 questions.** Score it.
- 0:20 — Review misses; re-read the 3 written lab scenarios in mock-exam and mentally walk
  the solutions.

## Day 5 (Fri) — EXAM DAY

- Morning (30–45 min max, no new material): skim flashcard misses, the allocation-strategy
  table, cmsh command bank, Xid table, `nvidia-smi mig` + `sacctmgr` + `dcgmi` syntax.
- Logistics: ID, quiet room, proctoring client check done yesterday, plain terminal mindset.
- **Exam strategy (120 min, ~30 MCQ + 3 labs):**
  - Labs are where points hide — budget ~25 min/lab = 75 min, leaving ~45 min for MCQs
    (90 s each).
  - Do a fast MCQ pass first if the interface allows (bank the easy 60%), flag and move on —
    never sink 5 minutes into one question.
  - In labs: read the FULL task before typing; verify each step's output before the next;
    if something fails, one diagnostic (`describe`/`journalctl`/`-lgi`) then adapt — don't
    thrash.
  - Leave 5 min to revisit flagged MCQs.

---

## Exit criteria (Thursday night — exam-ready)

- [ ] All 5 labs completed cold within 25 min each (or hand-written perfectly for MIG if no
      A100 available).
- [ ] Mock exam ≥ 24/30 in ≤ 45 minutes.
- [ ] All 3 mock lab scenarios solvable on paper without reference.
- [ ] I can recite: GPU Operator component order, Slurm upgrade order, A100 MIG profiles,
      Xid 48/63/74/79 meanings, dcgmi diag levels, fabric manager signature, cmsh mode names.
- [ ] Proctoring system check passed; sleep ≥ 7h.
