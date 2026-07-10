# Week 4 Plan — AI Operations (Domain 3, 22%) + full review + EXAM

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

Dates: Mon 2026-08-03 → Fri 2026-08-07 · 5 days × ~2 h · **Exam booked end of this week**

Domain 3 is the smallest (22%, ~11 questions) and the closest to your day job (Kubernetes, GPU Operator, DCGM) — two days of new material, then review, mock, and exam. Keep flashcards daily, all domains.

## Prerequisites before Monday

- Companion lesson: [Week 04 companion — operations primitives, normalization math, and PyTorch extension support](../../../companion-lessons/week-04.md).
- Math support: stable softmax, LayerNorm vs RMSNorm, and why subtracting `max(x)` preserves softmax.
- Systems/programming support: Kubernetes and Slurm primitives, GPU Operator components, PyO3/maturin, and DLPack/data-pointer safety.
- Gate: write the one-line difference between MIG and time-slicing, then decide what gets lighter because this is exam week.

---

## Day 1 (Mon) — Cluster orchestration: Kubernetes and Slurm
**Domain served: AI Operations (22%)**

- (15 min) Flashcards (all domains).
- (55 min) Into `notes.md`:
  - **Kubernetes for AI**: containers as the packaging unit; how GPUs surface in K8s (device plugin advertises `nvidia.com/gpu`, pods request it in `resources.limits`); **GPU Operator** = installs/manages driver, container toolkit, device plugin, DCGM exporter, MIG manager as one operator (you know this — write the exam-level one-liner)
  - **Slurm**: the HPC batch scheduler — jobs, partitions, queues; `sbatch`/`srun`; gres for GPUs; strong at large multi-node batch training with topology awareness
  - **K8s vs Slurm framing**: Slurm = batch HPC-style training clusters; K8s = services/inference + cloud-native ML platforms; many shops run both (or K8s with batch schedulers on top — your KAI Scheduler world)
- (35 min) Skim GPU Operator docs overview + a "Slurm vs Kubernetes for ML" NVIDIA/community article. Read the NGC Helm chart page for GPU Operator.
- (15 min) One-liners for: NGC containers as the runtime artifact, Base Command Manager (cluster provisioning/management), Run:ai (K8s GPU orchestration/scheduling layer NVIDIA acquired).

## Day 2 (Tue) — GPU monitoring + MIG / time-slicing / vGPU
**Domain served: AI Operations (22%)**

- (15 min) Flashcards.
- (45 min) Monitoring into `notes.md`:
  - **nvidia-smi**: on-node CLI — utilization, memory used, temperature, power, processes, ECC error counts; good for spot checks, not fleet monitoring
  - **DCGM** (Data Center GPU Manager): fleet-grade — background health checks, active diagnostics (`dcgmi diag`), policy/alerting, job-level stats; **DCGM exporter** → Prometheus/Grafana (the standard K8s pattern)
  - Key metrics to know cold: GPU utilization, memory utilization/used, temperature, power draw, **ECC errors** (single-bit correctable vs double-bit uncorrectable), **XID errors** (driver-reported error events), NVLink errors, clock throttling reasons (thermal/power)
- (45 min) Sharing technologies into `notes.md` — the classic exam triple:
  - **MIG**: hardware partitioning (Ampere+) into up to 7 instances with dedicated SM/memory slices → guaranteed isolation & QoS; best for inference/multi-tenant
  - **Time-slicing**: software sharing, GPUs context-switch between processes; no memory isolation, no QoS guarantee; fine for dev/burst, oversubscription
  - **vGPU**: virtualization layer — GPU shared/partitioned across VMs with a licensed driver stack (vWS/vCS); the VMware/VDI/enterprise-virtualization answer
  - Decision drill: isolation+QoS bare metal/K8s → MIG; VMs/VDI → vGPU; cheap dev sharing → time-slicing
- (15 min) Skim DCGM user guide intro + MIG user guide intro.

## Day 3 (Wed) — MOCK EXAM (closed book)
**Domain served: all three**

- (10 min) Flashcards warm-up.
- (60 min) Take `../mock-exam.md` under real conditions: 50 questions, 60-minute timer, no notes, no lookups. Write answers on paper/separate file — don't peek at the key.
- (45 min) Score it against the key. For every miss AND every lucky guess: read the explanation, find the topic in your notes, mark it for Day 4.
- Target: ≥ 80% (40/50). Below 70%: prioritize weakest domain tomorrow and consider whether to use the full 2 h Thursday on it.

## Day 4 (Thu) — Gap-filling review
**Domain served: whatever the mock exposed**

- (15 min) Flashcards — mark leeches (cards missed 3+ times), rewrite them.
- (60 min) Attack your mock-exam miss list, worst domain first. Re-read the relevant `notes.md` sections and source docs. Redo the self-check questions you got wrong across weeks 1–3.
- (30 min) Speed run all three weeks' exit-criteria checklists out loud — anything you can't say fluently, review now.
- (15 min) Logistics check: confirm exam appointment time, run the proctoring system check on the exam machine, prep ID, plan the room.

## Day 5 (Fri) — Light review + EXAM
**Domain served: all**

- (20 min, morning) Flashcards only — no new material on exam day.
- (20 min) Read your own `notes.md` summaries once, calmly.
- Exam hygiene: log in 15 min early; ID ready; clean desk; phone out of the room.
- Pacing math: 50 questions / 60 min = ~70 s each. First pass: answer everything, flag doubts. Second pass: flagged only. Never leave blanks — no penalty for guessing.
- Question strategy: the exam rewards judgment — eliminate the two obviously-wrong options first, then choose between the last two based on the *scenario constraint* (the "must", "lowest latency", "multi-tenant", "no code changes" phrase is the key).
- After: results, then breathe. Badge arrives via the cert portal.

---

## Exit criteria — before exam day (end of Day 4)

- [ ] I can explain how a pod gets a GPU in Kubernetes (device plugin, resource request) and what the GPU Operator automates
- [ ] I can position Slurm vs Kubernetes for a given customer workload in two sentences
- [ ] I can name what nvidia-smi shows and why DCGM (+ exporter → Prometheus) is the fleet answer
- [ ] I can list 6+ GPU health metrics and explain ECC single- vs double-bit and what an XID is
- [ ] I can recommend MIG vs time-slicing vs vGPU for three different scenarios with reasons
- [ ] I scored ≥ 80% on the mock exam (or closed every gap it exposed)
- [ ] I did the proctoring system check and my ID matches my portal name
- [ ] Weeks 1–3 exit criteria all still hold on a spot check
