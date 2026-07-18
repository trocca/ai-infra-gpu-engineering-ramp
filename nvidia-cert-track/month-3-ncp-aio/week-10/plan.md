# Week 10 Plan — Administration (Domain 2, 23%)

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

**Dates:** Mon 2026-09-14 → Fri 2026-09-18 · ~2h/day
**Domain:** Administration — 23%. Heavy overlap with your demo repo (KAI, MIG, K8s) — use that:
your job this week is mapping what you already demo onto **exam vocabulary** (Run:ai product
terms, Slurm admin verbs) and drilling the admin commands.

**Also this week:** book the exam slot for Fri 2026-10-02 (see syllabus checklist).

## Prerequisites before Monday

- Companion lesson: [Week 10 companion — tensor/pipeline parallelism math and admin command support](../../../companion-lessons/week-10.md).
- Source reading: [HF Ultra-Scale Playbook — model parallelism and 5D strategy](../../../references/hf-ultrascale-playbook.md#week-10---model-parallelism-and-5d-strategy).
- Math support: row-parallel vs column-parallel linear layers, collective placement, and pipeline bubble efficiency.
- Systems support: Slurm drain/resume, Kubernetes cordon/drain, Run:ai/KAI vocabulary, MIG `single` vs `mixed`.
- Gate: compute pipeline efficiency for `p=4`, `m=4` and `m=16`, then highlight every command you need to drill this week.

---

## Day 1 (Mon) — Slurm administration

- 0:00–0:45 — Partitions & node state ops: `scontrol show partition/node`, drain/resume
  (`scontrol update nodename=n1 state=drain reason="maint"`), `sinfo` state codes (idle, mix,
  alloc, drain, down), `scontrol update partitionname=...`. Holding/releasing jobs, `scancel`.
- 0:45–1:30 — Accounting & policy: slurmdbd associations tree (cluster → account → user),
  `sacctmgr` (add/modify account, user, QOS), QOS knobs (Priority, MaxWall, MaxTRESPerUser,
  preemption via `PreemptType=preempt/qos`), fairshare (`PriorityType=priority/multifactor`,
  weights). `sacct` vs `sreport`. Docs: https://slurm.schedmd.com/qos.html, accounting.html.
- 1:30–2:00 — **Do:** lab-slurm-basics Part C (accounting + QOS on the container cluster):
  create account/user/QOS, submit with `--qos`, verify with `sacct`.

## Day 2 (Tue) — Run:ai (and its KAI core)

- 0:00–0:45 — Run:ai product model, from docs (https://run-ai-docs.nvidia.com/): control plane
  + cluster; **projects** (map to namespaces) with **GPU quotas**; **departments** above
  projects; **over-quota** (use idle GPUs beyond quota, reclaimable); preemptible vs
  non-preemptible workloads; workload types (interactive/training/inference); node pools.
- 0:45–1:15 — Fractional GPUs in Run:ai: request fractions (e.g. 0.5 GPU) or explicit GPU
  memory; multiple pods share one GPU with software memory enforcement. Contrast with your
  demo repo's MIG/time-slicing/MPS story — Run:ai fractions ≈ managed time-slicing + memory
  limits, no hardware isolation.
- 1:15–2:00 — Map Run:ai → KAI (Run:ai's open-sourced scheduler, which you already demo):
  project+quota ↔ KAI queue with quota/over-quota weight; gang scheduling ↔ PodGroup;
  preemption ↔ queue priority reclaim. **Do:** start `labs/lab-runai-kai.md` (install KAI,
  create queues).

## Day 3 (Wed) — Kubernetes administration

- 0:00–0:45 — Namespaces, RBAC: Role vs ClusterRole, RoleBinding, ServiceAccounts;
  `kubectl auth can-i --as`. Practice writing a Role that allows pod create in one namespace.
- 0:45–1:30 — ResourceQuota (incl. `requests.nvidia.com/gpu: "4"`) and LimitRange; node
  maintenance: `kubectl cordon/drain/uncordon` (drain honors PodDisruptionBudgets); taints &
  tolerations vs nodeSelector/affinity for GPU nodes.
- 1:30–2:00 — **Do:** finish lab-runai-kai (two competing gang jobs, watch preemption); apply
  a GPU ResourceQuota to a namespace and prove a 2nd pod is rejected.

## Day 4 (Thu) — MIG configuration

- 0:00–0:30 — MIG concepts refresh against your demo repo: GPU instances (GI) vs compute
  instances (CI), profiles (A100-40GB: 1g.5gb ×7, 2g.10gb, 3g.20gb ×2, 4g.20gb, 7g.40gb),
  hardware isolation (SM + memory + L2 slices). L4/L40S do NOT support MIG; A100/H100/A30/
  B200 do.
- 0:30–1:15 — Manual flow: `nvidia-smi -i 0 -mig 1` (reset/drain required), `nvidia-smi mig
  -lgip`, `-cgi 3g.20gb,3g.20gb -C`, `-lgi`, `-lci`, `-dci -dgi`. K8s flow: MIG Manager via
  GPU Operator — label `nvidia.com/mig.config=all-3g.20gb`, `mig.strategy=single|mixed`
  (single: uniform profiles exposed as `nvidia.com/gpu`; mixed: `nvidia.com/mig-3g.20gb`).
  `nvidia-mig-parted` is the underlying tool. Docs: GPU Operator "MIG" page.
- 1:15–2:00 — **Do:** `labs/lab-mig-config.md` (rent 1×A100 spot for ~2h; if budget says no,
  do the read-along path and hand-write every command).

## Day 5 (Fri) — Datacenter architecture + review

- 0:00–0:45 — Datacenter architecture for AI: DGX/HGX node anatomy (8×GPU, NVSwitch,
  8×IB rails), SuperPOD scalable unit concept, leaf-spine / fat-tree fabrics, rail-optimized
  topology; power/cooling constraints (air vs liquid, ~10–120 kW/rack trend); storage tiers
  (local NVMe scratch, parallel FS like Lustre/GPFS, object).
- 0:45–1:15 — Where each management plane sits: BCM on the management network, BMC out-of-band,
  Slurm/K8s control on management, NCCL on the compute fabric — one mental diagram.
- 1:15–2:00 — `self-check.md` closed-book; log misses; flashcards for domains 1–2.

---

## Exit criteria (Friday) — "I can DO"

- [ ] I can drain a Slurm node, create a QOS with a wall-time limit, attach a user to an
      account, and verify with `sacct` — all without docs (did it in the lab).
- [ ] I can enable MIG and create 2× `3g.20gb` instances with `nvidia-smi mig` from memory,
      and do the same declaratively via the MIG Manager node label.
- [ ] I can explain Run:ai projects/quota/over-quota/preemption and point to the exact KAI
      queue behavior I demo that implements each.
- [ ] I can write a Role+RoleBinding and a GPU ResourceQuota from scratch in < 5 min.
- [ ] I can sketch a SuperPOD-style datacenter (networks, storage tiers, mgmt planes) on paper.
- [ ] Self-check score ≥ 15/18. Exam slot is booked.
