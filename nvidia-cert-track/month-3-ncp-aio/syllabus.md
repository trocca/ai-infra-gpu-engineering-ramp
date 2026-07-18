# NCP-AIO / NCP-AIOL — NVIDIA-Certified Professional: AI Operations

> **Details verified July 2026 against the official exam page — re-check before booking.**
> Official page: https://www.nvidia.com/en-us/learn/certification/ai-operations-professional/
> NVIDIA's current page labels this exam code **NCP-AIOL**; this repo keeps the shorter
> **NCP-AIO** shorthand in folder names and weekly references.

## What this exam is

NCP-AIO / NCP-AIOL validates that you can **operate** NVIDIA AI infrastructure — deploy it, administer it,
run workloads on it, and fix it when it breaks. It is the professional-level counterpart to
NCA-AIIO and sits squarely on the stack you already demo: GPU Operator, MIG, Slurm, Run:ai/KAI,
NGC, DCGM.

**The single most important fact about this exam:** it is *not* MCQ-only.

| Fact | Value |
|---|---|
| Price | $500 USD |
| Duration | **120 minutes** |
| Format | **~30 multiple-choice questions + 3 hands-on lab exercises** in the same session |
| Validity | 2 years |
| Delivery | Online proctored |
| Level | Professional (recommended: 2–3 years admin experience) |

### Time pressure math

120 minutes for 30 MCQ **and** 3 labs. If you give each lab 25 minutes (75 min total), you have
**45 minutes for 30 MCQs — 90 seconds each**. There is no time to derive answers from first
principles. The MCQs must be near-automatic so the labs get the time they need. Conversely, a lab
you can't finish in 25 minutes on a good day is a lab you will fail under proctoring. This is why
weeks 9–12 are built around **doing, timed**, not reading.

**The 120-minute anatomy — labs get 75 minutes; MCQs must fit in 45 or they eat lab time.**

```
0'                 45'                                        115'    120'
|---- 30 MCQs -----|-------------- 3 hands-on labs ------------|-------|
  45 min              25 min x 3 = 75 min                        ~5 min
  90 s per question   read fully -> act -> verify, every step    flagged
  bank easy, flag rest                                           MCQs
```

The labs are real terminal exercises: expect things like configuring MIG on a node, writing/fixing
a Slurm job or partition, diagnosing a Pending GPU pod, or driving BCM's `cmsh`. Muscle memory for
`nvidia-smi mig`, `kubectl describe`, `scontrol`, `sinfo`, `dcgmi`, and `helm` is the whole game.

## Domains and weights

| # | Domain | Weight | Topics (from official blueprint) |
|---|---|---|---|
| 1 | Installation & Deployment | **31%** | BCM configuration, Slurm and Kubernetes cluster setup, cluster monitoring, patch management, user management, networking |
| 2 | Administration | **23%** | Slurm administration, datacenter architecture, NVIDIA Run:ai, Kubernetes administration, MIG configuration |
| 3 | Workload Management | **23%** | Training/inference deployment, resource allocation, NGC containers |
| 4 | Troubleshooting & Optimization | **23%** | Docker issues, fabric manager, BCM troubleshooting, storage performance, container issues |

**The weights at a glance — one domain is heavier: Installation & Deployment alone is nearly a third of the MCQs.**

```
Installation & Deployment   ################  31%   (week 9)
Administration              ############      23%   (week 10)
Workload Management         ############      23%   (week 11)
Troubleshooting & Optim.    ############      23%   (weeks 11-12)
```

## How this maps to what you already know

Your demo repo already covers KAI gang scheduling, Kubeflow TrainJob v2, MIG vs time-slicing vs
MPS, NCCL transports, DRA ResourceClaims, GPU Operator, and vLLM/Dynamo/NIM. That is roughly
**half the exam** (most of Administration + Workload Management + GPU parts of Troubleshooting).
The genuinely new ground is:

- **BCM (Base Command Manager)** — head node, provisioning, software images, `cmsh` / Base View.
  Biggest single blind spot; it anchors the 31% domain. No free tier — study docs hard (week 9).
- **Slurm** as an admin (not just a user): slurmctld/slurmd/slurmdbd, `gres.conf`, partitions,
  QoS, accounting, fairshare.
- **Run:ai as a product** — you know KAI (its open-sourced scheduler core); learn the product
  vocabulary on top: projects, departments, quotas, over-quota, fractional GPUs.
- **Fabric Manager / NVSwitch** and **storage performance** troubleshooting.

## Recommended preparation

1. **NVIDIA DLI courses** (the official prep path):
   - "AI Infrastructure and Operations Fundamentals" (free, good survey)
   - DLI workshops on Base Command Manager / cluster administration and on
     "Introduction to Deploying, Managing, and Administering AI Infrastructure" — check the
     current catalog at https://www.nvidia.com/en-us/training/
2. **BCM documentation** — Administrator Manual is the canonical source:
   https://docs.nvidia.com/base-command-manager/ (read Ch. on installation, node provisioning,
   user management, monitoring, and the Slurm/Kubernetes integration chapters)
3. **GPU Operator docs**: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
4. **Run:ai docs**: https://run-ai-docs.nvidia.com/ (projects, quotas, scheduling, fractions)
5. **Slurm docs**: https://slurm.schedmd.com/ (quickstart admin guide, gres.html, qos.html,
   sacctmgr, slurm.conf man page)
6. **DCGM docs**: https://docs.nvidia.com/datacenter/dcgm/latest/ (dcgmi diag levels)
7. This folder: `week-9/` … `week-12/` plans, `labs/`, `flashcards.csv`, `mock-exam.md`.

## Study calendar (2h/day × 5 days/week)

| Week | Dates (2026) | Focus |
|---|---|---|
| 9 | Sep 7 – Sep 11 | Installation & Deployment (31%): BCM, Slurm setup, K8s setup, networking, patching, users |
| 10 | Sep 14 – Sep 18 | Administration (23%): Slurm admin, Run:ai, K8s admin, MIG, datacenter architecture |
| 11 | Sep 21 – Sep 25 | Workload Management (23%) + Troubleshooting part 1: NGC, training/inference deploys, Docker/fabric-manager/driver failures |
| 12 | Sep 28 – Oct 2 | Troubleshooting deep-dive, timed lab drills, mock exam, **EXAM (end of week)** |

## Booking checklist

- [ ] Re-verify format/price/duration on the official page (link above) — details can change.
- [ ] Create/verify NVIDIA certification account (certification portal via nvidia.com/learn).
- [ ] Book the online-proctored slot for **end of week 12 (Fri 2026-10-02)** — book by end of
      week 10 so the date pressure is real.
- [ ] System check: webcam, mic, stable connection, supported browser/proctoring client.
- [ ] Quiet room, cleared desk, government ID ready.
- [ ] The labs run in a browser terminal — practice in a plain terminal, not your tricked-out
      shell (no aliases, no fzf, default vim/nano).
- [ ] Day before: skim `flashcards.csv` misses + re-run one timed lab. Sleep.
