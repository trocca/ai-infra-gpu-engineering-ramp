# Week 12 Notes — Troubleshooting deep-dive + drills

## Storage performance

- Symptoms of storage-bound training:
- fio command bank + how to read output:
- Local NVMe vs Lustre/GPFS vs NFS trade-offs:
- Page cache effects on benchmarks:
- GPUDirect Storage:
- Standard fixes (staging, workers, sharded formats):

## NCCL / network debugging

- NCCL_DEBUG=INFO — what lines to look for:
- Transport selection ladder (P2P/NVLink → SHM → NET/IB → NET/Socket):
- NCCL_SOCKET_IFNAME / NCCL_IB_DISABLE / NCCL_P2P_DISABLE effects:
- Classic hang causes:
- nccl-tests invocation + bus BW interpretation:

## DCGM diagnostics

- dcgmi diag levels 1–4 (duration + what's added):
- dcgmi health / dmon / discovery:
- When to run r3 (pre-RMA, post-install burn-in):

## Xid table (memorize)

| Xid | Meaning | App or HW? | Action |
|---|---|---|---|
| 13 | | | |
| 31 | | | |
| 43 | | | |
| 48 | | | |
| 63/64 | | | |
| 74 | | | |
| 79 | | | |
| 94/95 | | | |

## Drill debriefs

### Drill 1 — GPU Operator (time: ___)
- Looked up:
### Drill 2 — MIG (time: ___)
- Looked up:
### Drill 3 — break/fix (time: ___)
- Looked up:
### Drill 4 — Slurm (time: ___)
- Looked up:
### Drill 5 — KAI (time: ___)
- Looked up:

## Mock exam results

- MCQ score: ___/30 in ___ min
- Missed questions + why:
- Lab scenarios — gaps:

## Exam-day quick sheet (fill by Thu night)

- GPU Operator component order:
- Slurm upgrade order:
- A100-40GB MIG profiles:
- cmsh modes:
- sacctmgr one-liners:
- nvidia-smi mig one-liners:
- dcgmi one-liners:
