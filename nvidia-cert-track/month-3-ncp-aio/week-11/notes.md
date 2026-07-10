# Week 11 Notes — Workload Management + Troubleshooting (part 1)

## NGC

- Catalog object types:
- Framework container contents / naming scheme:
- NGC CLI config + key commands:
- docker login nvcr.io ($oauthtoken):
- K8s imagePullSecret for NGC:
- Private registry paths (org/team):
- Container CUDA vs host driver compatibility rule:

## Training deployment — Slurm

- sbatch GPU flags (--gres vs --gpus):
- Multi-node flags (--nodes, --ntasks-per-node) + srun:
- CUDA_VISIBLE_DEVICES handling:
- enroot / pyxis:
- salloc / interactive:

## Training deployment — Kubernetes

- Plain Job with GPU limits:
- Kubeflow TrainJob v2 / TrainingRuntime (demo repo cross-ref):
- Gang scheduling interaction with KAI:
- torchrun env plumbing (MASTER_ADDR etc.):

## Inference deployment

- Triton: model repo layout, config.pbtxt:
- Triton ports + health endpoints:
- Dynamic batching / instance groups:
- NIM: run command, NGC_API_KEY, profiles:
- NIM Operator / helm:
- Triton vs NIM vs raw vLLM — operator's decision:

## Resource allocation strategies (fill the table)

| Strategy | Isolation | Granularity | HW support | Use when |
|---|---|---|---|---|
| Whole GPU | | | | |
| MIG | | | | |
| Time-slicing | | | | |
| MPS | | | | |
| Run:ai fractions | | | | |
| DRA ResourceClaims | | | | |

## Troubleshooting: container runtime chain

- Chain: app → containerd → runtime class → nvidia-container-runtime → libnvidia-container → driver
- nvidia-ctk runtime configure:
- containerd config.toml key fields:
- "could not select device driver" root causes:
- Operator validator failures:

## Troubleshooting: driver/toolkit mismatch

- NVML "Driver/library version mismatch":
- CUDA too new for driver:
- Host OK / container broken checklist:

## Troubleshooting: fabric manager

- What FM does (NVSwitch routing):
- Failure signature:
- Version-match rule:
- Logs / systemctl:
- When it's NOT needed:

## Troubleshooting: BCM

- Node stuck in node-installer:
- CMDaemon logs:
- Healthchecks:
- imageupdate --dry-run:

## Misses / to-revisit

-
