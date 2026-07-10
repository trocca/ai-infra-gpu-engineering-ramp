# Week 4 Notes — AI Operations + Review

## 1. Cluster orchestration

### Kubernetes for AI
- How a pod requests a GPU (device plugin, resource name):
- What the GPU Operator installs/manages (list the components):
- Why containers matter for AI workloads (NGC tie-in):
- Where MIG/time-slicing config lives in K8s:

### Slurm
- What it is / where it comes from:
- Core concepts (job, partition, node, gres):
- sbatch vs srun (one line):
- Why HPC/training shops like it:

### Choosing between them
- Slurm is the better fit when…
- Kubernetes is the better fit when…
- How they coexist in real orgs:

### Related NVIDIA tooling
- Base Command Manager:
- Run:ai:
- NGC containers/Helm charts in the ops workflow:

## 2. GPU monitoring

### nvidia-smi
- What it shows:
- Limitations for fleet monitoring:
- Useful flags I tried (`-l`, `--query-gpu=`, `topo -m`):

### DCGM
- What it is:
- Active vs background health checks / `dcgmi diag`:
- DCGM exporter → Prometheus → Grafana pattern:
- Job-level statistics:

### Key metrics (define each + why it matters)
- GPU utilization:
- Memory used / utilization:
- Temperature:
- Power draw:
- ECC errors — single-bit (correctable):
- ECC errors — double-bit (uncorrectable):
- XID errors:
- NVLink errors:
- Clock throttling (thermal/power):

## 3. GPU sharing: MIG vs time-slicing vs vGPU

| | MIG | Time-slicing | vGPU |
|---|---|---|---|
| Mechanism | | | |
| Isolation (memory/faults) | | | |
| QoS guarantee | | | |
| GPU generations | | | |
| Licensing needed? | | | |
| Best-fit scenario | | | |

- MIG instance anatomy (GPU slices, max 7 on A100/H100):
- MIG + Kubernetes (how instances appear as resources):
- vGPU editions in one line (vWS / vCS idea):
- Decision phrases → answer ("strict isolation multi-tenant" → …, "VDI/VMs" → …, "cheap dev sharing" → …):

## 4. Data center management for AI (light)

- Provisioning/imaging at scale (Base Command Manager idea):
- Firmware/driver lifecycle:
- Health-check-before-job pattern (DCGM diag in prologue):

## 5. Mock exam debrief

- Score: ___ / 50
- Domain 1 misses (topics):
- Domain 2 misses (topics):
- Domain 3 misses (topics):
- Question types that tricked me:
- Fixed by (what I re-studied):

## Parking lot
-
