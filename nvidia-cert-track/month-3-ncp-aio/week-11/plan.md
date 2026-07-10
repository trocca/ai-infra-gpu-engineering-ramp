# Week 11 Plan — Workload Management (23%) + Troubleshooting part 1

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

**Dates:** Mon 2026-09-21 → Fri 2026-09-25 · ~2h/day
**Domains:** Workload Management (23%) all week; Troubleshooting & Optimization (23%) starts
Thursday. Your vLLM/NIM/Triton/TrainJob demo work covers much of this — focus on the *operator's*
view (pull, deploy, allocate, verify) rather than the evangelist pitch.

## Prerequisites before Monday

- Companion lesson: [Week 11 companion — Kubernetes GPU serving, scheduling invariants, and observability support](../../../companion-lessons/week-11.md).
- Math/support: queue depth, request rate, TTFT/p95 latency, GPU utilization, memory used, and HPA signal quality.
- Systems support: GPU Operator, device plugin, DCGM exporter, readiness probes, pod events, image pulls, and resource names.
- Gate: write the shortest debug sequence for a Pending GPU pod and pick three metrics that prove an inference service is healthy.

---

## Day 1 (Mon) — NGC [Workload Mgmt]

- 0:00–0:30 — NGC catalog anatomy: containers (`nvcr.io/nvidia/pytorch:25.06-py3` naming),
  models, resources, Helm charts. Deep-learning framework containers vs NIM containers.
  What's inside a framework container (CUDA, cuDNN, NCCL, framework — pinned compatible set).
- 0:30–1:15 — NGC CLI + registry auth: `ngc config set` (API key), `ngc registry image list`,
  `ngc registry model download-version`. Docker login: user `$oauthtoken`, password = API key,
  `docker login nvcr.io`. K8s: `kubectl create secret docker-registry ngc-secret
  --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=$NGC_API_KEY` +
  imagePullSecrets. Private registry: `nvcr.io/<org>/<team>/...`.
- 1:15–2:00 — **Do:** on the lab VM — NGC CLI install, pull a PyTorch container, run
  `nvidia-smi` inside it. Note the container's CUDA version vs host driver
  (forward-compat rule: driver ≥ what the CUDA runtime needs).

## Day 2 (Tue) — Training deployment [Workload Mgmt]

- 0:00–0:45 — Slurm side: sbatch scripts with GPUs — `#SBATCH --gres=gpu:2` / `--gpus=2`,
  `--nodes`, `--ntasks-per-node`, `srun` inside; `CUDA_VISIBLE_DEVICES` is set by Slurm per
  step. Containers under Slurm: enroot/pyxis (`srun --container-image=`) as the NGC-blessed
  path. Interactive: `salloc`/`srun --pty`.
- 0:45–1:30 — K8s side: plain Job with `resources.limits.nvidia.com/gpu`, restartPolicy;
  then Kubeflow TrainJob v2 (your demo repo!) — TrainJob + TrainingRuntime split, how it
  gang-schedules under KAI. Multi-node NCCL env plumbing (MASTER_ADDR, WORLD_SIZE via
  torchrun).
- 1:30–2:00 — **Do:** lab-slurm-basics Part B (sbatch a real GPU job, watch `squeue`, read
  `sacct -j` output) if not finished; else re-run your TrainJob demo and narrate it in NCP-AIO
  vocabulary.

## Day 3 (Wed) — Inference deployment + resource allocation [Workload Mgmt]

- 0:00–0:45 — Triton Inference Server ops view: model repository layout
  (`<repo>/<model>/<version>/model.*` + `config.pbtxt`), ports (8000 HTTP / 8001 gRPC /
  8002 metrics), readiness (`/v2/health/ready`), dynamic batching, concurrent model instances.
- 0:45–1:30 — NIM ops view: pull from NGC (needs `NGC_API_KEY`), `docker run` with
  `--gpus all -e NGC_API_KEY -p 8000:8000`, OpenAI-compatible `/v1/chat/completions`;
  NIM Operator / helm on K8s; model profiles auto-selected per GPU. Contrast Triton vs NIM vs
  raw vLLM (your demo stack) — when an admin picks which.
- 1:30–2:00 — Resource allocation strategies recap table (build it in notes.md): whole GPU vs
  MIG vs time-slicing vs MPS vs fractions (Run:ai) vs DRA ResourceClaims (your demo repo;
  structured parameters GA since K8s 1.34) — isolation, granularity, when to use.

## Day 4 (Thu) — Troubleshooting: Docker/containerd + container runtime [Troubleshooting]

- 0:00–0:45 — The GPU container plumbing chain and where it breaks: app → containerd →
  runtime class `nvidia` → nvidia-container-runtime → libnvidia-container → driver.
  `nvidia-ctk runtime configure --runtime=docker|containerd`, `/etc/nvidia-container-runtime/
  config.toml`, containerd `config.toml` default_runtime_name. Classic failures:
  "could not select device driver with capabilities gpu", missing runtimeClassName, operator
  validator CrashLoopBackOff.
- 0:45–1:30 — Driver/toolkit mismatches: "Failed to initialize NVML: Driver/library version
  mismatch" (kernel module vs userspace libs after driver upgrade without reboot);
  CUDA-version-too-new for driver; `nvidia-smi` works on host but not in container.
- 1:30–2:00 — **Do:** `labs/lab-troubleshoot.md` drills (a) runtime class and (b) version
  mismatch simulation.

## Day 5 (Fri) — Troubleshooting: fabric manager + review [Troubleshooting]

- 0:00–0:45 — Fabric Manager: what it is (userspace service programming NVSwitch routing on
  HGX/DGX systems), `nvidia-fabricmanager` systemd unit, **version must exactly match the
  driver**. Symptom signature: `nvidia-smi` sees GPUs fine, but CUDA apps fail with
  "system not yet initialized" / CUDA init errors on 8-GPU NVSwitch boxes. Check
  `systemctl status nvidia-fabricmanager`, `/var/log/fabricmanager.log`. Not needed on
  PCIe-only or single-GPU nodes.
- 0:45–1:15 — BCM troubleshooting basics from the manual: node stuck in installer (check
  provisioning network/DHCP/TFTP), CMDaemon logs (`/var/log/cmdaemon`), healthchecks marking
  nodes down, `imageupdate` dry-run (`--dry-run`) to preview sync damage.
- 1:15–2:00 — `self-check.md`; update flashcards; book confirmation check (exam is next week
  Friday).

---

## Exit criteria (Friday) — "I can DO"

- [ ] I can authenticate to nvcr.io three ways (docker login, NGC CLI, K8s imagePullSecret)
      from memory.
- [ ] I can write an sbatch script for a 1-node 2-GPU job and a K8s Job manifest for 1 GPU
      in < 5 min each, no docs.
- [ ] I can stand up Triton or a NIM container and prove readiness with curl.
- [ ] I can fill in the full allocation-strategy table (MIG / time-slicing / MPS / fractions /
      DRA) with isolation + use-case per row, from memory.
- [ ] Given "nvidia-smi works on host, container can't see GPU", I list the exact 4-step check
      sequence (runtime class → toolkit config → device plugin → operator validator).
- [ ] I can state the fabric manager failure signature and its two most common causes
      (service not running, version ≠ driver version).
- [ ] Self-check ≥ 15/18.
