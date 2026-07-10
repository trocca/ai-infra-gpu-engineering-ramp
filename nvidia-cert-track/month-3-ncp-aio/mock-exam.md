# NCP-AIO Mock Exam

**Format mirrors the real thing:** 30 multiple-choice questions + 3 lab-scenario exercises.
**Timing drill:** 45 minutes for the MCQs (90 s each), then up to 25 min per lab scenario on
paper. Answer key and full lab solutions are at the BOTTOM — no peeking until scored.

Question mix by domain: Installation & Deployment ×9 (Q1–9), Administration ×7 (Q10–16),
Workload Management ×7 (Q17–23), Troubleshooting & Optimization ×7 (Q24–30).

---

## Section 1 — Multiple choice

**Q1.** You must apply a kernel security patch to 400 BCM-managed compute nodes. The correct
procedure is:

- A. SSH to each node with pdsh and run `apt upgrade`, then reboot in batches
- B. Update packages inside the assigned software image via `cm-chroot-sw-img`, then roll nodes through `imageupdate`/reboot by category
- C. Enable unattended-upgrades in the node image so nodes patch themselves nightly
- D. Rebuild the head node with the patched kernel; compute nodes inherit it at next PXE boot

**Q2.** A freshly racked node PXE-boots, loads the BCM node-installer, and then sits there
indefinitely. `cmsh device list` shows the node as `installing`. Which is the LEAST likely
cause?

- A. The node's MAC is matched to the wrong category with an invalid disk layout
- B. The software image sync is failing (check node-installer console via BMC)
- C. The compute fabric (InfiniBand) is down
- D. DHCP/TFTP responses from the provisioning interface aren't reaching the node

**Q3.** In cmsh you changed a node's category with `set category gpu-nodes` but after closing
the session the node still has the old category. Why?

- A. Category changes require Base View, not cmsh
- B. You didn't run `commit` before exiting
- C. The node must be powered off before category changes
- D. `set` only works in category mode, not device mode

**Q4.** During Slurm cluster setup, jobs submitted from the login node fail instantly with
authentication errors, but `sinfo` works from the controller. Two compute nodes also log
"Munge decode failed". Most likely cause?

- A. slurmdbd is not running
- B. gres.conf is missing on the compute nodes
- C. munge.key differs (or clocks are skewed) between nodes
- D. SelectType is set to select/linear

**Q5.** You are writing `slurm.conf` for nodes with 4 A100s each and want per-GPU scheduling
with CPU and memory tracking. Which pair is required?

- A. `SelectType=select/cons_tres` and `GresTypes=gpu` with `Gres=gpu:a100:4` on node lines
- B. `SelectType=select/linear` and `OverSubscribe=YES`
- C. `ProctrackType=proctrack/cgroup` and `TaskPlugin=task/affinity`
- D. `PriorityType=priority/multifactor` and `AccountingStorageType=accounting_storage/slurmdbd`

**Q6.** After `kubeadm init` completes on a new control plane, `kubectl get nodes` shows the
node `NotReady` and CoreDNS pods are `Pending`. What's the standard next step?

- A. Run `kubeadm join` on the control plane to register it
- B. Install a CNI network plugin matching the pod CIDR
- C. Remove the control-plane taint
- D. Restart kubelet with `--fail-swap-on=false`

**Q7.** You install GPU Operator on nodes that already run DGX OS with a pre-installed driver.
Which Helm setting is correct?

- A. `--set toolkit.enabled=false`
- B. `--set driver.enabled=false`
- C. `--set devicePlugin.enabled=false`
- D. `--set mig.strategy=none`

**Q8.** In the GPU Operator stack, which component is responsible for the node advertising
`nvidia.com/gpu: 8` to the Kubernetes scheduler?

- A. GPU Feature Discovery
- B. The NVIDIA device plugin DaemonSet
- C. DCGM exporter
- D. The container toolkit DaemonSet

**Q9.** Which statement about a well-designed GPU cluster's networks is correct?

- A. BMC/IPMI traffic should share the compute fabric to reduce cabling
- B. NCCL inter-node traffic should run on the management network to keep the fabric free for storage
- C. Provisioning (PXE) runs on the management network; NCCL runs on the IB/RoCE compute fabric; BMC is isolated out-of-band
- D. Storage traffic must always have a dedicated physical fabric, never shared with compute

**Q10.** A GPU node needs firmware maintenance. Running jobs must finish, but no new jobs may
start. Which command?

- A. `scontrol update nodename=gpu07 state=down reason="fw"`
- B. `scontrol update nodename=gpu07 state=drain reason="fw"`
- C. `scancel --nodelist=gpu07 && sinfo --set-state=maint`
- D. `sacctmgr modify node gpu07 set state=maintenance`

**Q11.** Users in account `prod` must never run jobs longer than 4 hours, regardless of
partition limits. The cleanest mechanism is:

- A. A cron job that scancels long jobs
- B. `PartitionName=... MaxTime=04:00:00` on every partition
- C. A QOS with `MaxWall=04:00:00` attached to the `prod` account, with QOS enforcement enabled
- D. Setting `DefaultTime=04:00:00` in slurm.conf

**Q12.** In Run:ai, project `research` (quota: 8 GPUs) is using 12 GPUs while the cluster is
idle. Project `prod` (quota: 8) now submits an 8-GPU non-preemptible workload but only 4 GPUs
are free. What happens?

- A. prod's workload queues until research's jobs finish naturally
- B. The scheduler reclaims: research's over-quota (preemptible) workloads are evicted until prod's 8 in-quota GPUs are satisfied
- C. prod's workload is rejected because the cluster is over capacity
- D. research keeps 12 GPUs because over-quota usage is protected once granted

**Q13.** Which Run:ai capability has NO hardware isolation between workloads sharing a GPU?

- A. Node pools
- B. MIG-backed workloads
- C. Fractional GPUs
- D. Departments

**Q14.** A user needs to list pods in every namespace but modify nothing. The correct RBAC
setup is:

- A. Role with get/list/watch on pods + RoleBinding in each namespace
- B. ClusterRole with get/list/watch on pods + ClusterRoleBinding
- C. ClusterRole with `*` on pods + RoleBinding in kube-system
- D. Role with get/list/watch on pods + ClusterRoleBinding

**Q15.** On an A100-40GB you need the largest possible number of fully isolated instances for
small inference services. Which MIG layout?

- A. 2 × 3g.20gb
- B. 7 × 1g.5gb
- C. 4 × 2g.10gb
- D. 8 × 1g.5gb

**Q16.** With GPU Operator managing MIG in `mixed` strategy, a node is labeled
`nvidia.com/mig.config=all-2g.10gb`. What resource do pods request?

- A. `nvidia.com/gpu: 1`
- B. `nvidia.com/mig-2g.10gb: 1`
- C. `nvidia.com/a100-mig: 1`
- D. `runai/fraction: 0.28`

**Q17.** Your CI pipeline must pull `nvcr.io/yourorg/team-ml/trainer:1.4` from your NGC private
registry into Kubernetes. Which is required?

- A. A docker-registry secret with username `$oauthtoken` and your NGC API key, referenced via imagePullSecrets
- B. An NGC CLI installation on every node
- C. Mirroring the image to Docker Hub since nvcr.io is public-only
- D. A CDI spec granting registry access

**Q18.** Why do teams standardize on NGC framework containers (e.g. `nvidia/pytorch:25.06-py3`)
instead of pip-installing PyTorch on hosts?

- A. NGC containers include the kernel driver, so hosts need nothing installed
- B. The container pins a co-tested CUDA/cuDNN/NCCL/framework stack; only a compatible host driver is needed
- C. pip wheels cannot use GPUs
- D. NGC containers are the only images the GPU Operator can schedule

**Q19.** A user requests `sbatch --nodes=2 --gres=gpu:4 train.sh` on a cluster with 4 GPUs per
node. How many GPUs does the job get?

- A. 4 total, 2 per node
- B. 8 total, 4 per node
- C. 4 total, on one node only
- D. The job is rejected as ambiguous

**Q20.** For a Triton deployment, model `bert` version 3 in ONNX format must live at:

- A. `models/bert/3/model.onnx` with `models/bert/config.pbtxt`
- B. `models/bert-v3/model.onnx` with inline config
- C. `models/3/bert/model.onnx`
- D. `models/bert/model.onnx` — versions are handled by image tags

**Q21.** An admin must serve a Llama-family model with minimal engineering effort, enterprise
support, and an OpenAI-compatible API, on NVIDIA GPUs. The most direct choice is:

- A. Build a TensorRT-LLM engine manually and wrap it in Flask
- B. Deploy the corresponding NIM container from NGC
- C. Run the training container and enable eval mode
- D. Use Triton with a hand-written Python backend

**Q22.** A distributed PyTorch TrainJob needs 4 pods × 1 GPU with all-or-nothing placement so
partially scheduled jobs don't deadlock the cluster. This requirement is called:

- A. Bin packing
- B. Gang scheduling
- C. Priority preemption
- D. Topology-aware placement

**Q23.** 30 data scientists run mostly-idle Jupyter notebooks that occasionally need short GPU
bursts, on 8 L4 GPUs. The most appropriate sharing strategy is:

- A. MIG, 7 slices per L4
- B. One dedicated L4 per top-8 scientist, queue for the rest
- C. Time-slicing (or Run:ai fractions) to oversubscribe each GPU
- D. MPS with one daemon per user across all GPUs

**Q24.** `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` returns
`could not select device driver "" with capabilities: [[gpu]]`. `nvidia-smi` works on the host.
The fix is:

- A. Reinstall the NVIDIA driver
- B. `nvidia-ctk runtime configure --runtime=docker` and restart docker (install toolkit if absent)
- C. Add the user to the `docker` group
- D. Set `NVIDIA_VISIBLE_DEVICES=all` in the shell

**Q25.** After an in-place driver package upgrade (no reboot), every command reports
`Failed to initialize NVML: Driver/library version mismatch`. Fastest reliable fix?

- A. Reinstall CUDA toolkit
- B. Downgrade nvidia-container-toolkit
- C. Reboot the node (or unload/reload the nvidia kernel modules)
- D. Run `nvidia-smi -r` to reset the GPUs

**Q26.** On an HGX 8×H100 server, `nvidia-smi` shows all GPUs healthy, but every CUDA program
exits at initialization with "system not yet initialized". First check?

- A. `systemctl status nvidia-fabricmanager` and that its version matches the driver
- B. `dcgmi diag -r 3`
- C. Whether MIG mode was accidentally enabled
- D. NCCL_SOCKET_IFNAME setting

**Q27.** dmesg on a node shows `NVRM: Xid ... : 79, GPU has fallen off the bus`. The correct
characterization is:

- A. Application bug — restart the job with smaller batch size
- B. Hardware/power/PCIe event — drain the node and investigate physically (sensors, reseat, RMA path)
- C. Driver/library mismatch — reboot fixes it permanently
- D. Normal during MIG reconfiguration

**Q28.** A GPU logged Xid 63 (row remapped, pending). Correct operational response?

- A. Immediate RMA — the GPU is failing
- B. Nothing — Xid 63 is informational only
- C. Drain at a convenient window and reset/reboot the GPU to activate the remap, then return to service and monitor
- D. Disable ECC to prevent recurrence

**Q29.** A 2-node NCCL training job hangs at initialization. `NCCL_DEBUG=INFO` on node A shows
it advertising the IP of `docker0`. The fix is:

- A. `NCCL_P2P_DISABLE=1`
- B. Set `NCCL_SOCKET_IFNAME` to the real cluster interface (or exclude with `^docker,lo`) on all ranks
- C. Increase `NCCL_TIMEOUT`
- D. `NCCL_IB_DISABLE=1`

**Q30.** Training throughput on one node is half that of identical peers. `dcgmi diag -r 2`
flags low PCIe bandwidth on GPU 3. Best next check?

- A. `nvidia-smi -q` link generation/width for GPU 3 (downtrained x16→x4 or Gen4→Gen1 indicates slot/riser/BIOS issue)
- B. `NCCL_DEBUG=INFO` output
- C. fio against the scratch disk
- D. `kubectl describe node` allocatable resources

---

## Section 2 — Lab scenarios (write full solutions before checking)

### Lab A — MIG provisioning under time pressure

A bare-metal node has one A100-40GB (GPU 0), driver installed, no MIG configured. Task:
configure exactly **2 × 3g.20gb** instances, prove they exist, and run
`nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L` in Docker against ONE of them.
List the exact command sequence including verification. (Assume the GPU is idle.)

### Lab B — Diagnose a Pending GPU pod

Given this (abbreviated) output, identify the root cause and give the fix commands.

```
$ kubectl describe pod trainer-0
...
    Limits:
      nvidia.com/gpu: 1
Events:
  Warning  FailedScheduling  0/3 nodes are available: 3 Insufficient nvidia.com/gpu.

$ kubectl describe node gpu-node-1 | grep -A8 Allocatable
Allocatable:
  cpu:                     31
  memory:                  250Gi
  nvidia.com/mig-3g.20gb:  2

$ kubectl get pods -n gpu-operator | grep -v Running | grep -v Completed
(no output)
```

### Lab C — Slurm partition with GPU gres

Requirements: nodes `gn[01-04]` each have 8 CPUs, 64 GB RAM (use 60000 MB usable), and
2 × A100. Create (1) the `slurm.conf` node and partition definitions for a partition `ml`
that is the default, max walltime 24 h; (2) the `gres.conf` for the nodes; (3) an
account `nlp` with user `dana`; (4) show the sbatch header dana uses to get 1 node with
2 GPUs for 8 hours, and the command to verify her job's GPU binding after it starts.

---
---

# ANSWER KEY — score before reading explanations

MCQ answers: 1-B, 2-C, 3-B, 4-C, 5-A, 6-B, 7-B, 8-B, 9-C, 10-B, 11-C, 12-B, 13-C, 14-B,
15-B, 16-B, 17-A, 18-B, 19-B, 20-A, 21-B, 22-B, 23-C, 24-B, 25-C, 26-A, 27-B, 28-C, 29-B, 30-A

## One-line explanations

1. **B** — BCM's model is image-based: patch the image once, provision everywhere; A/C cause image drift, D isn't how provisioning works.
2. **C** — Provisioning uses the management network; the compute fabric being down doesn't block the node-installer (A, B, D all directly break provisioning).
3. **B** — cmsh stages changes; `commit` persists them.
4. **C** — "Munge decode failed" = key mismatch or clock skew; slurmdbd (A) affects accounting, not auth.
5. **A** — cons_tres + GresTypes/Gres is the GPU-scheduling pair; the others are useful but not what enables per-GPU allocation.
6. **B** — kubeadm never installs a CNI; node stays NotReady and CoreDNS Pending until one is applied.
7. **B** — Pre-installed driver → disable the operator's driver DaemonSet; everything else still needed.
8. **B** — The device plugin registers GPU capacity with kubelet; GFD only labels, DCGM only monitors, toolkit only wires the runtime.
9. **C** — Standard separation: PXE/mgmt, NCCL on the fabric, BMC out-of-band; D is too absolute (storage often shares the HS fabric).
10. **B** — drain = finish running work, accept nothing new; down (A) kills scheduling state immediately.
11. **C** — QOS MaxWall attached to the account enforces it account-wide across partitions; B changes limits for everyone.
12. **B** — Over-quota usage is preemptible by definition; in-quota non-preemptible work triggers reclaim.
13. **C** — Fractions are software memory-limits on a shared GPU; MIG (B) is the hardware-isolated one.
14. **B** — Cluster-wide read across namespaces = ClusterRole + ClusterRoleBinding, read-only verbs.
15. **B** — 1g.5gb ×7 is the max instance count on A100-40GB; 8 (D) exceeds the 7-slice hardware limit.
16. **B** — Mixed strategy exposes per-profile resource names (`nvidia.com/mig-2g.10gb`); single strategy would expose plain `nvidia.com/gpu`.
17. **A** — nvcr.io private images need the `$oauthtoken` + API-key pull secret; NGC CLI on nodes (B) is not how kubelet pulls.
18. **B** — The value is the pinned co-tested stack, driver stays on the host (A is wrong: containers never include the kernel driver).
19. **B** — `--gres=gpu:4` is per node × 2 nodes = 8 GPUs.
20. **A** — Repository layout is `<model>/<version>/model.<ext>` with `config.pbtxt` at the model level.
21. **B** — NIM = prebuilt, supported, OpenAI-compatible microservice; A/D are engineering projects, C is nonsense.
22. **B** — All-or-nothing co-scheduling of a pod group is gang scheduling (KAI/Run:ai, Kubeflow integration).
23. **C** — Bursty, idle-heavy notebooks want oversubscription; L4 has no MIG (A impossible), B strands capacity.
24. **B** — Docker has no registered GPU runtime; `nvidia-ctk runtime configure` writes daemon.json, restart applies it.
25. **C** — Userspace/kernel-module skew is resolved by reloading the module or rebooting; nothing else touches the loaded module.
26. **A** — NVSwitch systems need Fabric Manager running (and version == driver) before CUDA can initialize.
27. **B** — Xid 79 is the GPU dropping off PCIe: hardware/power/thermal until proven otherwise; drain and investigate.
28. **C** — Xid 63 self-heals via remap after a reset; RMA (A) is for Xid 64/remap-failure or recurrence.
29. **B** — Ranks advertising docker0 can't be reached; pin `NCCL_SOCKET_IFNAME` (or exclude interfaces) consistently.
30. **A** — Low PCIe bandwidth on one GPU = check negotiated link width/gen first; it localizes to slot/riser/BIOS.

## Lab solutions

### Lab A solution

```bash
# 1. Confirm state and enable MIG mode
nvidia-smi --query-gpu=name,mig.mode.current --format=csv
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi --gpu-reset -i 0          # apply pending enable (GPU is idle)
nvidia-smi --query-gpu=mig.mode.current --format=csv    # -> Enabled

# 2. Create 2x 3g.20gb with compute instances
sudo nvidia-smi mig -cgi 3g.20gb,3g.20gb -C

# 3. Prove it
nvidia-smi mig -lgi                        # two GPU instances, profile 3g.20gb
nvidia-smi -L                              # shows MIG 3g.20gb Device 0 and Device 1 with UUIDs

# 4. Run the container on ONE instance (either form accepted)
sudo docker run --rm --gpus '"device=0:0"' \
  nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
# or, using the MIG UUID from nvidia-smi -L:
sudo docker run --rm --gpus '"device=MIG-<uuid>"' \
  nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
```

Grading points: MIG enable **with reset**, correct `-cgi ... -C` (forgetting `-C` = no compute
instance = CUDA can't run), a verification command, and container scoped to one instance
(not `--gpus all`).

### Lab B solution

**Root cause:** resource-name mismatch. The nodes are MIG-partitioned under the **mixed**
strategy and advertise `nvidia.com/mig-3g.20gb: 2`; they advertise **zero** `nvidia.com/gpu`.
The pod requests `nvidia.com/gpu: 1`, so every node reports `Insufficient nvidia.com/gpu`.
The operator stack is healthy (no non-Running pods), so this is configuration, not failure.

**Fix (either direction is acceptable — state one):**

1. Fix the workload to request the MIG resource:

```yaml
resources:
  limits:
    nvidia.com/mig-3g.20gb: 1
```

`kubectl apply` the corrected manifest (or `kubectl delete pod trainer-0` and resubmit via its
controller).

2. Or change the cluster to match the workloads — switch to the `single` MIG strategy so
uniform MIG slices appear as plain `nvidia.com/gpu`:

```bash
kubectl patch clusterpolicies.nvidia.com/cluster-policy --type merge \
  -p '{"spec":{"mig":{"strategy":"single"}}}'
# wait for device plugin restart, then verify:
kubectl describe node gpu-node-1 | grep nvidia.com/gpu   # -> nvidia.com/gpu: 2
```

Grading points: reading Allocatable to spot `mig-3g.20gb` vs requested `gpu`; explicitly ruling
out device-plugin failure; a concrete fix with verification.

### Lab C solution

**(1) slurm.conf additions:**

```
GresTypes=gpu
SelectType=select/cons_tres
NodeName=gn[01-04] CPUs=8 RealMemory=60000 Gres=gpu:a100:2 State=UNKNOWN
PartitionName=ml Nodes=gn[01-04] Default=YES MaxTime=24:00:00 State=UP
```

**(2) gres.conf (on each gn node, or with NodeName= prefixes centrally):**

```
NodeName=gn[01-04] Name=gpu Type=a100 File=/dev/nvidia0
NodeName=gn[01-04] Name=gpu Type=a100 File=/dev/nvidia1
```

(`AutoDetect=nvml` alone is an acceptable alternative if slurmd is NVML-built.)

Reload: `scontrol reconfigure` (and restart slurmd if node definitions changed).

**(3) Accounting:**

```bash
sacctmgr -i add account nlp Description="NLP team"
sacctmgr -i add user dana account=nlp
```

**(4) dana's job + verification:**

```bash
#!/bin/bash
#SBATCH --partition=ml
#SBATCH --account=nlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=08:00:00
srun python train.py
```

Verify binding once running:

```bash
squeue -u dana                                   # job id + node
sacct -j <jobid> --format=JobID,AllocTRES%40     # shows gres/gpu=2
srun --jobid=<jobid> --overlap bash -c 'echo $CUDA_VISIBLE_DEVICES; nvidia-smi -L'
```

Grading points: `Gres=` on node lines **and** matching gres.conf, `GresTypes`/`cons_tres`
present, Default+MaxTime on the partition, account→user association order, `--gres=gpu:a100:2`
(or `--gpus-per-node=2`) with time and partition, and a real verification command.

---

**Scoring guide:** ≥24/30 MCQ and all three labs substantially correct = ready.
20–23 = one more drill day on your weakest domain. <20 = redo that week's self-check and labs
before burning the exam fee.
