# Week 11 Self-Check — Workload Management + Troubleshooting (part 1)

Closed book. Target ≥ 15/18.

**1. What username do you use for `docker login nvcr.io`, and what's the password?**

<details><summary>Answer</summary>
Username is the literal string <code>$oauthtoken</code> (quote it in shells), password is your NGC API key. Same credentials for K8s: <code>kubectl create secret docker-registry ngc-secret --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=$NGC_API_KEY</code>.
</details>

**2. What guarantees does an NGC deep-learning framework container give you that a pip install doesn't?**

<details><summary>Answer</summary>
A tested, pinned-compatible stack: CUDA toolkit, cuDNN, NCCL, and the framework built against each other, tuned and QA'd monthly (tags like 25.06-py3). Only external dependency is a new-enough host driver. Removes the CUDA/framework version-matrix problem.
</details>

**3. A container built with CUDA 12.8 runs on a host with driver 550 (CUDA 12.4 era). Works or not, and why?**

<details><summary>Answer</summary>
Generally works for minor versions: CUDA minor-version compatibility lets 12.x runtimes run on any 12.x-capable driver (and forward-compat libs can bridge more). Rule of thumb the exam wants: host driver must be new enough for the CUDA *major* version; too-old driver → "CUDA driver version is insufficient for CUDA runtime version".
</details>

**4. Write the SBATCH header for 2 nodes × 4 GPUs × 4 tasks per node, 2-hour limit, partition gpu.**

<details><summary>Answer</summary>
<pre>
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
srun ./train.sh
</pre>
(<code>--gpus-per-node=4</code> is the equivalent newer syntax.)
</details>

**5. `--gres=gpu:2` vs `--gpus=2` — difference?**

<details><summary>Answer</summary>
<code>--gres=gpu:2</code> is per-node: 2 GPUs on each allocated node. <code>--gpus=2</code> is a job-total GPU count that Slurm may spread across nodes. With one node they coincide; multi-node they don't.
</details>

**6. How does a Slurm job's process end up seeing only its allocated GPUs?**

<details><summary>Answer</summary>
Slurm (via gres/gpu + cgroup/task plugins) sets <code>CUDA_VISIBLE_DEVICES</code> for each step to the allocated device indices and, with cgroup enforcement (ConstrainDevices=yes), blocks the device files of other GPUs — so even a misbehaving app can't grab unallocated GPUs.
</details>

**7. What are enroot and pyxis?**

<details><summary>Answer</summary>
NVIDIA's container path for Slurm: enroot is a rootless container runtime that turns Docker/NGC images into unprivileged sandboxes; pyxis is the Slurm SPANK plugin exposing it as <code>srun --container-image=nvcr.io#nvidia/pytorch:25.06-py3</code>. The standard way to run NGC containers under Slurm (BCM ships them).
</details>

**8. Sketch the Triton model repository layout for an ONNX model "resnet" v1, and name the three ports.**

<details><summary>Answer</summary>
<pre>
model_repo/
  resnet/
    config.pbtxt
    1/
      model.onnx
</pre>
Ports: 8000 HTTP/REST, 8001 gRPC, 8002 Prometheus metrics. Readiness: <code>GET /v2/health/ready</code>.
</details>

**9. Minimal docker command to run a NIM, and what API does it expose?**

<details><summary>Answer</summary>
<pre>
docker login nvcr.io   # $oauthtoken / NGC key
docker run --rm --gpus all -e NGC_API_KEY \
  -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
</pre>
Exposes an OpenAI-compatible API (<code>/v1/chat/completions</code>, <code>/v1/models</code>) on 8000. NIM picks an optimized engine profile (TensorRT-LLM/vLLM) for the detected GPU.
</details>

**10. An admin must run 7 small inference services on one A100 with strong isolation; another team wants max throughput for many identical CUDA processes from one user. Pick sharing strategies.**

<details><summary>Answer</summary>
Isolation case: MIG — 7× 1g.5gb, hardware-isolated memory/SM/faults. Throughput-one-user case: MPS — concurrent kernels from multiple processes share the GPU without context-switch serialization (no fault isolation, fine within one trust domain). Time-slicing would serialize; fractions are Run:ai-managed sharing.
</details>

**11. Where do DRA ResourceClaims fit vs the device plugin (cross-ref your demo repo)?**

<details><summary>Answer</summary>
Device plugin: opaque countable resource (<code>nvidia.com/gpu: 2</code>), no attribute selection. DRA (structured parameters, GA in K8s 1.34): DeviceClasses + ResourceClaims with CEL selectors ("A100 with ≥40GB", specific MIG profile), claims schedulable/shareable like PVCs. NVIDIA's DRA driver is the successor path to the device plugin for complex GPU/topology requests.
</details>

**12. Pod error: "could not select device driver \"\" with capabilities: [[gpu]]" (or container can't see GPU while host nvidia-smi works). Give your 4-step diagnostic sequence.**

<details><summary>Answer</summary>
(1) Runtime wiring: is <code>runtimeClassName: nvidia</code> set / is nvidia the default runtime in containerd config? (2) Toolkit: is nvidia-container-toolkit installed and configured (<code>nvidia-ctk runtime configure</code>, restart containerd)? (3) Device plugin: is the DaemonSet running and does <code>kubectl describe node</code> show <code>nvidia.com/gpu</code> capacity? (4) GPU Operator validators: <code>kubectl get pods -n gpu-operator</code> — a failing validator pinpoints the broken layer.
</details>

**13. `nvidia-smi` prints "Failed to initialize NVML: Driver/library version mismatch". Cause and fix?**

<details><summary>Answer</summary>
Userspace driver libraries were upgraded while the old kernel module is still loaded (driver update without reload). Fix: reboot, or unload/reload modules (<code>rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia; modprobe nvidia</code>) — reboot is the reliable answer. Verify versions: <code>cat /proc/driver/nvidia/version</code> vs installed libs.
</details>

**14. On an 8-GPU HGX box, `nvidia-smi` looks healthy but every CUDA app fails at init with "system not yet initialized". First thing you check?**

<details><summary>Answer</summary>
<code>systemctl status nvidia-fabricmanager</code>. On NVSwitch systems Fabric Manager must be running to program NVLink/NVSwitch routing before CUDA can initialize. Second check: FM package version exactly matches the driver version (mismatched FM refuses to start — see /var/log/fabricmanager.log).
</details>

**15. When is Fabric Manager NOT required?**

<details><summary>Answer</summary>
On systems without NVSwitch: single-GPU nodes, PCIe-only multi-GPU servers, or NVLink-direct (non-switch) topologies. It's specifically the NVSwitch fabric supervisor for HGX/DGX baseboards.
</details>

**16. A BCM compute node hangs forever in the node-installer stage. Name three things to check.**

<details><summary>Answer</summary>
(1) Provisioning network path: DHCP/PXE/TFTP from the head node reaching the node (right network, port security, cabling). (2) Node identification: MAC/category assignment correct in cmsh, image assigned. (3) Head-node side: CMDaemon logs (/var/log/cmdaemon, node-installer console via BMC) for disk layout or image sync errors.
</details>

**17. What does `imageupdate --dry-run` (cmsh) tell you and why run it first?**

<details><summary>Answer</summary>
It previews the rsync between the software image and the live node — which files would be added/changed/deleted — without applying. Run it first because a sync can clobber files changed on the node outside the image (or reveal you forgot exclude lists for stateful paths).
</details>

**18. Your KAI/TrainJob demo submits a 4-pod PyTorch job but only 3 GPUs are free; what happens and which exam concept is this?**

<details><summary>Answer</summary>
Nothing schedules: gang scheduling (PodGroup minMember=4) is all-or-nothing, so all pods stay Pending until 4 GPUs are available — avoiding deadlocked partial allocations. Exam concept: gang scheduling in Run:ai/KAI (and Kubeflow TrainJob integration) under Workload Management/Administration.
</details>
