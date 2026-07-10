# Week 4 Self-Check — AI Operations

Target: ≥ 80% (13/16). Do this before the mock exam (Day 3) if possible.

**1. How does a Kubernetes pod get access to a GPU?**

<details><summary>Answer</summary>
The NVIDIA device plugin (a DaemonSet) advertises GPUs to the kubelet as the extended resource `nvidia.com/gpu`; a pod requests it via `resources.limits: nvidia.com/gpu: 1` and the scheduler places it on a node with a free GPU. The container toolkit injects the driver libraries/devices into the container.
</details>

**2. What problem does the NVIDIA GPU Operator solve, and name four components it manages.**

<details><summary>Answer</summary>
It automates the full GPU software stack lifecycle on Kubernetes nodes instead of hand-installing per node. It deploys and manages: the GPU driver (containerized), NVIDIA container toolkit, device plugin, DCGM exporter (monitoring), GPU feature discovery/node labeling, and the MIG manager.
</details>

**3. A national lab runs large multi-node batch training jobs with job queues and fair-share policies. A SaaS company serves inference microservices with autoscaling. Which orchestrator fits each, and why?**

<details><summary>Answer</summary>
The lab: Slurm — an HPC batch scheduler built for queued, gang-scheduled multi-node jobs, partitions, and fair-share accounting. The SaaS company: Kubernetes — built for long-running containerized services, autoscaling, rolling updates, and service networking.
</details>

**4. In Slurm, what are a partition and gres?**

<details><summary>Answer</summary>
A partition is a named group of nodes with its own limits/policies (like a queue) that jobs are submitted to. GRES = "generic resources" — Slurm's mechanism for scheduling non-CPU resources, notably GPUs (`--gres=gpu:4`).
</details>

**5. When would you use nvidia-smi vs DCGM?**

<details><summary>Answer</summary>
nvidia-smi: interactive spot checks on a single node — quick view of utilization, memory, temperature, power, processes. DCGM: fleet-scale operations — continuous background health monitoring, active diagnostics, policies/alerts, job stats, and Prometheus integration via the DCGM exporter. Rule of thumb: human debugging → nvidia-smi; production monitoring → DCGM.
</details>

**6. What is the standard GPU monitoring pipeline on Kubernetes?**

<details><summary>Answer</summary>
DCGM exporter (deployed by the GPU Operator) exposes per-GPU metrics in Prometheus format → Prometheus scrapes them → Grafana dashboards + Alertmanager rules. Metrics carry pod/namespace labels so usage maps to workloads.
</details>

**7. What's the difference between a single-bit and a double-bit ECC error, and the operational response to each?**

<details><summary>Answer</summary>
Single-bit errors are detected AND corrected by ECC — the job continues; track the rate, as growth indicates degrading memory. Double-bit errors are detected but uncorrectable — the affected application typically crashes; pages may be retired, and recurring ones mean draining the node / RMA-ing the GPU.
</details>

**8. A GPU shows 100% utilization but the training job is slow. Name two metrics you'd check next and what they'd tell you.**

<details><summary>Answer</summary>
Clock throttling reasons + temperature/power: thermal or power-capping throttle means the GPU runs at reduced clocks despite "100% busy". Also check memory bandwidth/SM efficiency-type counters (via DCGM) — "utilization" in nvidia-smi only means a kernel was resident, not that the GPU was doing useful dense work. NVLink/network errors are another candidate for multi-GPU jobs.
</details>

**9. What is an XID error?**

<details><summary>Answer</summary>
An error event code reported by the NVIDIA driver into the system log when a GPU problem occurs (e.g. Xid 48 double-bit ECC, Xid 79 GPU fallen off the bus). XIDs are the first artifact to check when a GPU job dies unexpectedly.
</details>

**10. Explain MIG in two sentences.**

<details><summary>Answer</summary>
Multi-Instance GPU (Ampere and later) partitions one physical GPU at the hardware level into up to 7 instances, each with dedicated SM slices, memory, and cache. Each instance is fully isolated (memory and faults) with guaranteed QoS, and appears as its own GPU to workloads and Kubernetes.
</details>

**11. MIG vs time-slicing: a customer wants to pack many small inference services from different teams onto A100s and guarantee one team can't disturb another. Which, and why?**

<details><summary>Answer</summary>
MIG — hardware partitioning gives dedicated memory and compute slices per instance, so noisy neighbors can't cause OOM or latency interference. Time-slicing has no memory isolation or QoS: all processes share the full GPU memory and take turns, so one tenant can starve or crash others. Time-slicing suits friendly/bursty dev workloads, not multi-tenant guarantees.
</details>

**12. When is NVIDIA vGPU the right answer instead of MIG or time-slicing?**

<details><summary>Answer</summary>
When workloads run inside virtual machines — VDI, VMware/enterprise-virtualization environments, or clouds selling VM-based GPU shares. vGPU is the licensed driver/manager stack that shares or partitions a GPU across VMs (it can even use MIG for the backing partitions). MIG/time-slicing alone address containers/bare metal, not VM boundaries.
</details>

**13. How do MIG instances show up in Kubernetes?**

<details><summary>Answer</summary>
The device plugin (with the MIG manager configuring the geometry) advertises them as schedulable resources — either as generic `nvidia.com/gpu` (single strategy) or as sized resources like `nvidia.com/mig-1g.10gb` (mixed strategy) that pods request explicitly.
</details>

**14. What does Base Command Manager do?**

<details><summary>Answer</summary>
NVIDIA's cluster management software: provisioning/imaging of nodes, driver and software lifecycle, cluster health monitoring, and deployment of workload managers (Slurm or Kubernetes) — the ops layer for DGX BasePOD/SuperPOD deployments.
</details>

**15. Why do operators run DCGM diagnostics in a job prologue on training clusters?**

<details><summary>Answer</summary>
To catch unhealthy GPUs before a job starts: a short active health check (dcgmi diag) in the scheduler prologue prevents a multi-node job from being scheduled onto a node with failing memory, NVLink, or thermal issues — far cheaper than a 512-GPU job crashing hours in.
</details>

**16. A pod requests `nvidia.com/gpu: 1` but stays Pending on a GPU node. Name two likely causes an ops engineer would check.**

<details><summary>Answer</summary>
(1) The GPU software stack isn't healthy on that node — device plugin not running / GPU Operator components failed — so the node advertises 0 GPUs; (2) all GPUs already allocated (GPUs are not oversubscribed by default — each is exclusively assigned). Also plausible: taints/node selectors, or requesting a MIG resource name that doesn't match the configured geometry.
</details>
