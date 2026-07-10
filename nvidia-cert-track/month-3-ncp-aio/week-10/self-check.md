# Week 10 Self-Check — Administration

Closed book. Target ≥ 15/18.

**1. Give the exact command to drain node gpu01 for maintenance with a reason, and to return it to service.**

<details><summary>Answer</summary>
<code>scontrol update nodename=gpu01 state=drain reason="maintenance"</code> — running jobs finish, no new jobs land. Return: <code>scontrol update nodename=gpu01 state=resume</code>.
</details>

**2. In `sinfo`, what do node states `mix`, `alloc`, and `drng` mean?**

<details><summary>Answer</summary>
<code>alloc</code>: all allocatable resources in use. <code>mix</code>: partially allocated (some CPUs/GPUs busy, some free). <code>drng</code>: draining — drain requested while jobs still running; becomes <code>drain</code>/<code>drained</code> when they finish.
</details>

**3. Create a QOS "short" with 30-min max walltime and max 2 GPUs per user, and let account "dev" use it. Commands?**

<details><summary>Answer</summary>
<pre>
sacctmgr add qos short MaxWall=00:30:00 MaxTRESPerUser=gres/gpu=2
sacctmgr modify account dev set qos+=short
</pre>
Users submit with <code>sbatch --qos=short</code>. Enforcement requires slurmdbd + AccountingStorageEnforce=limits,qos.
</details>

**4. What is a Slurm association, and what happens to job submission if a user has none?**

<details><summary>Answer</summary>
An association is the (cluster, account, user, [partition]) tuple in the slurmdbd database to which limits, fairshare, and QOS attach. With <code>AccountingStorageEnforce=associations</code>, a user without one gets submissions rejected (Invalid account).
</details>

**5. How does Slurm fairshare priority work in one paragraph?**

<details><summary>Answer</summary>
With <code>PriorityType=priority/multifactor</code>, job priority is a weighted sum of factors: fairshare (how much the account/user's recent usage is under/over its allocated share — usage decays with a half-life), age, job size, partition, QOS, and nice. Heavy recent users sink, underserved accounts float up. Weights set by PriorityWeight* parameters.
</details>

**6. Map these Run:ai concepts to KAI Scheduler primitives: project quota, over-quota, gang scheduling.**

<details><summary>Answer</summary>
Project quota → KAI queue with a resource quota (deserved GPUs). Over-quota → queue's over-quota weight lets workloads exceed quota using idle capacity, subject to reclaim/preemption when the owning queue wants its quota back. Gang scheduling → PodGroup with minMember: all pods placed atomically or none (KAI is the open-sourced core of the Run:ai scheduler).
</details>

**7. In Run:ai, what happens to an over-quota preemptible training job when the quota-owning project submits work?**

<details><summary>Answer</summary>
It gets preempted (reclaimed): the scheduler evicts the over-quota workload to give the in-quota project its deserved resources. Training workloads are preemptible by default; non-preemptible workloads can't exceed quota in the first place.
</details>

**8. How do Run:ai fractional GPUs differ from MIG?**

<details><summary>Answer</summary>
Fractions are software sharing: multiple containers time-share one full GPU with Run:ai enforcing per-workload GPU memory limits — no hardware isolation, faults/noisy neighbors can interfere. MIG is hardware partitioning with dedicated SM/memory/L2 slices and fault isolation, only on supported GPUs (A100/H100/A30/B200...), fixed profile shapes.
</details>

**9. Role vs ClusterRole — and which do you need to let a user list nodes?**

<details><summary>Answer</summary>
Role is namespaced; ClusterRole is cluster-wide and also usable for cluster-scoped resources. Nodes are cluster-scoped, so listing nodes needs a ClusterRole + ClusterRoleBinding.
</details>

**10. Write a ResourceQuota limiting namespace "team-a" to 4 GPUs.**

<details><summary>Answer</summary>
<pre>
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-a
spec:
  hard:
    requests.nvidia.com/gpu: "4"
</pre>
Pods requesting GPUs beyond the total are rejected at admission.
</details>

**11. What does `kubectl drain` do that `kubectl cordon` doesn't, and what commonly blocks it?**

<details><summary>Answer</summary>
Cordon only marks the node unschedulable. Drain also evicts existing pods (respecting PodDisruptionBudgets). Commonly blocked by: PDBs that would be violated, DaemonSet pods (need <code>--ignore-daemonsets</code>), and pods with local storage (<code>--delete-emptydir-data</code>).
</details>

**12. List the MIG profiles on an A100-40GB and the max count of each.**

<details><summary>Answer</summary>
1g.5gb ×7, 2g.10gb ×3, 3g.20gb ×2, 4g.20gb ×1, 7g.40gb ×1 (combinations allowed if slices fit, e.g. 4g.20gb + 3g.20gb, or 3g.20gb + 3g.20gb + 1g.5gb). On A100-80GB the memory doubles: 1g.10gb … 7g.80gb.
</details>

**13. Exact commands: enable MIG on GPU 0 and create two 3g.20gb instances with compute instances.**

<details><summary>Answer</summary>
<pre>
sudo nvidia-smi -i 0 -mig 1        # may require GPU reset / drain first
sudo nvidia-smi mig -cgi 3g.20gb,3g.20gb -C
nvidia-smi mig -lgi                # verify
</pre>
<code>-C</code> auto-creates the default compute instance in each GPU instance. (Profile IDs work too: <code>-cgi 9,9</code> on A100.)
</details>

**14. GI vs CI in MIG?**

<details><summary>Answer</summary>
GPU Instance (GI): the hardware partition — memory slices + SM slices, fault-isolated. Compute Instance (CI): a subdivision of a GI's SMs sharing that GI's memory. Usually 1 CI = whole GI (<code>-C</code>), but a GI can be split into multiple CIs.
</details>

**15. In GPU Operator, how do you declaratively set all GPUs on node gpu01 to 3g.20gb, and what's the difference between mig.strategy single and mixed?**

<details><summary>Answer</summary>
Label the node: <code>kubectl label node gpu01 nvidia.com/mig.config=all-3g.20gb --overwrite</code> — MIG Manager (using nvidia-mig-parted) applies the profile from its ConfigMap. <code>single</code>: all GPUs same profile, advertised as plain <code>nvidia.com/gpu</code>. <code>mixed</code>: heterogeneous profiles, advertised as <code>nvidia.com/mig-3g.20gb</code> etc.
</details>

**16. A colleague wants MIG on an L4 fleet. Response?**

<details><summary>Answer</summary>
L4 doesn't support MIG (no MIG hardware on Ada L4/L40S). Offer time-slicing (device-plugin replicas — no isolation) or MPS (concurrent processes, memory limits, no fault isolation), or MIG-capable GPUs (A30/A100/H100/B200).
</details>

**17. What is a rail-optimized network topology and why does it help NCCL?**

<details><summary>Answer</summary>
Each of the 8 NICs (rails) on every GPU node connects to its own leaf switch plane, so GPU i on node A reaches GPU i on node B in one hop on rail i. NCCL's ring/tree algorithms send GPU-to-same-index-GPU traffic, so rails avoid cross-rail hops and congestion — full bisection bandwidth per rail.
</details>

**18. In one sentence each: where do BCM, BMC/IPMI, Slurm control traffic, and NCCL traffic run in a well-built cluster?**

<details><summary>Answer</summary>
BCM/CMDaemon on the internal management network; BMC/IPMI on the isolated out-of-band network; slurmctld↔slurmd RPC on the management network; NCCL bulk traffic on the high-speed compute fabric (IB/RoCE), pinned with NCCL_SOCKET_IFNAME/IB HCA selection.
</details>
