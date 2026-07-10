# Week 9 Self-Check — Installation & Deployment

Closed book. Target ≥ 15/18.

**1. In BCM, what is the difference between a node category and a software image?**

<details><summary>Answer</summary>
A software image is the filesystem (OS + packages) that gets provisioned onto nodes. A category is a group of nodes sharing settings — including *which* software image they get, plus disk layout, network config, and other properties. Many categories can point at different (or the same) images; you patch the image, and all nodes in categories using it pick it up on imageupdate/reboot.
</details>

**2. Describe the BCM node provisioning sequence from power-on to a running compute node.**

<details><summary>Answer</summary>
Node PXE-boots from the provisioning (head) node → loads the node-installer environment → node is identified (MAC/position) and matched to its category → node-installer partitions disks per category layout and syncs the assigned software image → switches root into the installed OS and boots it → node reports up to CMDaemon.
</details>

**3. Give the cmsh one-liner to list all devices, and the interactive sequence to power on node003.**

<details><summary>Answer</summary>
One-liner: <code>cmsh -c "device; list"</code>. Interactive: <code>cmsh</code> → <code>device</code> → <code>power -n node003 on</code> (or <code>use node003</code> then <code>power on</code>). Power control goes through the BMC/IPMI interface BCM manages.
</details>

**4. You edited a setting in cmsh but it isn't taking effect. What did you probably forget?**

<details><summary>Answer</summary>
<code>commit</code>. cmsh changes are staged and only applied when committed (Base View has the equivalent Save). Uncommitted changes are shown by <code>refresh</code>/prompt markers.
</details>

**5. What is the correct way to patch OS packages across 200 BCM compute nodes?**

<details><summary>Answer</summary>
Patch the software image, not the nodes: chroot into the image on the head node (<code>cm-chroot-sw-img /cm/images/&lt;image&gt;</code>), run apt/yum update inside, exit, then push with <code>imageupdate</code> (live sync) or reboot nodes per category. Hand-patching live nodes drifts from the image and is undone at next provisioning.
</details>

**6. Name the three core Slurm daemons and where each runs.**

<details><summary>Answer</summary>
<code>slurmctld</code> — controller, on the head/controller node (optionally HA pair). <code>slurmd</code> — on every compute node, launches jobs. <code>slurmdbd</code> — accounting daemon, fronts the MySQL/MariaDB accounting database, usually on the head or a DB node. (Plus munge everywhere for auth.)
</details>

**7. Slurm jobs fail with credential/auth errors on some nodes. Two classic root causes?**

<details><summary>Answer</summary>
Munge key mismatch (the same <code>/etc/munge/munge.key</code> must be on every node) and clock skew (munge credentials are time-stamped — nodes need NTP-synced clocks).
</details>

**8. Write a minimal slurm.conf node + partition definition for node01 with one L4 GPU, 8 CPUs, 30 GB RAM.**

<details><summary>Answer</summary>
<pre>
GresTypes=gpu
SelectType=select/cons_tres
NodeName=node01 CPUs=8 RealMemory=30000 Gres=gpu:l4:1 State=UNKNOWN
PartitionName=gpu Nodes=node01 Default=YES MaxTime=INFINITE State=UP
</pre>
And gres.conf on node01: <code>AutoDetect=nvml</code> (or <code>Name=gpu Type=l4 File=/dev/nvidia0</code>).
</details>

**9. What does `AutoDetect=nvml` in gres.conf do, and what's the manual alternative?**

<details><summary>Answer</summary>
It makes slurmd query the NVIDIA driver via NVML to discover GPUs (count, type, device files, NUMA/core affinity) automatically. Manual alternative: explicit lines like <code>Name=gpu Type=a100 File=/dev/nvidia0 Cores=0-15</code> per GPU.
</details>

**10. In what order do you upgrade Slurm components, and why?**

<details><summary>Answer</summary>
slurmdbd first, then slurmctld, then slurmd (then client commands). Newer daemons speak older RPC protocols (typically two versions back) but not vice versa, so the DB/controller must be newest to keep accepting traffic from not-yet-upgraded slurmds.
</details>

**11. List the GPU Operator components in dependency order and one sentence on each.**

<details><summary>Answer</summary>
(1) NFD/GFD — label nodes with GPU presence/attributes; (2) driver DaemonSet — kernel driver as a container (skip with driver.enabled=false if pre-installed); (3) container toolkit — installs nvidia-container-runtime and configures containerd; (4) device plugin — advertises nvidia.com/gpu to kubelet; (5) DCGM exporter — GPU metrics for Prometheus; (6) MIG Manager — applies MIG partition configs; plus validator pods that gate readiness.
</details>

**12. When must you set `driver.enabled=false` when installing GPU Operator?**

<details><summary>Answer</summary>
When the node already has the NVIDIA driver installed on the host — e.g. DGX OS, cloud GPU images with pre-baked drivers, or admin-managed drivers. The operator then only deploys toolkit/plugin/DCGM etc. Running the driver container on top of an existing host driver fails.
</details>

**13. What is CDI and how does it relate to the container toolkit?**

<details><summary>Answer</summary>
Container Device Interface — a vendor-neutral spec (JSON files describing device nodes, mounts, hooks) for exposing devices to containers. nvidia-ctk generates CDI specs; runtimes (containerd/CRI-O/podman) inject GPUs from them, replacing the legacy runtime-hook injection. GPU Operator can enable it with cdi.enabled=true.
</details>

**14. kubeadm init succeeded but pods stay Pending with no network. What's missing?**

<details><summary>Answer</summary>
A CNI plugin. kubeadm does not install one; nodes stay NotReady/coredns Pending until you apply Calico/Flannel/Cilium etc., matching the --pod-network-cidr you gave kubeadm init.
</details>

**15. Which BCM wizards deploy Slurm and Kubernetes respectively?**

<details><summary>Answer</summary>
<code>cm-wlm-setup</code> for workload managers (Slurm etc.) and <code>cm-kubernetes-setup</code> for Kubernetes. Both configure the cluster from the head node using BCM's knowledge of nodes/categories.
</details>

**16. Name the four logically separate networks in a typical GPU cluster and what runs on each.**

<details><summary>Answer</summary>
Management/provisioning network (PXE, BCM CMDaemon, SSH); out-of-band/BMC network (IPMI/Redfish power and console); high-speed compute fabric (InfiniBand or RoCE — NCCL inter-node traffic); storage network (parallel FS traffic, sometimes shared with compute fabric). Exam angle: know which failure shows up where.
</details>

**17. Which DCGM metric names would you alert on for (a) GPU utilization, (b) framebuffer memory used, (c) Xid errors?**

<details><summary>Answer</summary>
(a) <code>DCGM_FI_DEV_GPU_UTIL</code>, (b) <code>DCGM_FI_DEV_FB_USED</code>, (c) <code>DCGM_FI_DEV_XID_ERRORS</code> — all exposed by dcgm-exporter on :9400/metrics.
</details>

**18. How do you add user alice to BCM, and separately give her a Slurm account association?**

<details><summary>Answer</summary>
BCM: <code>cmsh</code> → <code>user</code> → <code>add alice</code> → set properties → <code>commit</code> (stored in BCM's LDAP). Slurm accounting: <code>sacctmgr add account research</code> then <code>sacctmgr add user alice account=research</code> — without an association (when AccountingStorageEnforce includes associations) her jobs are rejected.
</details>
