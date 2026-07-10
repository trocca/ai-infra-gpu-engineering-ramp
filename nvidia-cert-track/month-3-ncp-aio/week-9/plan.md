# Week 9 Plan — Installation & Deployment (Domain 1, 31%)

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

**Dates:** Mon 2026-09-07 → Fri 2026-09-11 · ~2h/day
**Domain:** Installation & Deployment — 31%, the single biggest domain. Everything this week
maps to it unless noted. The exam's labs love this domain because it's inherently hands-on.

BCM has **no free tier** — you study it from the Administrator Manual and drill the *equivalent*
operations (provisioning concepts, Slurm setup, K8s setup) on cheap cloud VMs. Know `cmsh`
syntax cold from the docs even though you can't run it.

## Prerequisites before Monday

- Companion lesson: [Week 09 companion — distributed training, all-reduce cost, and NCCL support](../../../companion-lessons/week-09.md).
- Math support: all-reduce byte cost, rank/world-size vocabulary, and scaling efficiency.
- Systems/programming support: `torch.distributed`, DDP hooks, NCCL logs, cloud GPU setup, and teardown discipline.
- Gate: explain all-reduce in three sentences and confirm the cloud GPU budget before launching anything.

---

## Day 1 (Mon) — BCM architecture & node provisioning

- 0:00–0:45 — BCM Administrator Manual (https://docs.nvidia.com/base-command-manager/), read:
  intro + architecture chapters. Head node roles (management, provisioning, monitoring, HA via
  `cmha`), regular/compute nodes, node categories, software images.
- 0:45–1:30 — Node provisioning flow: PXE boot → node-installer → image sync from provisioning
  node. Software image lifecycle: clone image → modify (`cm-chroot-sw-img` to chroot in,
  apt/yum inside) → assign to category → `imageupdate` / reboot to push. Grab-and-sync vs full
  provisioning.
- 1:30–2:00 — `cmsh` basics from the manual: modes (`device`, `category`, `softwareimage`,
  `network`, `user`, `partition`, `monitoring`), `list`, `show`, `use`, `set` + `commit`.
  One-liners: `cmsh -c "device; list"`. Base View = the GUI equivalent. Write 10 cmsh commands
  from memory into `notes.md`.

## Day 2 (Tue) — Slurm cluster setup (as an installer, not a user)

- 0:00–0:30 — Architecture: `slurmctld` (controller), `slurmd` (per compute node), `slurmdbd`
  (accounting → MySQL/MariaDB), `munge` (auth — clocks and identical keys matter). Read Slurm
  Quick Start Admin Guide (https://slurm.schedmd.com/quickstart_admin.html).
- 0:30–1:15 — `slurm.conf` essentials: `SlurmctldHost`, `NodeName=` lines (CPUs, RealMemory,
  `Gres=gpu:l4:1`), `PartitionName=` lines, `GresTypes=gpu`, `SelectType=select/cons_tres`,
  `AccountingStorageType=accounting_storage/slurmdbd`. `gres.conf`: `AutoDetect=nvml` vs
  explicit `Name=gpu Type=l4 File=/dev/nvidia0`. Read https://slurm.schedmd.com/gres.html.
- 1:15–2:00 — **Do:** start `labs/lab-slurm-basics.md` Part A (bring up the containerized
  cluster, verify `sinfo`). BCM tie-in: BCM deploys Slurm via `cm-wlm-setup` wizard — know that
  name.

## Day 3 (Wed) — Kubernetes cluster setup + GPU Operator

- 0:00–0:30 — kubeadm flow (know the sequence, you won't hand-build in the exam): container
  runtime (containerd) → `kubeadm init --pod-network-cidr=...` → install CNI → `kubeadm join`.
  BCM equivalent: `cm-kubernetes-setup` wizard. Taints on control plane, kubeconfig location.
- 0:30–1:00 — GPU Operator stack, in dependency order: **driver → container toolkit → device
  plugin → DCGM exporter → GFD/NFD → (MIG Manager)**. What each DaemonSet does; when to set
  `driver.enabled=false` (pre-installed driver, e.g. DGX OS / cloud images). CDI. Docs:
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html
- 1:00–2:00 — **Do:** `labs/lab-gpu-operator.md` end-to-end (rent the L4 VM today; keep it for
  weeks 9–12 labs, stop it when idle).

## Day 4 (Thu) — Networking + monitoring

- 0:00–0:45 — Cluster networking: BCM network objects (internalnet, externalnet, ipmi/BMC net),
  node interfaces set per category/device in cmsh. Fabrics: Ethernet vs InfiniBand, RoCE, rails.
  Compute vs storage vs management vs out-of-band networks in a DGX-style pod. Where NCCL fits
  (your demo repo's NCCL transports section = the app view of this same fabric).
- 0:45–1:30 — Monitoring: BCM built-in monitoring (metrics, healthchecks, actions in cmsh
  `monitoring` mode); DCGM + dcgm-exporter → Prometheus → Grafana on K8s. Key metric names:
  `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_XID_ERRORS`.
- 1:30–2:00 — **Do:** on the lab cluster, `curl` the dcgm-exporter `/metrics` endpoint and find
  those three metrics (lab-gpu-operator step 7).

## Day 5 (Fri) — Patching, upgrades, user management + review

- 0:00–0:45 — Patch management: BCM pattern = patch the **software image** (chroot), then
  `imageupdate`/reboot nodes by category; never hand-patch live nodes. Driver upgrades on K8s =
  GPU Operator upgrade / driver version bump in ClusterPolicy (rolling, node-by-node). Slurm
  upgrade order: **slurmdbd → slurmctld → slurmd** (DB first, protocol compat window).
- 0:45–1:30 — User management: BCM users/groups via cmsh `user` mode (backed by LDAP);
  external identity (AD/LDAP) integration concept. Slurm associations: `sacctmgr add account`,
  `sacctmgr add user ... account=...`. K8s: users are certs/OIDC + RBAC (deep-dive next week).
- 1:30–2:00 — `self-check.md` closed-book, log misses in notes.

---

## Exit criteria (Friday) — "I can DO"

- [ ] I can describe the BCM provisioning chain (PXE → node-installer → image sync) and write
      5 correct `cmsh` one-liners (list devices, power on a node, assign a category, list
      software images, commit a change) without looking.
- [ ] I can bring up a working 1-node K8s+GPU Operator cluster from a bare GPU VM in
      **≤ 30 minutes** (did it in lab-gpu-operator).
- [ ] I can write a minimal valid `slurm.conf` NodeName/PartitionName pair and matching
      `gres.conf` for a 1-GPU node from memory.
- [ ] I can name the GPU Operator components in dependency order and say what breaks if each
      is missing.
- [ ] I can state the Slurm upgrade order and the BCM image-based patch workflow.
- [ ] Self-check score ≥ 15/18.
