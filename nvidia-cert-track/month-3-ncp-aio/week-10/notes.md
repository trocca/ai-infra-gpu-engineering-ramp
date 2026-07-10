# Week 10 Notes — Administration

## Slurm administration

- Node states & transitions (idle/mix/alloc/drain/down):
- scontrol verbs I'll actually use:
- Partition config knobs:
- Associations tree (cluster → account → user):
- sacctmgr command bank:
- QOS: limits, priority, preemption:
- Fairshare / multifactor priority:
- sacct vs sreport:

## Run:ai

- Control plane vs cluster:
- Projects ↔ namespaces:
- Departments:
- Quota / over-quota / reclaim:
- Preemptible vs non-preemptible:
- Workload types:
- Fractional GPUs (fraction vs memory request):
- Node pools:

## Run:ai ↔ KAI mapping (tie to demo repo)

- Project+quota ↔ queue:
- Gang scheduling ↔ PodGroup:
- Over-quota weight:
- Preemption/reclaim behavior observed in lab:

## Kubernetes administration

- Role vs ClusterRole, bindings:
- ServiceAccounts:
- ResourceQuota incl. GPUs:
- LimitRange:
- cordon / drain / uncordon, PDBs:
- Taints & tolerations vs affinity for GPU nodes:

## MIG configuration

- GI vs CI:
- A100-40GB profile table:
- Which GPUs support MIG / which don't:
- nvidia-smi mig command bank:
- MIG Manager: node label, config ConfigMap:
- mig.strategy single vs mixed (resource names):
- mig-parted:
- MIG vs time-slicing vs MPS (demo repo cross-ref):

## Datacenter architecture

- DGX/HGX node anatomy:
- NVSwitch/NVLink domain:
- SuperPOD scalable unit:
- Leaf-spine, rail-optimized fabric:
- Power/cooling constraints:
- Storage tiers:
- Which plane on which network:

## Misses / to-revisit

-
