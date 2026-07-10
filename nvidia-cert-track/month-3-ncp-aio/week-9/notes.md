# Week 9 Notes — Installation & Deployment

## BCM architecture

- Head node roles:
- Node categories vs software images:
- HA (`cmha`):
- Base View vs cmsh:

## Node provisioning

- PXE → node-installer → image sync flow:
- Full vs sync provisioning:
- `cm-chroot-sw-img` workflow:
- `imageupdate` vs reboot:

## cmsh command bank (write from memory, verify against manual)

- Device mode:
- Category mode:
- Softwareimage mode:
- Network mode:
- User mode:
- Monitoring mode:

## Slurm setup

- Daemons (slurmctld / slurmd / slurmdbd / munge):
- slurm.conf key lines:
- gres.conf (AutoDetect=nvml vs explicit):
- cons_tres:
- cm-wlm-setup:

## Kubernetes setup

- kubeadm sequence:
- containerd + runtime config:
- cm-kubernetes-setup:
- Control-plane taints, kubeconfig:

## GPU Operator

- Component stack in order:
- driver.enabled=false — when:
- CDI:
- ClusterPolicy:
- Helm install command:

## Networking

- BCM network objects:
- Management / compute / storage / OOB networks:
- IB vs RoCE vs Ethernet, rails:
- Link to NCCL transports (demo repo):

## Monitoring

- BCM monitoring (metrics, healthchecks, actions):
- dcgm-exporter → Prometheus:
- Key DCGM_FI_* metrics:

## Patching & upgrades

- BCM image-based patching:
- Slurm upgrade order:
- GPU Operator / driver upgrades:

## User management

- BCM users (cmsh user mode, LDAP):
- sacctmgr accounts/users:
- K8s identity model:

## Misses / to-revisit

-
