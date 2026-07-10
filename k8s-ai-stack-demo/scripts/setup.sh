#!/usr/bin/env bash
# One-time platform install — "everything below the line", installed once.
# Idempotent-ish; run pieces individually if you prefer.
set -euo pipefail

echo "==> [Layer 2] GPU Operator (driver, toolkit, device plugin, DCGM, MIG mgr)"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
kubectl get ns gpu-operator >/dev/null 2>&1 || kubectl create ns gpu-operator
helm upgrade --install gpu-operator nvidia/gpu-operator \
  -n gpu-operator --wait --timeout 15m

echo "==> [Layer 7] KAI Scheduler (gangs, queues, fair-share, fractional GPU)"
helm repo add nvidia-k8s https://nvidia.github.io/KAI-Scheduler
helm repo update
helm upgrade --install kai-scheduler nvidia-k8s/kai-scheduler \
  -n kai-scheduler --create-namespace --wait

echo "==> [Layer 8] Kubeflow Trainer v2 (TrainJob CRD + runtimes; includes JobSet)"
kubectl apply --server-side -k \
  "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=v2.0.0"
# ships ClusterTrainingRuntimes incl. torch-distributed; verify:
kubectl get clustertrainingruntime || true

echo "==> [Layer 8] LeaderWorkerSet (multi-node inference groups)"
LWS_VERSION=v0.9.0   # re-pin to latest release before demoing
kubectl apply --server-side -f \
  "https://github.com/kubernetes-sigs/lws/releases/download/${LWS_VERSION}/manifests.yaml"

echo
echo "Platform installed. Sanity checks:"
echo "  kubectl get pods -n gpu-operator"
echo "  kubectl get pods -n kai-scheduler"
echo "  kubectl get nodes -o json | jq '.items[].status.capacity' | grep nvidia"
