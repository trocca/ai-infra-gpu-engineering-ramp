#!/usr/bin/env bash
# COMPLETE — GPU Operator + monitoring + KAI Scheduler installs (Helm).
# Chart repos and names are real; versions float to latest stable — pin
# them in your writeup once you know what you deployed
# (`helm list -A` after install).
set -euo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

command -v helm >/dev/null || {
  echo "installing helm..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
}

SECTION="${1:-all}"

# ============================================================================
# Section 1: NVIDIA GPU Operator
# repo: https://helm.ngc.nvidia.com/nvidia   chart: gpu-operator
# ============================================================================
install_gpu_operator() {
  echo "=== GPU Operator ==="
  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
  helm repo update

  # driver.enabled=true       : Operator installs the driver (node has none — that's deliberate)
  # toolkit env CONTAINERD_*  : point the container toolkit at K3S's containerd, not a system one
  # dcgmExporter serviceMonitor: so kube-prometheus-stack auto-discovers GPU metrics (Day 3)
  helm upgrade --install gpu-operator nvidia/gpu-operator \
    -n gpu-operator --create-namespace \
    --set driver.enabled=true \
    --set toolkit.env[0].name=CONTAINERD_CONFIG \
    --set toolkit.env[0].value=/var/lib/rancher/k3s/agent/etc/containerd/config.toml \
    --set toolkit.env[1].name=CONTAINERD_SOCKET \
    --set toolkit.env[1].value=/run/k3s/containerd/containerd.sock \
    --set toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS \
    --set toolkit.env[2].value=nvidia \
    --set dcgmExporter.serviceMonitor.enabled=true \
    --wait --timeout 15m

  echo "--- verification ---"
  kubectl get pods -n gpu-operator
  kubectl get runtimeclass
  echo "--- CUDA smoke test pod ---"
  kubectl run cuda-vectoradd --rm -it --restart=Never \
    --image=nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04 \
    --overrides='{"spec":{"runtimeClassName":"nvidia","containers":[{"name":"cuda-vectoradd","image":"nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04","resources":{"limits":{"nvidia.com/gpu":1}}}]}}' \
    || echo "smoke test failed — debug before continuing (kubectl describe node | grep nvidia)"
}

# ============================================================================
# Section 2: kube-prometheus-stack
# repo: https://prometheus-community.github.io/helm-charts
# chart: kube-prometheus-stack
# ============================================================================
install_monitoring() {
  echo "=== kube-prometheus-stack ==="
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  helm repo update
  helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
    -n monitoring --create-namespace \
    -f "$(dirname "$0")/../k8s/monitoring-values.yaml" \
    --wait --timeout 10m
  echo "Grafana: http://<node-ip>:30300  (admin / see monitoring-values.yaml)"
}

# ============================================================================
# Section 3: KAI Scheduler
# GitHub: https://github.com/NVIDIA/KAI-Scheduler
# Charts are published as OCI packages on ghcr.io (see the repo's releases
# for the current tag; v0.5+ as of early 2026).
# ============================================================================
install_kai() {
  echo "=== KAI Scheduler ==="
  KAI_VERSION="${KAI_VERSION:-$(curl -sf https://api.github.com/repos/NVIDIA/KAI-Scheduler/releases/latest | grep -oP '"tag_name":\s*"\K[^"]+' || echo v0.5.5)}"
  echo "installing KAI ${KAI_VERSION}"
  helm upgrade --install kai-scheduler \
    oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler \
    -n kai-scheduler --create-namespace \
    --version "${KAI_VERSION}" \
    --set "global.gpuSharing=true" \
    --wait --timeout 10m
  kubectl get pods -n kai-scheduler
  echo "next: kubectl apply -f k8s/kai/queue.yaml"
}

case "$SECTION" in
  gpu-operator) install_gpu_operator ;;
  monitoring)   install_monitoring ;;
  kai)          install_kai ;;
  all)          install_gpu_operator; install_monitoring; install_kai ;;
  *) echo "usage: $0 [gpu-operator|monitoring|kai|all]"; exit 1 ;;
esac
