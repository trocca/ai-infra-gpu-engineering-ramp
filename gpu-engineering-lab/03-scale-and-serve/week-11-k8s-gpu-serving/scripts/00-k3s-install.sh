#!/usr/bin/env bash
# COMPLETE — k3s single-node install for the week-11 GPU node.
# Run as a sudo-capable user on a fresh Ubuntu 22.04/24.04 cloud node.
# Read every line before running; you will be asked to explain them.
set -euo pipefail

echo "=== [1/3] Installing k3s (single node, traefik disabled — we don't need an ingress) ==="
# --disable traefik : less noise on a lab node
# --write-kubeconfig-mode 644 : lets the non-root user run kubectl
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik --write-kubeconfig-mode 644" sh -

echo "=== [2/3] kubectl access for this user ==="
mkdir -p "$HOME/.kube"
cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config" 2>/dev/null || sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
export KUBECONFIG="$HOME/.kube/config"

echo "=== [3/3] Waiting for the node to be Ready ==="
kubectl wait --for=condition=Ready node --all --timeout=180s
kubectl get nodes -o wide

echo
echo "k3s up. NOTE for the GPU Operator step: k3s runs its OWN containerd at"
echo "  /run/k3s/containerd/containerd.sock"
echo "The GPU Operator install script passes the toolkit env vars for this."
echo "Next: ./scripts/01-gpu-operator.sh"
