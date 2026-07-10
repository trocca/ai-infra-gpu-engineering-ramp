#!/usr/bin/env bash
# The demo storyline, scene by scene. Run interactively — each scene pauses
# so you can narrate. Matches Part B of the pitch script.
set -euo pipefail
cd "$(dirname "$0")/.."

pause() { echo; read -rp "── [enter] next scene ──" _; echo; }

echo "SCENE 1 — 'Kubernetes knows about the GPUs' (the platform did this, not me)"
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPUs:.status.capacity.nvidia\.com/gpu'
pause

echo "SCENE 2 — 'One Helm chart': the GPU Operator stack on every node"
kubectl get pods -n gpu-operator -o wide
echo
echo "…and a GPU inside a container, zero node config:"
kubectl run smi --rm -it --restart=Never \
  --image=nvcr.io/nvidia/cuda:13.3.0-base-ubuntu24.04 \
  --overrides='{"spec":{"containers":[{"name":"smi","image":"nvcr.io/nvidia/cuda:13.3.0-base-ubuntu24.04","command":["nvidia-smi"],"resources":{"limits":{"nvidia.com/gpu":"1"}}}]}}' || true
pause

echo "SCENE 3 — 'All I write is this' (open train/train.py, train/Dockerfile,"
echo "train/trainjob.yaml on screen — read the TrainJob out loud: numNodes,"
echo "image, command. Point at what is ABSENT: no IPs, no ranks, no NCCL config.)"
pause

echo "SCENE 4 — submit, and watch the machine work"
kubectl apply -f train/trainjob.yaml
sleep 10
kubectl get pods -l trainer.kubeflow.org/trainjob-name=tinygpt-ddp -o wide
echo
echo "The injected env-var contract (the operator wrote these, not me):"
POD=$(kubectl get pods -l trainer.kubeflow.org/trainjob-name=tinygpt-ddp -o name | head -1)
kubectl describe "$POD" | grep -E "MASTER_ADDR|MASTER_PORT|RANK|WORLD_SIZE|PET_" || true
echo
echo "The headless Service (stable DNS = the rendezvous point):"
kubectl get svc | grep tinygpt || true
echo
echo "Now: kubectl logs -f <rank0-pod>   → watch torchrun rendezvous +"
echo "NCCL_DEBUG=INFO printing the transport it chose (NVLink vs NET/IB)."
pause

echo "SCENE 5 — the scheduler earns its keep (the deadlock demo)"
echo "5a. naive — watch both jobs wedge with partial GPUs:"
kubectl apply -f scheduling/deadlock-naive.yaml
sleep 15
kubectl get pods -l demo=deadlock
echo "   ^ some Running, some Pending, NEITHER job can ever finish. Clean up:"
read -rp "── [enter] to clean up + run the gang version ──" _
kubectl delete -f scheduling/deadlock-naive.yaml --wait=false
echo
echo "5b. gang-scheduled with KAI — all-or-nothing, forward progress:"
kubectl apply -f scheduling/deadlock-gang-kai.yaml
sleep 15
kubectl get pods -l demo=gang
echo "   ^ one job fully Running, the other fully Pending. When A finishes, B runs."
pause

echo "SCENE 6 — serve a model (vLLM, OpenAI-compatible)"
kubectl apply -f serve/vllm-single.yaml
kubectl rollout status deploy/vllm --timeout=10m
kubectl port-forward svc/vllm 8000:8000 &
PF=$!; sleep 3
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Explain NVLink in one sentence:","max_tokens":64}' | head -c 800
kill $PF 2>/dev/null || true
pause

echo "SCENE 7 — 'the line' (back to the stack diagram):"
echo "Everything I touched: train.py, a Dockerfile, ~25 lines of YAML."
echo "Everything else was automatic — and every year NVIDIA moves the line up."
