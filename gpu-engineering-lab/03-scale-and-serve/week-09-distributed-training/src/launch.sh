#!/usr/bin/env bash
# COMPLETE — torchrun launch recipes for week 09. Do not edit; copy-paste.
#
# Usage: ./src/launch.sh <recipe> [extra args passed to train_dist.py]
set -euo pipefail
cd "$(dirname "$0")/.."

RECIPE="${1:-help}"
shift || true

case "$RECIPE" in
  # ---- Day 1: local, CPU, gloo — free, debuggable, no GPU needed ----
  local-cpu)
    torchrun --standalone --nproc_per_node=2 \
      -m src.train_dist --backend gloo --impl "${IMPL:-manual}" "$@"
    ;;

  # ---- Day 1: local, single RTX 5090 — API familiarity ----
  local-1gpu)
    torchrun --standalone --nproc_per_node=1 \
      -m src.train_dist --backend nccl --impl single "$@"
    ;;

  # ---- Days 2-3: cloud node, 2 GPUs, single node ----
  cloud-2gpu)
    torchrun --standalone --nproc_per_node=2 \
      -m src.train_dist --backend nccl --impl "${IMPL:-manual}" "$@"
    ;;

  # ---- Day 3: the three-way loss-parity comparison, one command ----
  compare)
    for impl in manual ddp fsdp; do
      echo "=== impl=$impl ==="
      torchrun --standalone --nproc_per_node=2 \
        -m src.train_dist --backend nccl --impl "$impl" \
        --out "bench/results/train_${impl}_ws2.json" "$@"
    done
    torchrun --standalone --nproc_per_node=1 \
      -m src.train_dist --backend nccl --impl single \
      --out "bench/results/train_single_ws1.json" "$@"
    ;;

  # ---- Day 4: NCCL transport visibility ----
  nccl-debug)
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
    torchrun --standalone --nproc_per_node=2 \
      -m src.train_dist --backend nccl --impl ddp --steps 20 "$@" \
      2>&1 | tee bench/results/nccl_debug.log
    grep -E "via|Channel|Trees|Ring" bench/results/nccl_debug.log | sort -u || true
    ;;

  # ---- Day 4: build + run nccl-tests (run from the cloud node) ----
  nccl-tests)
    if [ ! -d /tmp/nccl-tests ]; then
      git clone https://github.com/NVIDIA/nccl-tests /tmp/nccl-tests
      make -C /tmp/nccl-tests -j"$(nproc)" MPI=0
    fi
    /tmp/nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 2 \
      | tee bench/results/nccl_tests_allreduce.txt
    ;;

  *)
    echo "recipes: local-cpu | local-1gpu | cloud-2gpu | compare | nccl-debug | nccl-tests"
    echo "env: IMPL=manual|ddp|fsdp (default manual)"
    exit 1
    ;;
esac
