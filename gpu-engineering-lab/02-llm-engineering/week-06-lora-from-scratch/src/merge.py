"""Day 4 — fold LoRA adapters into dense weights and save a standalone model.

Usage:
    python -m src.merge --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter runs/lora-bf16/adapter.pt --out runs/lora-bf16/merged

Why merging is free lunch at inference: y = Wx + (alpha/r)BAx = (W + (alpha/r)BA)x.
One weight update, zero extra latency forever after — unlike bottleneck adapters
or prefix tuning, which stay in the forward pass.

Only merge bf16 runs. Merging into a 4-bit base means dequantize -> add ->
requantize, which changes the quantization error — out of scope this week.
"""

from __future__ import annotations

import argparse
import os

import torch

from .lora import LoRALinear, iter_lora_modules, wrap_model_with_lora


def merge_and_save(model_name: str, adapter_path: str, out_dir: str) -> None:
    """TODO(Day 4):
      1. Load base model in bf16 on CPU or GPU.
      2. Re-apply wrap_model_with_lora using the config saved in adapter.pt;
         load the LoRA state dict (strict=False must report ZERO missing
         lora_ keys — assert it).
      3. For each LoRALinear: module.merge_().
      4. Replace each LoRALinear with its (now-updated) base layer, so the
         saved model has NO custom classes (same parent-setattr trick as
         wrap_model_with_lora, in reverse).
      5. model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir).
    """
    raise NotImplementedError("Day 4: implement merge_and_save")


@torch.no_grad()
def verify_merge(model_name: str, adapter_path: str, merged_dir: str,
                 atol: float = 1e-4) -> None:
    """TODO(Day 4): load adapter model and merged model, run both on ~5 fixed
    prompts in fp32 (cast for the comparison), assert max |logits diff| <= atol.
    Print the max diff — this number goes in RESULTS.md."""
    raise NotImplementedError("Day 4: implement verify_merge")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    merge_and_save(args.model, args.adapter, args.out)
    if args.verify:
        verify_merge(args.model, args.adapter, args.out)
