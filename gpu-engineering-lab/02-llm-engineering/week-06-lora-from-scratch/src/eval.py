"""Day 4 — evaluation: lm-eval-harness (quantitative) + fixed-prompt generations
(qualitative).

Usage:
    python -m src.eval --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter runs/lora-bf16/adapter.pt \
        --tasks hellaswag arc_easy winogrande --out runs/lora-bf16/eval

Honesty note: a 1.5B model tuned on 5k Alpaca rows will move benchmarks little
(sometimes down — Alpaca teaches format, not knowledge). Report the numbers you
get. The qualitative table is where instruction tuning is visible.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

QUALITATIVE_PROMPTS = [
    "Explain what a GPU kernel is to a new engineer in three sentences.",
    "Write a haiku about gradient descent.",
    "List three differences between TCP and UDP.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Give me a recipe idea using only eggs, rice, and spinach.",
    "What does the acronym RAID stand for, and what is RAID 1?",
    "Translate to French: 'The weather is beautiful today.'",
    "Write a polite email declining a meeting invitation.",
]


def load_model_with_optional_adapter(model_name: str, adapter_path: str | None):
    """TODO(Day 4): load base model (bf16, cuda); if adapter_path given,
    re-apply wrap_model_with_lora with the config stored in adapter.pt and
    load the LoRA-only state dict (strict=False). Return (model, tokenizer)."""
    raise NotImplementedError("Day 4: implement load_model_with_optional_adapter")


def run_lm_eval(model, tokenizer, tasks: list[str], out_dir: str) -> dict:
    """TODO(Day 4): run lm-eval-harness programmatically.

    Key pieces (lm-eval >= 0.4):
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)
        results = simple_evaluate(model=lm, tasks=tasks)
    Save results["results"] to {out_dir}/lm_eval.json and return it.
    """
    raise NotImplementedError("Day 4: implement run_lm_eval")


def run_qualitative(model, tokenizer, out_dir: str) -> list[dict]:
    """TODO(Day 4): greedy-ish generations (temperature 0.7, top_p 0.9,
    max_new_tokens 200) for QUALITATIVE_PROMPTS, applying the model's chat
    template (tokenizer.apply_chat_template). Save {out_dir}/generations.json.
    Run this for BOTH base and adapter models and build the side-by-side
    markdown table in RESULTS.md."""
    raise NotImplementedError("Day 4: implement run_qualitative")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="omit to evaluate the base model")
    p.add_argument("--tasks", nargs="+", default=["hellaswag", "arc_easy", "winogrande"])
    p.add_argument("--out", default="runs/eval")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tokenizer = load_model_with_optional_adapter(args.model, args.adapter)
    results = run_lm_eval(model, tokenizer, args.tasks, args.out)
    print(json.dumps(results, indent=2, default=str))
    run_qualitative(model, tokenizer, args.out)


if __name__ == "__main__":
    main()
