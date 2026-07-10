"""Fine-tune a 1-2B instruct model with YOUR LoRA (Day 2) or QLoRA (Day 3).

Usage:
    python -m src.train_lora --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset yahma/alpaca-cleaned --num-examples 5000 \
        --r 16 --alpha 32 --targets q_proj k_proj v_proj o_proj \
        --out runs/lora-bf16
    python -m src.train_lora ... --load-4bit --out runs/qlora-nf4

Model access notes:
  - Qwen/Qwen2.5-1.5B-Instruct: ungated, downloads without a token.
  - meta-llama/Llama-3.2-1B-Instruct: GATED — accept the license on the model
    page and `huggingface-cli login` first.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from .lora import count_trainable_parameters, wrap_model_with_lora

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n{input_block}### Response:\n"
)


def load_base_model(model_name: str, load_4bit: bool):
    """TODO(Day 2 + Day 3): load tokenizer + model.

    Day 2 (bf16): AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").
    Day 3 (QLoRA): pass quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True).
    Both: model.gradient_checkpointing_enable(); model.config.use_cache = False
    (checkpointing and KV cache are incompatible during training).
    Tokenizer: set pad_token = eos_token if missing.
    """
    raise NotImplementedError("Day 2: implement load_base_model")


def build_dataset(tokenizer, dataset_name: str, num_examples: int, max_len: int):
    """TODO(Day 2): alpaca-cleaned -> tokenized, LOSS-MASKED examples.

    Steps:
      1. datasets.load_dataset(dataset_name, split="train").shuffle(seed=42)
         .select(range(num_examples)).
      2. Per example: prompt = PROMPT_TEMPLATE (input_block = "### Input:\n{input}\n\n"
         when the input field is non-empty); target = output + eos.
      3. input_ids = enc(prompt) + enc(target); labels = [-100]*len(prompt_ids)
         + enc(target). Key idea: -100 masks the prompt from the loss — you
         train the model to WRITE answers, not to predict instructions.
      4. Truncate to max_len; return list of dicts + a collate_fn that pads
         input_ids with pad_token and labels with -100.
    """
    raise NotImplementedError("Day 2: implement build_dataset")


def train(args: argparse.Namespace) -> None:
    """TODO(Day 2): the loop (plain PyTorch — you already wrote one in week 05).

      1. model/tokenizer = load_base_model(args.model, args.load_4bit)
      2. wrap_model_with_lora(model, args.targets, args.r, args.alpha, args.dropout)
         Print trainable / total params (should be ~0.2-1%).
      3. DataLoader over build_dataset with the pad collate.
      4. AdamW over ONLY trainable params, lr ~2e-4, cosine or constant.
      5. bf16 autocast, grad accumulation to effective batch ~32 sequences,
         clip 1.0. torch.cuda.reset_peak_memory_stats() at start.
      6. Log per step to {out}/log.csv: step, loss, lr, tok/s, peak_vram_gb.
      7. Save LoRA-only state dict at the end:
         {name: p for name, p in model.state_dict().items() if "lora_" in name}
         plus a config dict -> {out}/adapter.pt.  (Small file — that's the point.)
    """
    raise NotImplementedError("Day 2: implement train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning, from scratch")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned")
    p.add_argument("--num-examples", type=int, default=5000)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--load-4bit", action="store_true", help="Day 3: QLoRA (NF4)")
    p.add_argument("--out", default="runs/lora-bf16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    train(a)
