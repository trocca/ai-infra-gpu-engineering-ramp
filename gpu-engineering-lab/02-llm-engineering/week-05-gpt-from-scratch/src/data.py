"""TinyStories data pipeline: download -> tokenize -> uint16 memmap -> batches.

Usage:
    python -m src.data --out data/ --num-tokens 100_000_000     # one-time prep
    get_batch("data/train.bin", batch_size, seq_len, device)     # in train.py

Design: pre-tokenize once into a flat binary of uint16 token ids (gpt2 vocab
50257 < 65536, so uint16 fits). Training then samples random contiguous windows
via np.memmap — zero tokenization cost in the hot loop, near-zero RAM.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

ENCODING = "gpt2"  # tiktoken encoding; swap for a sentencepiece model if you train one


def get_tokenizer():
    """TODO(Day 1): return the tokenizer.

    tiktoken path (recommended):  tiktoken.get_encoding(ENCODING)
    sentencepiece path: train a ~8k-vocab model on a TinyStories sample with
    spm.SentencePieceTrainer, load with spm.SentencePieceProcessor. If you go
    this route, also change GPTConfig.vocab_size and the dtype check below.
    """
    raise NotImplementedError("Day 1: implement get_tokenizer")


def prepare(out_dir: str, num_tokens: int = 100_000_000, val_fraction: float = 0.0005) -> None:
    """TODO(Day 3): build train.bin / val.bin.

    Steps:
      1. datasets.load_dataset("roneneldan/TinyStories", split="train")
      2. Iterate stories, encode each, append an end-of-text token between
         documents (tiktoken: enc.eot_token) — the model must learn where
         stories end.
      3. Stop once num_tokens collected. Split off val_fraction for val.
      4. np.array(ids, dtype=np.uint16).tofile(...) -> train.bin / val.bin.
      5. Print token counts; write them to a small meta.json for provenance.
    """
    raise NotImplementedError("Day 3: implement prepare")


def get_batch(
    bin_path: str,
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO(Day 3): sample a batch of (x, y) windows from the memmap.

    Steps:
      1. data = np.memmap(bin_path, dtype=np.uint16, mode="r")
         (re-creating the memmap each call is the standard nanoGPT trick to
         avoid a memory leak from touched pages accumulating).
      2. Random start offsets ix ~ U[0, len(data) - seq_len - 1), batch_size of them.
      3. x = data[i : i+seq_len], y = data[i+1 : i+1+seq_len]  (next-token targets).
      4. To torch.long; move to device with non_blocking=True after .pin_memory().
    """
    raise NotImplementedError("Day 3: implement get_batch")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare TinyStories token bins")
    p.add_argument("--out", default="data")
    p.add_argument("--num-tokens", type=int, default=100_000_000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prepare(args.out, args.num_tokens)
