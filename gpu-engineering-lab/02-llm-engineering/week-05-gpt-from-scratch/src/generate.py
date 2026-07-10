"""KV-cache generation + sampling (temperature / top-k / top-p).

Contract (tests/test_kv_cache.py depends on these exact signatures):

    KVCache(n_layer, batch_size, n_head, max_seq_len, d_head, device, dtype)
        .seq_len -> int                  # tokens currently cached
        .update(layer_idx, k, v) -> (k_all, v_all)
            k, v: (B, n_head, T_new, d_head); returns the FULL cached
            (B, n_head, seq_len_after, d_head) views for that layer.
            Only advance .seq_len after the LAST layer of a step.

    generate(model, idx, max_new_tokens, temperature=1.0, top_k=None,
             top_p=None, use_kv_cache=True) -> LongTensor (B, T + max_new_tokens)
        temperature == 0.0 means GREEDY (argmax) — this is what the
        cache-vs-recompute equivalence test uses.

Usage:
    python -m src.generate --ckpt checkpoints/d12/best.pt \
        --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8 --top-p 0.95
"""

from __future__ import annotations

import argparse

import torch


class KVCache:
    """Preallocated per-layer key/value tensors + an append cursor.

    Key idea: without a cache, decoding token N re-computes attention keys and
    values for all N-1 previous tokens in every layer — O(N^2) total work.
    Caching makes each new token O(N): one qkv projection for the new token, a
    dot product against stored keys. Decode then becomes memory-bandwidth-bound
    (streaming the cache + weights), which is why week 08 pages this structure.
    """

    def __init__(self, n_layer: int, batch_size: int, n_head: int,
                 max_seq_len: int, d_head: int,
                 device: torch.device | str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        # TODO(Day 4): allocate self.k and self.v of shape
        # (n_layer, batch_size, n_head, max_seq_len, d_head), zeros, on
        # device/dtype. Set self.seq_len = 0. Preallocating (vs torch.cat every
        # step) avoids an O(N^2) total-copy pattern and allocator churn.
        raise NotImplementedError("Day 4: implement KVCache.__init__")

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO(Day 4): write k, v into rows [seq_len : seq_len + T_new] of
        # layer layer_idx; return the [:seq_len + T_new] slices. Advance
        # self.seq_len by T_new only when layer_idx == n_layer - 1 (all layers
        # must see the same "past length" within one forward pass).
        raise NotImplementedError("Day 4: implement KVCache.update")


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: int | None = None, top_p: float | None = None
                      ) -> torch.Tensor:
    """Pick the next token id from last-position logits (B, vocab) -> (B, 1).

    TODO(Day 4):
      - temperature == 0.0: return argmax (greedy). Deterministic — used by tests.
      - else divide logits by temperature.
      - top_k: keep the k highest logits, set the rest to -inf (torch.topk).
      - top_p (nucleus): sort logits desc, softmax, cumulative-sum; mask tokens
        AFTER cumulative prob exceeds top_p (always keep the first token);
        scatter the mask back to unsorted order. Key idea: adapts the candidate
        set to the distribution's shape — flat distribution keeps many tokens,
        peaked keeps few — which fixed top-k cannot do.
      - softmax -> torch.multinomial(probs, 1).
    """
    raise NotImplementedError("Day 4: implement sample_next_token")


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, use_kv_cache: bool = True
             ) -> torch.Tensor:
    """TODO(Day 4): two paths, identical outputs at temperature 0.

    use_kv_cache=False (naive baseline — keep it, it is the benchmark control):
      loop: logits, _ = model(idx_cropped_to_max_seq_len); sample from the last
      position; torch.cat the new token onto idx.

    use_kv_cache=True:
      1. Build a KVCache from model.config and idx.shape.
      2. PREFILL: one forward over the whole prompt with kv_cache — populates
         every layer's cache and yields the first next-token logits.
      3. DECODE: loop feeding ONLY the newly sampled token (B, 1); RoPE offset
         and attention span come from cache.seq_len.
      model.eval() first; respect model.config.max_seq_len (stop or crop).
    """
    raise NotImplementedError("Day 4: implement generate")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate from a trained checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-kv-cache", action="store_true")
    args = p.parse_args()

    # TODO(Day 4): load checkpoint (weights_only=False for the config dataclass,
    # or store config as a dict), rebuild GPT, encode the prompt with the
    # tokenizer from src.data, call generate(), decode, print.
    raise NotImplementedError("Day 4: implement main")


if __name__ == "__main__":
    main()
