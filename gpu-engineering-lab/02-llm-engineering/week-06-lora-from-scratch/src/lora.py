"""LoRA implemented by hand. Reference for parity: HuggingFace PEFT.

Contract (tests/test_lora.py depends on these exact signatures):

    LoRALinear(base: nn.Linear-like, r: int, alpha: float, dropout: float = 0.0)
        .lora_A: nn.Parameter (r, in_features)   — kaiming-uniform init
        .lora_B: nn.Parameter (out_features, r)  — ZERO init
        .scaling == alpha / r
        forward(x) = base(x) + dropout(x) @ A^T @ B^T * scaling
        merge_()   — fold scaling * B @ A into base.weight in place
        unmerge_() — undo it

    wrap_model_with_lora(model, target_modules, r, alpha, dropout=0.0) -> model
        Replaces every nn.Linear whose qualified name ENDS WITH one of
        target_modules (e.g. "q_proj") with LoRALinear, in place. Freezes all
        non-LoRA params (requires_grad=False); A and B stay trainable.

    count_trainable_parameters(model) -> int          (provided, COMPLETE)
    iter_lora_modules(model) -> yields (name, LoRALinear)   (provided, COMPLETE)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * B(A(dropout(x)))  with base frozen.

    QLoRA note (Day 3): call `self.base(x)` as a black box. When the base layer
    is a bitsandbytes Linear4bit, its .weight is a packed Params4bit — do NOT
    read base.weight on the forward path. Only merge_() needs a real dense
    weight (merging into a 4-bit base is out of scope — dequantize first or
    merge only bf16 runs).
    """

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert r > 0, "rank must be positive"
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        in_features = base.in_features
        out_features = base.out_features

        # TODO(Day 1):
        #   1. Freeze the base layer (requires_grad_(False) on its params).
        #   2. self.lora_A = nn.Parameter(torch.empty(r, in_features)); init with
        #      nn.init.kaiming_uniform_(a=math.sqrt(5))  (matches PEFT).
        #   3. self.lora_B = nn.Parameter(torch.zeros(out_features, r)).
        #      Key idea: B=0 makes DeltaW = B@A = 0, so at step 0 the wrapped
        #      layer IS the pretrained layer. A must be nonzero or grad(A)=0
        #      forever (grad_A ~ B^T ..., grad_B ~ ... A^T).
        #   4. self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity().
        #   Match A/B dtype+device to the base layer's params.
        raise NotImplementedError("Day 1: implement LoRALinear.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(Day 1): base(x) + dropout(x) @ A^T @ B^T * scaling.
        # If self.merged, just return base(x) (the delta already lives in W).
        # Keep the LoRA path in the same dtype as x (bf16 under autocast).
        raise NotImplementedError("Day 1: implement LoRALinear.forward")

    @torch.no_grad()
    def merge_(self) -> None:
        # TODO(Day 4): base.weight += scaling * (B @ A), cast to weight dtype;
        # set self.merged = True. Idempotence guard: no-op if already merged.
        raise NotImplementedError("Day 4: implement merge_")

    @torch.no_grad()
    def unmerge_(self) -> None:
        # TODO(Day 4): exact inverse of merge_.
        raise NotImplementedError("Day 4: implement unmerge_")


def wrap_model_with_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """TODO(Day 1): programmatic model surgery. No model-specific hardcoding.

    Steps:
      1. Freeze ALL params: p.requires_grad_(False).
      2. Collect replacements first, then apply (never mutate while iterating
         named_modules()):
           for name, module in model.named_modules():
               if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in target_modules:
                   -> replace
      3. Replace via the PARENT: parent = model.get_submodule(name up to last dot);
         setattr(parent, child_name, LoRALinear(module, r, alpha, dropout)).
      4. LoRALinear.__init__ already froze the base and left A/B trainable.
    """
    raise NotImplementedError("Day 1: implement wrap_model_with_lora")


# ----------------------------------------------------------------- provided --

def count_trainable_parameters(model: nn.Module) -> int:
    """COMPLETE. Used by the PEFT-parity test."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def iter_lora_modules(model: nn.Module):
    """COMPLETE. Yields (qualified_name, LoRALinear) pairs."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module
