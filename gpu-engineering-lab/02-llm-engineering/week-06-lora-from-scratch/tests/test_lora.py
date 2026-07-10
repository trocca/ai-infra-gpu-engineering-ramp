"""COMPLETE test harness — do not edit. Oracle: HuggingFace PEFT.

Runs on CPU with a tiny transformer-shaped model (no downloads). The PEFT
parity tests require `pip install peft` and are skipped if it is missing.
"""

import copy

import pytest
import torch
import torch.nn as nn

from src.lora import (
    LoRALinear,
    count_trainable_parameters,
    iter_lora_modules,
    wrap_model_with_lora,
)

torch.manual_seed(0)
D = 32
TARGETS = ["q_proj", "v_proj"]
R, ALPHA = 8, 16


class TinyBlock(nn.Module):
    """Transformer-shaped naming so suffix matching mirrors a real HF model."""

    def __init__(self, d: int = D):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.up_proj = nn.Linear(d, 4 * d, bias=False)
        self.down_proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        a = self.o_proj(torch.tanh(self.q_proj(x)) * self.v_proj(x) + self.k_proj(x))
        return x + a + self.down_proj(torch.nn.functional.gelu(self.up_proj(x)))


class TinyModel(nn.Module):
    def __init__(self, n_blocks: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(TinyBlock() for _ in range(n_blocks))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_wrapped(call):
    model = TinyModel()
    reference = copy.deepcopy(model)
    wrapped = call(wrap_model_with_lora, model, TARGETS, R, ALPHA, 0.0)
    assert wrapped is not None, "wrap_model_with_lora must return the model"
    return wrapped, reference


# -------------------------------------------------------------- unit behavior

def test_lora_linear_shapes_and_scaling(call):
    base = nn.Linear(16, 24, bias=False)
    m = call(LoRALinear, base, R, ALPHA, 0.0)
    assert m.lora_A.shape == (R, 16)
    assert m.lora_B.shape == (24, R)
    assert m.scaling == pytest.approx(ALPHA / R)
    assert torch.all(m.lora_B == 0), "B must be zero-initialized"
    assert not torch.all(m.lora_A == 0), "A must NOT be zero-initialized"
    assert not base.weight.requires_grad, "base must be frozen"


def test_identity_at_init(call):
    """B=0 => wrapped model output equals the original model, exactly-ish."""
    wrapped, reference = make_wrapped(call)
    x = torch.randn(4, 10, D)
    y_wrapped = call(wrapped.forward, x)
    torch.testing.assert_close(y_wrapped, reference(x), atol=1e-6, rtol=1e-6)


def test_wrap_replaces_only_targets(call):
    wrapped, _ = make_wrapped(call)
    lora_names = {name for name, _ in iter_lora_modules(wrapped)}
    assert len(lora_names) == 4  # 2 blocks x (q_proj, v_proj)
    for name in lora_names:
        assert name.rsplit(".", 1)[-1] in TARGETS
    for name, module in wrapped.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in ("k_proj", "o_proj", "up_proj", "down_proj"):
            assert isinstance(module, nn.Linear), f"{name} should be untouched"


def test_only_lora_params_trainable(call):
    wrapped, _ = make_wrapped(call)
    for name, p in wrapped.named_parameters():
        if p.requires_grad:
            assert "lora_" in name, f"non-LoRA param is trainable: {name}"
    expected = 4 * (R * D + D * R)  # 4 wrapped layers, A:(r,in) + B:(out,r)
    assert count_trainable_parameters(wrapped) == expected


def test_gradients_flow_to_lora_only(call):
    wrapped, _ = make_wrapped(call)
    x = torch.randn(2, 5, D)
    call(wrapped.forward, x).sum().backward()
    for name, p in wrapped.named_parameters():
        if "lora_A" in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"no gradient reached {name} (did you forget the LoRA path?)"
        if "lora_" not in name:
            assert p.grad is None, f"gradient leaked into frozen param {name}"


# ------------------------------------------------------------------ merging

def randomize_adapters(model):
    for _, m in iter_lora_modules(model):
        m.lora_B.data.normal_(std=0.05)  # make the delta non-trivial


def test_merge_equivalence(call):
    """merged dense weights must reproduce adapter outputs <= 1e-5 (fp32)."""
    wrapped, _ = make_wrapped(call)
    randomize_adapters(wrapped)
    x = torch.randn(4, 10, D)
    y_adapter = call(wrapped.forward, x)

    for _, m in iter_lora_modules(wrapped):
        call(m.merge_)
    y_merged = wrapped(x)
    torch.testing.assert_close(y_merged, y_adapter, atol=1e-5, rtol=1e-5)


def test_unmerge_restores_base_weights(call):
    wrapped, _ = make_wrapped(call)
    randomize_adapters(wrapped)
    originals = {name: m.base.weight.detach().clone()
                 for name, m in iter_lora_modules(wrapped)}
    for _, m in iter_lora_modules(wrapped):
        call(m.merge_)
        call(m.unmerge_)
    for name, m in iter_lora_modules(wrapped):
        torch.testing.assert_close(m.base.weight, originals[name],
                                   atol=1e-6, rtol=1e-6)


def test_merge_is_idempotent(call):
    wrapped, _ = make_wrapped(call)
    randomize_adapters(wrapped)
    x = torch.randn(2, 4, D)
    for _, m in iter_lora_modules(wrapped):
        call(m.merge_)
    y1 = wrapped(x)
    for _, m in iter_lora_modules(wrapped):
        call(m.merge_)  # second merge must be a no-op
    torch.testing.assert_close(wrapped(x), y1, atol=0, rtol=0)


# -------------------------------------------------------------- PEFT parity

def test_trainable_param_count_matches_peft(call):
    peft = pytest.importorskip("peft", reason="pip install peft for parity tests")

    ours = TinyModel()
    call(wrap_model_with_lora, ours, TARGETS, R, ALPHA, 0.0)
    ours_count = count_trainable_parameters(ours)

    theirs = TinyModel()
    config = peft.LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0,
                             target_modules=TARGETS, bias="none")
    theirs = peft.get_peft_model(theirs, config)
    theirs_count = sum(p.numel() for p in theirs.parameters() if p.requires_grad)

    assert ours_count == theirs_count, (
        f"trainable-param mismatch: ours={ours_count} peft={theirs_count}"
    )


def test_identity_at_init_matches_peft_behavior(call):
    """Both implementations must leave the function unchanged at init."""
    peft = pytest.importorskip("peft", reason="pip install peft for parity tests")
    torch.manual_seed(7)
    base = TinyModel()
    x = torch.randn(2, 6, D)
    y_ref = base(x)

    ours = copy.deepcopy(base)
    call(wrap_model_with_lora, ours, TARGETS, R, ALPHA, 0.0)
    torch.testing.assert_close(call(ours.forward, x), y_ref, atol=1e-6, rtol=1e-6)

    config = peft.LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0,
                             target_modules=TARGETS, bias="none")
    theirs = peft.get_peft_model(copy.deepcopy(base), config)
    torch.testing.assert_close(theirs(x), y_ref, atol=1e-6, rtol=1e-6)
