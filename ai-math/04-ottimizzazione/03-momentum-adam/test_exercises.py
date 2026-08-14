"""Tests for the exercises of lesson 03: momentum and Adam.

Run `pytest` from this folder. Deterministic: no random numbers.
"""

import torch

import exercises


def f_ciotola(p):
    return (p**2).sum()


def test_passo_momentum_primo_passo():
    w, v = exercises.passo_momentum(
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]), lr=0.1, beta=0.9
    )
    assert torch.allclose(v, torch.tensor([1.0]))
    assert torch.allclose(w, torch.tensor([-0.1]))


def test_passo_momentum_accumula():
    # Constant gradient 1: v should follow 1, 1.9, 2.71 (the hand example).
    w = torch.tensor([0.0])
    v = torch.tensor([0.0])
    attesi = [1.0, 1.9, 2.71]
    for atteso in attesi:
        w, v = exercises.passo_momentum(w, v, torch.tensor([1.0]), lr=0.1, beta=0.9)
        assert torch.allclose(v, torch.tensor([atteso]), atol=1e-6)


def test_passo_momentum_come_torch():
    p = torch.tensor([3.0, -2.0], requires_grad=True)
    opt = torch.optim.SGD([p], lr=0.05, momentum=0.9)
    w = torch.tensor([3.0, -2.0])
    v = torch.zeros(2)
    for _ in range(15):
        opt.zero_grad()
        f_ciotola(p).backward()
        grad = p.grad.clone()
        opt.step()
        w, v = exercises.passo_momentum(w, v, grad, lr=0.05, beta=0.9)
        # Feed OUR w's gradient next time: keep the two in lockstep by
        # comparing after each step instead.
        assert torch.allclose(w, p.detach(), atol=1e-5)


def test_velocita_a_regime():
    assert abs(exercises.velocita_a_regime(0.1, 0.9) - 1.0) < 1e-9
    assert abs(exercises.velocita_a_regime(0.01, 0.99) - 1.0) < 1e-9
    assert abs(exercises.velocita_a_regime(0.1, 0.0) - 0.1) < 1e-9


def test_passo_adam_come_torch_un_passo():
    p = torch.tensor([1.0, -3.0], requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.1)
    opt.zero_grad()
    f_ciotola(p).backward()
    grad = p.grad.clone()
    w0 = torch.tensor([1.0, -3.0])
    opt.step()
    w, m, v = exercises.passo_adam(
        w0, torch.zeros(2), torch.zeros(2), grad, t=1, lr=0.1
    )
    assert torch.allclose(w, p.detach(), atol=1e-6)


def test_passo_adam_come_torch_molti_passi():
    p = torch.tensor([2.0, -1.0], requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.05)
    w = torch.tensor([2.0, -1.0])
    m = torch.zeros(2)
    v = torch.zeros(2)
    for t in range(1, 31):
        opt.zero_grad()
        f_ciotola(p).backward()
        grad = p.grad.clone()
        opt.step()
        w, m, v = exercises.passo_adam(w, m, v, grad, t, lr=0.05)
        assert torch.allclose(w, p.detach(), atol=1e-5), f"divergence at step {t}"


def test_allena_con_adam_trova_il_fondo():
    out = exercises.allena_con_adam(f_ciotola, torch.tensor([4.0, -3.0]), lr=0.3, passi=100)
    assert torch.allclose(out, torch.zeros(2), atol=0.05)
    assert not out.requires_grad
