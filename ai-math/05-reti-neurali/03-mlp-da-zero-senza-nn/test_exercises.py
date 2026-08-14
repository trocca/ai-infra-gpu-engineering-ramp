"""Tests for the lesson 03 exercises: an MLP from scratch, no nn.

Run `pytest` from this folder. Deterministic: fixed seed.
"""

import torch

import exercises

X_XOR = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_XOR = torch.tensor([0.0, 1.0, 1.0, 0.0])


def test_relu_a_mano():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
    out = exercises.relu_a_mano(x)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 0.0, 0.5, 3.0]))


def test_relu_a_mano_matrice():
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    assert torch.allclose(exercises.relu_a_mano(x), torch.relu(x))


def test_forward_mlp_forma():
    torch.manual_seed(0)
    W1 = torch.randn(2, 8)
    b1 = torch.randn(8)
    W2 = torch.randn(8)
    b2 = torch.tensor(0.5)
    out = exercises.forward_mlp(X_XOR, W1, b1, W2, b2)
    assert out.shape == (4,)


def test_forward_mlp_contro_torch():
    torch.manual_seed(1)
    W1 = torch.randn(2, 6)
    b1 = torch.randn(6)
    W2 = torch.randn(6)
    b2 = torch.tensor(-0.3)
    out = exercises.forward_mlp(X_XOR, W1, b1, W2, b2)
    expected = torch.relu(X_XOR @ W1 + b1) @ W2 + b2
    assert torch.allclose(out, expected, atol=1e-6)


def test_conta_parametri():
    W1 = torch.zeros(2, 8)
    b1 = torch.zeros(8)
    W2 = torch.zeros(8)
    b2 = torch.zeros(1)
    # 16 + 8 + 8 + 1 = 33
    assert exercises.conta_parametri(W1, b1, W2, b2) == 33


def test_xor_a_mano_forme():
    W1, b1, W2, b2 = exercises.xor_a_mano()
    assert W1.shape == (2, 2)
    assert b1.shape == (2,)
    assert W2.shape == (2,)


def test_xor_a_mano_tabella_esatta():
    W1, b1, W2, b2 = exercises.xor_a_mano()
    out = exercises.forward_mlp(X_XOR, W1, b1, W2, b2)
    assert torch.allclose(out, Y_XOR, atol=1e-6), (
        "the hand-built network must reproduce XOR exactly"
    )
