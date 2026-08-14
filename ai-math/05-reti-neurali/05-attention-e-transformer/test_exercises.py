"""Test per gli esercizi della lezione 05: attention e transformer.

Esegui `pytest` da questa cartella. Deterministici: seed fissato.
"""

import torch
import torch.nn.functional as F

import exercises

torch.manual_seed(0)
N, D = 4, 3
Q = torch.randn(N, D)
K = torch.randn(N, D)
V = torch.randn(N, D)


def test_softmax_per_righe():
    M = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = exercises.softmax_per_righe(M)
    assert torch.allclose(out, F.softmax(M, dim=-1), atol=1e-6)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_softmax_per_righe_stabile():
    M = torch.tensor([[1000.0, 999.0], [-1000.0, -1000.0]])
    out = exercises.softmax_per_righe(M)
    assert torch.isfinite(out).all(), "overflow: manca il trucco del massimo per riga?"


def test_punteggi_attenzione():
    out = exercises.punteggi_attenzione(Q, K)
    assert out.shape == (N, N)
    expected = Q @ K.T / (D**0.5)
    assert torch.allclose(out, expected, atol=1e-6)


def test_punteggi_esempio_a_mano():
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = exercises.punteggi_attenzione(q, k)
    expected = torch.tensor([[1.0, 0.0]]) / (2**0.5)
    assert torch.allclose(out, expected, atol=1e-5)


def test_attenzione_contro_torch():
    out = exercises.attenzione(Q, K, V)
    expected = F.scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(out, expected, atol=1e-5)


def test_maschera_causale():
    mask = exercises.maschera_causale(3)
    assert mask.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            if j > i:
                assert mask[i, j].item() == float("-inf")
            else:
                assert mask[i, j].item() == 0.0


def test_attenzione_causale_contro_torch():
    out = exercises.attenzione_causale(Q, K, V)
    expected = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert torch.allclose(out, expected, atol=1e-5)


def test_attenzione_causale_primo_token():
    # Token 0 can only see itself: its output must be exactly V[0].
    out = exercises.attenzione_causale(Q, K, V)
    assert torch.allclose(out[0], V[0], atol=1e-5)
