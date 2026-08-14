"""Tests for the lesson 03 exercises: entropy, KL and cross entropy.

Run `pytest` from this folder. Deterministic: no random numbers.
"""

import torch
import torch.nn.functional as F

import exercises


def test_sorpresa_certezza():
    out = exercises.sorpresa(1.0)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.0), atol=1e-6)


def test_sorpresa_evento_raro():
    raro = exercises.sorpresa(0.01)
    comune = exercises.sorpresa(0.9)
    assert raro.item() > comune.item()


def test_entropia_moneta_onesta():
    out = exercises.entropia(torch.tensor([0.5, 0.5]))
    assert torch.allclose(torch.as_tensor(out), torch.log(torch.tensor(2.0)), atol=1e-6)


def test_entropia_moneta_truccata_minore():
    onesta = exercises.entropia(torch.tensor([0.5, 0.5]))
    truccata = exercises.entropia(torch.tensor([0.9, 0.1]))
    assert truccata.item() < onesta.item()


def test_kl_con_se_stessa_e_zero():
    p = torch.tensor([0.9, 0.1])
    out = exercises.kl_divergence(p, p)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.0), atol=1e-6)


def test_kl_come_torch():
    p = torch.tensor([0.9, 0.1])
    q = torch.tensor([0.6, 0.4])
    out = exercises.kl_divergence(p, q)
    expected = F.kl_div(q.log(), p, reduction="sum")
    assert torch.allclose(torch.as_tensor(out), expected, atol=1e-6)


def test_kl_non_simmetrica():
    p = torch.tensor([0.9, 0.1])
    q = torch.tensor([0.5, 0.5])
    assert not torch.allclose(
        torch.as_tensor(exercises.kl_divergence(p, q)),
        torch.as_tensor(exercises.kl_divergence(q, p)),
    )


def test_softmax_a_mano():
    logits = torch.tensor([2.0, 0.5, -1.0])
    out = exercises.softmax_a_mano(logits)
    assert torch.allclose(out, F.softmax(logits, dim=0), atol=1e-6)
    assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-6)


def test_softmax_a_mano_logits_enormi():
    # Without the max trick, exp(1000) overflows to inf.
    logits = torch.tensor([1000.0, 999.0])
    out = exercises.softmax_a_mano(logits)
    assert torch.isfinite(out).all(), "overflow: missing the max trick?"
    assert torch.allclose(out, F.softmax(logits, dim=0), atol=1e-6)


def test_cross_entropy_a_mano():
    logits = torch.tensor([2.0, 0.5, -1.0])
    for classe in range(3):
        out = exercises.cross_entropy_a_mano(logits, classe)
        expected = F.cross_entropy(logits.unsqueeze(0), torch.tensor([classe]))
        assert torch.allclose(torch.as_tensor(out), expected, atol=1e-5)


def test_cross_entropy_punisce_la_sicurezza_sbagliata():
    sicuro_giusto = exercises.cross_entropy_a_mano(torch.tensor([5.0, 0.0]), 0)
    sicuro_sbagliato = exercises.cross_entropy_a_mano(torch.tensor([5.0, 0.0]), 1)
    assert sicuro_sbagliato.item() > 10 * sicuro_giusto.item()
