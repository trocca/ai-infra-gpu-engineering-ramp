"""Tests for the lesson 02 exercises: likelihood and MLE.

Run `pytest` from this folder. Deterministic: seeds are fixed.
"""

import torch

import exercises

FLIPS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def test_likelihood_moneta_mezzo():
    out = exercises.likelihood_moneta(0.5, FLIPS)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.5**10), atol=1e-8)


def test_likelihood_moneta_ordina_le_ipotesi():
    l_05 = exercises.likelihood_moneta(0.5, FLIPS)
    l_07 = exercises.likelihood_moneta(0.7, FLIPS)
    l_09 = exercises.likelihood_moneta(0.9, FLIPS)
    assert l_07 > l_05, "p=0.7 must be more plausible than p=0.5"
    assert l_07 > l_09, "p=0.7 must be more plausible than p=0.9"


def test_log_likelihood_coerente_col_log():
    p = 0.6
    log_diretta = exercises.log_likelihood_moneta(p, FLIPS)
    log_indiretta = torch.log(exercises.likelihood_moneta(p, FLIPS))
    assert torch.allclose(torch.as_tensor(log_diretta), log_indiretta, atol=1e-5)


def test_mle_moneta():
    out = exercises.mle_moneta(FLIPS)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.7))


def test_mle_moneta_altra_sequenza():
    lanci = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = exercises.mle_moneta(lanci)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.25))


def test_mle_su_griglia():
    griglia = torch.linspace(0.01, 0.99, 99)
    out = exercises.mle_su_griglia(FLIPS, griglia)
    assert abs(out.item() - 0.7) < 0.011


def test_nll_gaussiana_formula():
    dati = torch.tensor([1.0, 2.0, 3.0])
    mu, sigma = 2.0, 1.5
    n = 3
    expected = n * torch.log(
        torch.tensor(sigma) * torch.sqrt(torch.tensor(2 * torch.pi))
    ) + ((dati - mu) ** 2).sum() / (2 * sigma**2)
    out = exercises.nll_gaussiana(mu, dati, sigma)
    assert torch.allclose(torch.as_tensor(out), expected, atol=1e-5)


def test_nll_gaussiana_minima_sulla_media():
    torch.manual_seed(0)
    dati = torch.distributions.Normal(250.0, 20.0).sample((500,))
    media = dati.mean().item()
    nll_media = exercises.nll_gaussiana(media, dati, 20.0)
    for mu in [media - 10, media - 1, media + 1, media + 10]:
        assert exercises.nll_gaussiana(mu, dati, 20.0) > nll_media
