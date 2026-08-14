"""Test per gli esercizi della lezione 02: SGD e minibatch.

Esegui `pytest` da questa cartella. Deterministici: seed fissati.
"""

import torch

import exercises

X5 = torch.tensor(
    [
        [-1.26, -1.23],
        [-0.63, -0.35],
        [0.00, -0.35],
        [0.63, 0.53],
        [1.26, 1.40],
    ]
)
Y5 = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])


def test_dividi_in_minibatch_taglie():
    batches = exercises.dividi_in_minibatch(X5, Y5, 2)
    assert len(batches) == 3
    taglie = [len(b[0]) for b in batches]
    assert taglie == [2, 2, 1]


def test_dividi_in_minibatch_contenuto():
    batches = exercises.dividi_in_minibatch(X5, Y5, 2)
    assert torch.equal(batches[0][0], X5[:2])
    assert torch.equal(batches[0][1], Y5[:2])
    assert torch.equal(batches[2][1], Y5[4:])


def test_mescola_dataset_riproducibile():
    Xa, ya = exercises.mescola_dataset(X5, Y5, seed=7)
    Xb, yb = exercises.mescola_dataset(X5, Y5, seed=7)
    assert torch.equal(Xa, Xb) and torch.equal(ya, yb)


def test_mescola_dataset_allineato():
    # Every (row, label) pair must survive the shuffle together.
    Xa, ya = exercises.mescola_dataset(X5, Y5, seed=3)
    for i in range(5):
        j = (X5 == Xa[i]).all(dim=1).nonzero()[0].item()
        assert ya[i].item() == Y5[j].item(), "riga e prezzo disallineati"


def test_gradiente_minibatch_contro_formula():
    w = torch.tensor([10.0, 5.0])
    b = torch.tensor([50.0])
    gw, gb = exercises.gradiente_minibatch(w, b, X5, Y5)
    err = X5 @ w + b - Y5
    expected_gw = 2 / len(Y5) * X5.T @ err
    expected_gb = 2 * err.mean()
    assert torch.allclose(gw, expected_gw, atol=1e-4)
    assert torch.allclose(torch.as_tensor(gb).squeeze(), expected_gb, atol=1e-4)


def test_epoca_sgd_riduce_la_loss():
    w = torch.zeros(2)
    b = torch.zeros(1)

    def loss_totale(w, b):
        return ((X5 @ w + b - Y5) ** 2).mean().item()

    prima = loss_totale(w, b)
    for epoch in range(30):
        w, b = exercises.epoca_sgd(w, b, X5, Y5, lr=0.1, batch_size=2, seed=epoch)
    dopo = loss_totale(w, b)
    assert dopo < prima / 10, f"la loss doveva crollare: prima {prima}, dopo {dopo}"
    assert not w.requires_grad and not b.requires_grad
