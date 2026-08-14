"""Soluzioni complete degli esercizi della lezione 03: MLP da zero.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def relu_a_mano(x):
    """La ReLU: max(0, x), elemento per elemento."""
    return torch.clamp(x, min=0)


def forward_mlp(X, W1, b1, W2, b2):
    """Il forward di un MLP a uno strato nascosto."""
    h = relu_a_mano(X @ W1 + b1)
    return h @ W2 + b2


def conta_parametri(W1, b1, W2, b2):
    """Conta il numero totale di parametri della rete."""
    return W1.numel() + b1.numel() + W2.numel() + b2.numel()


def xor_a_mano():
    """Costruisci A MANO i pesi che risolvono XOR, senza training."""
    W1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    b1 = torch.tensor([0.0, -1.0])
    W2 = torch.tensor([1.0, -2.0])
    b2 = torch.tensor(0.0)
    return W1, b1, W2, b2
