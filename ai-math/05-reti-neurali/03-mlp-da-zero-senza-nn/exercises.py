"""Esercizi sulla lezione 03: MLP da zero, senza nn.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def relu_a_mano(x):
    """La ReLU: max(0, x), elemento per elemento.

    Non usare torch.relu ne' F.relu. Due strade possibili:
    torch.clamp, oppure torch.where, oppure x * (x > 0).
    """
    # TODO
    raise NotImplementedError


def forward_mlp(X, W1, b1, W2, b2):
    """Il forward di un MLP a uno strato nascosto.

    X ha una riga per esempio. Le due righe della SPIEGAZIONE:
    h = ReLU(X @ W1 + b1), out = h @ W2 + b2.
    Riusa la tua relu_a_mano. Restituisci out.
    """
    # TODO
    raise NotImplementedError


def conta_parametri(W1, b1, W2, b2):
    """Conta il numero totale di parametri della rete.

    Somma il numero di elementi di ogni tensor (metodo .numel()).
    Restituisci un int. I numeri tipo '7B' nei nomi dei modelli sono
    esattamente questo conteggio, in miliardi.
    """
    # TODO
    raise NotImplementedError


def xor_a_mano():
    """Costruisci A MANO i pesi che risolvono XOR, senza training.

    Restituisci la tupla (W1, b1, W2, b2) con le forme:
    W1 (2, 2), b1 (2,), W2 (2,), b2 scalare.
    La ricetta della SPIEGAZIONE: il primo neurone nascosto calcola
    ReLU(x1 + x2), il secondo ReLU(x1 + x2 - 1), e l'uscita combina
    h1 - 2*h2. Il test verifica la tabella XOR esatta con la TUA
    forward_mlp.
    """
    # TODO
    raise NotImplementedError
