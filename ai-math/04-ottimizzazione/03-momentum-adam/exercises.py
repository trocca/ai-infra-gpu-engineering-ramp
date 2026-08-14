"""Esercizi sulla lezione 03: momentum e Adam.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def passo_momentum(w, v, grad, lr, beta):
    """Applica un passo di gradient descent con momentum.

    Formule (le stesse di torch.optim.SGD con momentum):
    v nuovo = beta * v + grad
    w nuovo = w - lr * v nuovo
    Restituisci la tupla (w_nuovo, v_nuovo).
    """
    # TODO
    raise NotImplementedError


def velocita_a_regime(lr, beta):
    """Calcola il passo effettivo a regime con gradiente costante 1.

    Con gradiente sempre uguale a 1, la velocita' v converge a una
    somma geometrica: 1 + beta + beta^2 + ... = 1 / (1 - beta).
    Il passo effettivo a regime e' quindi lr / (1 - beta).
    Restituisci quel numero come float. Con beta = 0.9 il momentum
    amplifica il passo di 10 volte: ecco perche' accelera.
    """
    # TODO
    raise NotImplementedError


def passo_adam(w, m, v, grad, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Applica un passo di Adam. t e' il numero del passo, da 1 in su.

    Le sei righe della SPIEGAZIONE:
    m nuovo = beta1 * m + (1 - beta1) * grad
    v nuovo = beta2 * v + (1 - beta2) * grad^2
    m_hat = m nuovo / (1 - beta1^t)
    v_hat = v nuovo / (1 - beta2^t)
    w nuovo = w - lr * m_hat / (sqrt(v_hat) + eps)
    Restituisci la tupla (w_nuovo, m_nuovo, v_nuovo).
    Il test confronta la tua implementazione con torch.optim.Adam.
    """
    # TODO
    raise NotImplementedError


def allena_con_adam(f, w0, lr, passi):
    """Minimizza f partendo da w0 usando il TUO passo_adam.

    f prende un tensor e restituisce un tensor scalare. Per ogni passo
    t da 1 a passi: calcola il gradiente di f (con autograd, come nelle
    lezioni precedenti), poi aggiorna con passo_adam.
    m e v partono da zeri della stessa forma di w0.
    Restituisci il punto finale (tensor senza requires_grad).
    """
    # TODO
    raise NotImplementedError
