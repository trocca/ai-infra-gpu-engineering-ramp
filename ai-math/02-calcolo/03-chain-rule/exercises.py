"""Esercizi sulla lezione 03: la chain rule.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def catena_due_anelli(x):
    """Calcola A MANO la derivata di y = (3x + 1)^2 nel punto x.

    Niente autograd: usa la chain rule come nella SPIEGAZIONE.
    Anello interno u = 3x + 1 con du/dx = 3, anello esterno y = u^2
    con dy/du = 2u. Moltiplica. Restituisci un float.
    Esempio: catena_due_anelli(2.0) restituisce 42.0.
    """
    # TODO
    raise NotImplementedError


def catena_loss_a_mano(w, x, y):
    """Calcola A MANO dloss/dw per il modello a una casa.

    La catena: pred = w * x, err = pred - y, loss = err^2.
    Le pendenze: dloss/derr = 2*err, derr/dpred = 1, dpred/dw = x.
    w, x, y sono float. Restituisci dloss/dw come float.
    """
    # TODO: forward first (compute err), then multiply the three links
    raise NotImplementedError


def catena_loss_autograd(w, x, y):
    """Calcola dloss/dw con autograd, stessa catena dell'esercizio sopra.

    Passi: tensor per w con requires_grad=True, forward, backward,
    restituisci il gradiente come float. I test controllano che il tuo
    risultato a mano e questo coincidano.
    """
    # TODO
    raise NotImplementedError


def catena_tre_anelli(x):
    """Calcola A MANO la derivata di t = ((2x)^2 + 1)^3 nel punto x.

    Tre anelli: u = 2x, v = u^2 + 1, t = v^3.
    Le pendenze: du/dx = 2, dv/du = 2u, dt/dv = 3v^2.
    Calcola prima i valori u e v (forward), poi moltiplica le tre
    pendenze (backward). Restituisci un float.
    """
    # TODO
    raise NotImplementedError
