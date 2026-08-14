"""Esercizi sulla lezione 04: autograd sotto il cofano.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def grad_locale_prodotto(a, b):
    """Le derivate locali del prodotto f = a * b.

    Regola del prodotto, la piu' usata di tutto il backward: la
    derivata rispetto a un fattore e' L'ALTRO fattore.
    a e b sono float. Restituisci la tupla (df/da, df/db) come float.
    """
    # TODO
    raise NotImplementedError


def grad_attraverso_relu(z, grad_in_arrivo):
    """Propaga il gradiente attraverso la ReLU.

    z e' il valore PRIMA della ReLU (tensor), grad_in_arrivo e' il
    gradiente che arriva dall'alto (tensor, stessa forma).
    Regola: dove z > 0 la ReLU e' un filo dritto (il gradiente passa
    intatto), dove z <= 0 la ReLU ha tagliato (il gradiente muore: 0).
    Restituisci il gradiente propagato.
    """
    # TODO
    raise NotImplementedError


def backprop_lineare(x, w, b, t):
    """Backprop A MANO sul modello scalare: loss = (w*x + b - t)^2.

    Tutti gli argomenti sono float. La catena, come in SPIEGAZIONE:
    out = w*x + b, err = out - t, loss = err^2.
    Derivate: dloss/derr = 2*err, poi derr/dout = 1,
    dout/dw = x, dout/db = 1.
    Restituisci la tupla (dloss_dw, dloss_db) come float. Niente
    autograd: il test la usa per controllarti.
    """
    # TODO
    raise NotImplementedError


def backprop_due_strati(x, w1, w2, t):
    """Backprop A MANO su due strati scalari SENZA attivazione.

    Modello: h = w1 * x, out = w2 * h, loss = (out - t)^2.
    Tutti float. Calcola prima il forward (h, out, err), poi scendi:
    dloss/dout = 2*err
    dloss/dw2 = dloss/dout * h        (regola del prodotto)
    dloss/dh  = dloss/dout * w2       (il gradiente continua a scendere)
    dloss/dw1 = dloss/dh * x
    Restituisci la tupla (dloss_dw1, dloss_dw2) come float.
    Questo e' il backward di una "rete" a due strati, nudo e crudo.
    """
    # TODO
    raise NotImplementedError
