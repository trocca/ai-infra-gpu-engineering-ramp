"""Esercizi sulla lezione 02: derivate parziali e gradiente.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def parziale_finita(f, punto, indice, h=1e-4):
    """Calcola la derivata parziale di f rispetto a una sola variabile.

    f prende un tensor (il punto) e restituisce un tensor scalare.
    punto e' un tensor 1D. indice dice rispetto a quale componente
    derivare. Ricetta: copia il punto (punto.clone()), aggiungi h solo
    alla componente indice, e usa la differenza finita.
    Restituisci un float.
    """
    # TODO
    raise NotImplementedError


def gradiente_autograd(f, punto):
    """Calcola il gradiente completo di f nel punto dato, con autograd.

    punto e' un tensor 1D SENZA requires_grad. Passi: fanne una copia
    con requires_grad=True (punto.clone().requires_grad_(True)),
    calcola f, chiama backward, restituisci il tensor .grad.
    """
    # TODO
    raise NotImplementedError


def gradiente_ciotola(punto):
    """Restituisci il gradiente ESATTO di f(x, y) = x^2 + y^2 nel punto.

    Niente autograd qui: scrivi la formula a mano. La derivata parziale
    rispetto a x e' 2x, quella rispetto a y e' 2y.
    punto e' un tensor [x, y]. Restituisci un tensor [2x, 2y].
    """
    # TODO
    raise NotImplementedError


def passo_in_discesa(f, punto, lr):
    """Fai un passo nella direzione di discesa piu' ripida.

    Calcola il gradiente di f nel punto (riusa gradiente_autograd) e
    restituisci il nuovo punto: punto - lr * gradiente.
    lr (learning rate) e' la lunghezza del passo. Il nuovo punto deve
    essere un tensor senza requires_grad (usa .detach() se serve).
    Anteprima del modulo 04: questo E' il gradient descent.
    """
    # TODO
    raise NotImplementedError
