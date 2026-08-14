"""Esercizi sulla lezione 01: vettori.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella: quando tutti i test passano, hai finito.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def crea_vettore(numeri):
    """Crea un tensor PyTorch a partire da una lista di numeri Python.

    Il tensor deve avere dtype torch.float32.
    Esempio: crea_vettore([1, 2, 3]) restituisce tensor([1., 2., 3.]).
    """
    # TODO: use torch.tensor with an explicit dtype
    raise NotImplementedError


def scala_vettore(v, c):
    """Moltiplica il vettore v per lo scalare c.

    Esempio: scala_vettore(tensor([2., 1.]), 3) restituisce tensor([6., 3.]).
    """
    # TODO
    raise NotImplementedError


def combinazione_lineare(u, v, a, b):
    """Restituisci la combinazione lineare a*u + b*v.

    Una combinazione lineare e' la somma di vettori, ognuno scalato dal
    proprio peso. E' l'operazione che una rete neurale fa in continuazione.
    Esempio: combinazione_lineare([1., 0.], [0., 1.], 2, 3) restituisce [2., 3.].
    """
    # TODO
    raise NotImplementedError


def dot_manuale(u, v):
    """Calcola il prodotto scalare tra u e v con un ciclo for esplicito.

    Non usare torch.dot ne' l'operatore @. Il punto dell'esercizio e'
    scrivere a mano il ciclo: moltiplica posizione per posizione e somma.
    Restituisci un float Python (usa .item() alla fine).
    Esempio: dot_manuale(tensor([2., 1.]), tensor([1., 3.])) restituisce 5.0.
    """
    # TODO: accumulate the products in a loop, then return a Python float
    raise NotImplementedError


def predici_prezzo(x, w, b):
    """Predici il prezzo di una casa con un modello lineare.

    x e' il vettore della casa (metri quadri, stanze), w il vettore dei
    pesi, b il bias (un numero). La predizione e' il prodotto scalare
    tra x e w, piu' b. Restituisci un tensor scalare (va bene il
    risultato diretto delle operazioni torch, senza .item()).
    Esempio: predici_prezzo([50., 2.], [2., 10.], 20.) restituisce 140.
    """
    # TODO: this time torch.dot is allowed
    raise NotImplementedError
