"""Esercizi sulla lezione 03: norme e distanze.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def norma_l1(v):
    """Calcola la norma L1 di v: somma dei valori assoluti.

    Non usare torch.linalg.norm: costruiscila con torch.abs e sum.
    Restituisci un tensor scalare.
    Esempio: norma_l1(tensor([3., -4.])) restituisce 7.
    """
    # TODO
    raise NotImplementedError


def norma_l2(v):
    """Calcola la norma L2 di v: radice della somma dei quadrati.

    Non usare torch.linalg.norm: costruiscila con le operazioni base
    (moltiplicazione, sum, torch.sqrt). Restituisci un tensor scalare.
    Esempio: norma_l2(tensor([3., 4.])) restituisce 5.
    """
    # TODO
    raise NotImplementedError


def norma_linf(v):
    """Calcola la norma L infinito di v: il massimo dei valori assoluti.

    Non usare torch.linalg.norm. Restituisci un tensor scalare.
    Esempio: norma_linf(tensor([3., -4.])) restituisce 4.
    """
    # TODO
    raise NotImplementedError


def distanza_euclidea(u, v):
    """Calcola la distanza L2 tra u e v.

    Ricorda la regola: prima la differenza, poi la norma della
    differenza. Puoi riusare la tua norma_l2.
    Esempio: distanza_euclidea([1., 1.], [4., 5.]) restituisce 5.
    """
    # TODO
    raise NotImplementedError


def normalizza_colonne(X):
    """Standardizza ogni colonna di X: sottrai la media, dividi per la std.

    X e' una matrice (righe = esempi, colonne = feature). Ogni colonna
    del risultato deve avere media 0 e deviazione standard 1.
    Suggerimento: X.mean(dim=0) e X.std(dim=0) lavorano per colonna.
    """
    # TODO
    raise NotImplementedError


def cosine_sim(u, v):
    """Calcola la similarita' coseno tra u e v.

    Formula: prodotto scalare diviso per il prodotto delle due norme L2.
    Non usare F.cosine_similarity: costruiscila con torch.dot e la tua
    norma_l2. Restituisci un tensor scalare.
    Esempio: cosine_sim([1., 0.], [0., 1.]) restituisce 0.
    """
    # TODO
    raise NotImplementedError
