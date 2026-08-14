"""Esercizi sulla lezione 04: autovalori, autovettori e SVD.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def applica_matrice(A, v):
    """Applica la matrice A al vettore v, cioe' calcola A @ v.

    Riscaldamento: e' una riga sola, ma e' l'operazione al centro di
    tutta la lezione.
    """
    # TODO
    raise NotImplementedError


def e_autovettore(A, v, lam):
    """Verifica se v e' un autovettore di A con autovalore lam.

    Deve valere A @ v = lam * v. Confronta i due lati con
    torch.allclose (usa atol=1e-5) e restituisci True o False.
    """
    # TODO
    raise NotImplementedError


def quoziente_rayleigh(A, v):
    """Calcola il quoziente di Rayleigh: (v . (A @ v)) / (v . v).

    Se v e' un autovettore, questo numero e' esattamente il suo
    autovalore. Se non lo e', e' una stima. Restituisci un tensor
    scalare. Usa torch.dot e l'operatore @.
    """
    # TODO
    raise NotImplementedError


def ricostruisci_da_eigh(eigvals, Q):
    """Ricostruisci la matrice originale dai pezzi di torch.linalg.eigh.

    eigvals e' il vettore degli autovalori, Q la matrice con gli
    autovettori nelle colonne. La formula e' Q @ diag(eigvals) @ Q.T.
    Suggerimento: torch.diag trasforma un vettore in matrice diagonale.
    """
    # TODO
    raise NotImplementedError


def ricostruisci_da_svd(U, S, Vh):
    """Ricostruisci la matrice originale dai tre pezzi della SVD.

    U, S, Vh sono l'output di torch.linalg.svd(X, full_matrices=False).
    La formula e' U @ diag(S) @ Vh.
    """
    # TODO
    raise NotImplementedError


def approssima_rango_k(X, k):
    """Restituisci la migliore approssimazione di rango k di X via SVD.

    Passi: calcola la SVD compatta di X, poi tieni solo le prime k
    colonne di U, i primi k valori di S e le prime k righe di Vh, e
    ricombina con la stessa formula della ricostruzione.
    Con k uguale al rango pieno deve restituire X (a meno di errori di
    arrotondamento).
    """
    # TODO: torch.linalg.svd(X, full_matrices=False), then slice and rebuild
    raise NotImplementedError
