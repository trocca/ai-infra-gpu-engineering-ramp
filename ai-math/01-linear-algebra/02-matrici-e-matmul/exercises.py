"""Esercizi sulla lezione 02: matrici e matmul.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def estrai_diagonale(A):
    """Restituisci la diagonale della matrice quadrata A come vettore.

    La diagonale sono gli elementi A[0,0], A[1,1], A[2,2] e cosi' via.
    Non usare torch.diag: scrivilo con un ciclo, oppure con l'indexing.
    Esempio: per [[1., 2.], [3., 4.]] restituisce tensor([1., 4.]).
    """
    # TODO
    raise NotImplementedError


def trasposta_manuale(A):
    """Restituisci la trasposta di A senza usare A.T ne' torch.transpose.

    Costruisci una nuova matrice dove riga e colonna sono scambiate:
    il risultato alla posizione [j, i] vale A[i, j].
    Suggerimento: crea prima una matrice di zeri della forma giusta.
    """
    # TODO: build a zeros matrix of shape (cols, rows), then fill it
    raise NotImplementedError


def predizioni_case(X, w, b):
    """Calcola le predizioni per tutte le case in un colpo solo.

    X e' la matrice del dataset (una riga per casa), w il vettore dei
    pesi, b il bias. Restituisci il vettore delle predizioni usando una
    sola matmul (operatore @), senza cicli for.
    """
    # TODO
    raise NotImplementedError


def matrice_identita(n):
    """Costruisci la matrice identita' n x n senza usare torch.eye.

    Deve avere 1.0 sulla diagonale e 0.0 ovunque altro.
    Suggerimento: parti da torch.zeros(n, n) e riempi la diagonale.
    """
    # TODO
    raise NotImplementedError


def matmul_manuale(A, B):
    """Moltiplica A per B con tre cicli for espliciti.

    Non usare @, torch.matmul, torch.mm e nemmeno torch.dot.
    Regola: il risultato C ha una riga per ogni riga di A e una colonna
    per ogni colonna di B. La cella C[i, j] e' la somma dei prodotti
    A[i, k] * B[k, j] per ogni k.
    Funziona anche con matrici rettangolari: (2x3) @ (3x4) da' (2x4).
    """
    # TODO: three nested loops over i, j, k
    raise NotImplementedError
