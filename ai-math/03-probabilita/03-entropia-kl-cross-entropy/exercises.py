"""Esercizi sulla lezione 03: entropia, KL e cross entropy.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def sorpresa(p):
    """Calcola la sorpresa di un evento di probabilita' p: -log p.

    p e' un float o un tensor scalare. Usa il log naturale (torch.log).
    Restituisci un tensor scalare.
    """
    # TODO
    raise NotImplementedError


def entropia(p):
    """Calcola l'entropia H(P) di una distribuzione discreta.

    p e' un tensor di probabilita' (positive, somma 1, niente zeri).
    Formula: somma di p * (-log p). Restituisci un tensor scalare.
    Esempio: entropia([0.5, 0.5]) restituisce log 2, circa 0.693.
    """
    # TODO
    raise NotImplementedError


def kl_divergence(p, q):
    """Calcola KL(P || Q) tra due distribuzioni discrete.

    Formula: somma di p * (log p - log q). Vale 0 solo se p e q sono
    identiche, altrimenti e' positiva. Restituisci un tensor scalare.
    """
    # TODO
    raise NotImplementedError


def softmax_a_mano(logits):
    """Trasforma un vettore di punteggi grezzi in probabilita'.

    Versione numericamente stabile: sottrai il massimo dai logits
    prima dell'esponenziale (non cambia il risultato, evita overflow),
    poi exp e normalizza dividendo per la somma.
    Non usare F.softmax. Restituisci un tensor come i logits.
    """
    # TODO
    raise NotImplementedError


def cross_entropy_a_mano(logits, classe_vera):
    """Calcola la cross entropy loss per un singolo esempio.

    logits e' il vettore dei punteggi del modello, classe_vera un
    intero. La ricetta completa: softmax dei logits (riusa la tua
    softmax_a_mano), poi meno log della probabilita' della classe vera.
    Non usare F.cross_entropy: il test la usa per controllarti.
    Restituisci un tensor scalare.
    """
    # TODO
    raise NotImplementedError
