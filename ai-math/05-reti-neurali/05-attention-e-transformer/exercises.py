"""Esercizi sulla lezione 05: attention e transformer.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import math

import torch


def softmax_per_righe(M):
    """Softmax applicata a ogni RIGA della matrice M.

    Versione stabile: sottrai a ogni riga il suo massimo prima di
    esponenziare. Suggerimento: M.max(dim=-1, keepdim=True).values e
    somme con dim=-1, keepdim=True, cosi' le forme si allineano.
    Non usare F.softmax. Ogni riga del risultato deve sommare a 1.
    """
    # TODO
    raise NotImplementedError


def punteggi_attenzione(Q, K):
    """La matrice dei punteggi di affinita', scalata.

    Formula: Q @ K trasposta, diviso la radice quadrata della
    dimensione dei vettori (l'ultima dimensione di Q).
    Restituisci la matrice n x n dei punteggi.
    """
    # TODO
    raise NotImplementedError


def attenzione(Q, K, V):
    """Scaled dot product attention completa.

    I tre passi: punteggi (riusa punteggi_attenzione), softmax per
    righe (riusa softmax_per_righe), media pesata dei value (una
    matmul con V). Il test confronta con F.scaled_dot_product_attention.
    """
    # TODO
    raise NotImplementedError


def maschera_causale(n):
    """Costruisci la maschera additiva causale n x n.

    Deve valere 0 dove guardare e' permesso (diagonale compresa) e
    meno infinito (float('-inf')) sopra la diagonale, dove sta il
    futuro. Sommata ai punteggi prima della softmax, spegne il futuro.
    Suggerimento: torch.triu con diagonal=1 seleziona il futuro.
    """
    # TODO
    raise NotImplementedError


def attenzione_causale(Q, K, V):
    """Attention con maschera causale: nessuno sguardo al futuro.

    Come attenzione, ma somma la tua maschera_causale ai punteggi
    prima della softmax. Il test confronta con
    F.scaled_dot_product_attention(is_causal=True).
    """
    # TODO
    raise NotImplementedError
