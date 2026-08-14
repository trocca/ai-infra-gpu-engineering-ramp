"""Esercizi sulla lezione 01: variabili casuali e distribuzioni.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def valore_atteso(valori, probabilita):
    """Calcola il valore atteso E[X] di una variabile casuale discreta.

    valori e probabilita sono tensor della stessa lunghezza. Il valore
    atteso e' la media pesata: somma di valore * probabilita.
    Restituisci un tensor scalare.
    Esempio: per valori [0., 1.] e probabilita [0.3, 0.7] restituisce 0.7.
    """
    # TODO
    raise NotImplementedError


def varianza(valori, probabilita):
    """Calcola la varianza di una variabile casuale discreta.

    La varianza e' il valore atteso dello scarto al quadrato:
    E[(X - mu)^2], dove mu e' il valore atteso di X.
    Passi: calcola mu (riusa valore_atteso), poi la media pesata di
    (valori - mu)^2. Restituisci un tensor scalare.
    Esempio: Bernoulli(0.7) ha varianza 0.7 * 0.3 = 0.21.
    """
    # TODO
    raise NotImplementedError


def simula_moneta(p, n, seed):
    """Simula n lanci di una moneta truccata Bernoulli(p).

    Fissa il seed con torch.manual_seed(seed) PRIMA di campionare, cosi'
    il risultato e' riproducibile. Usa torch.distributions.Bernoulli e
    il suo metodo .sample((n,)). Restituisci il tensor dei lanci (0 e 1).
    """
    # TODO
    raise NotImplementedError


def probabilita_empirica(campioni, valore):
    """Stima la probabilita' di un valore dalla frequenza nei campioni.

    Restituisci la frazione di elementi di campioni uguali a valore,
    come tensor scalare. Suggerimento: il confronto == produce un tensor
    di True e False, e .float().mean() lo trasforma in frequenza.
    """
    # TODO
    raise NotImplementedError


def entro_k_sigma(campioni, mu, sigma, k):
    """Calcola la frazione di campioni dentro mu piu' o meno k sigma.

    Conta quanti campioni cadono nell'intervallo aperto
    (mu - k*sigma, mu + k*sigma) e dividi per il totale.
    Restituisci un tensor scalare. Per una gaussiana con k=2 il
    risultato deve venire vicino a 0.95.
    """
    # TODO
    raise NotImplementedError
