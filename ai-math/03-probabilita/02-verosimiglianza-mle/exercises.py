"""Esercizi sulla lezione 02: verosimiglianza e MLE.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def likelihood_moneta(p, lanci):
    """Calcola la likelihood della sequenza di lanci per il parametro p.

    lanci e' un tensor di 0 e 1 (1 = testa). Se k e' il numero di teste
    e n il totale, la likelihood e' p^k * (1-p)^(n-k).
    p puo' essere un float o un tensor. Restituisci un tensor scalare.
    """
    # TODO
    raise NotImplementedError


def log_likelihood_moneta(p, lanci):
    """Calcola il logaritmo della likelihood della moneta.

    Usa direttamente la formula in forma di somma:
    k * log(p) + (n - k) * log(1 - p).
    Non calcolare prima la likelihood e poi il log: il punto della
    forma logaritmica e' evitare i numeri piccolissimi.
    """
    # TODO
    raise NotImplementedError


def mle_moneta(lanci):
    """Restituisci la stima di massima verosimiglianza per la moneta.

    Per la Bernoulli la risposta ha una formula chiusa: la frequenza
    osservata, teste diviso lanci totali. Restituisci un tensor scalare.
    """
    # TODO
    raise NotImplementedError


def mle_su_griglia(lanci, griglia):
    """Trova la MLE provando tutti i valori di p nella griglia.

    griglia e' un tensor di candidati p. Calcola la log likelihood per
    ogni candidato (riusa log_likelihood_moneta) e restituisci il
    candidato con il valore piu' alto.
    Il risultato deve coincidere (quasi) con mle_moneta: due strade,
    stessa risposta.
    """
    # TODO
    raise NotImplementedError


def nll_gaussiana(mu, dati, sigma):
    """Calcola la negative log likelihood di Normal(mu, sigma) sui dati.

    Formula (costanti incluse, cosi' il test e' preciso):
    nll = n * log(sigma * sqrt(2*pi)) + sum((dati - mu)^2) / (2*sigma^2)
    dove n e' il numero di dati. Usa torch.log, torch.sqrt e
    torch.tensor(torch.pi) dove serve. Restituisci un tensor scalare.
    Minimizzare questa funzione rispetto a mu equivale a minimizzare la
    somma dei quadrati: ecco da dove viene la MSE.
    """
    # TODO
    raise NotImplementedError
