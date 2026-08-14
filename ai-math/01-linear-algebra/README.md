# Modulo 01: algebra lineare

L'algebra lineare è il linguaggio dei dati. Un dataset è una matrice. Un esempio è un vettore. Una rete neurale, sotto il cofano, è quasi solo moltiplicazione di matrici. Questo modulo costruisce quel linguaggio pezzo per pezzo.

## Lezioni

| Cartella | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| `01-vettori` | Vettori, somma, prodotto scalare | Descrivere un dato come lista di numeri e fare una predizione con un prodotto scalare |
| `02-matrici-e-matmul` | Matrici, trasposta, matmul | Fare predizioni su tutto il dataset in un colpo solo con `X @ w` |
| `03-norme-e-distanze` | Norme L1, L2, L infinito, distanze, coseno | Misurare quanto due dati sono simili, e capire perché serve normalizzare |
| `04-autovalori-svd` | Autovalori, autovettori, SVD | Trovare le direzioni importanti di una matrice e comprimerla |

Le lezioni vanno fatte in ordine: ognuna usa i concetti della precedente.

## Riferimenti ai libri

* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), il testo principale:
  * Capitolo 2, Linear Algebra: sezioni 2.1 fino a 2.4 per le lezioni 01 e 02.
  * Capitolo 3, Analytic Geometry: sezioni 3.1 fino a 3.4 per la lezione 03.
  * Capitolo 4, Matrix Decompositions: sezioni 4.2, 4.4 e 4.5 per la lezione 04.
* **Introduction to Linear Algebra** (Strang), di supporto:
  * Capitolo 1 per i vettori, capitolo 2 per le matrici, capitoli 6 e 7 per autovalori e SVD.
* **MIT 18.06** (video lezioni di Strang), di supporto:
  * Lezione 1 per iniziare, lezione 3 per la moltiplicazione tra matrici, lezione 21 per gli autovalori, lezione 29 per la SVD.

## Tempo stimato

Circa 2 settimane a un ritmo di 4 o 5 ore a settimana. Le prime due lezioni sono veloci. Le ultime due meritano più calma, in particolare la SVD.

## Il filo conduttore

In questo modulo compare per la prima volta il dataset delle 5 case: per ogni casa conosciamo metri quadri, numero di stanze e prezzo. Qui lo useremo come matrice. Nei moduli successivi ci calcoleremo sopra derivate, faremo gradient descent e ci addestreremo una rete neurale. Stesso dato, strumenti sempre più potenti.
