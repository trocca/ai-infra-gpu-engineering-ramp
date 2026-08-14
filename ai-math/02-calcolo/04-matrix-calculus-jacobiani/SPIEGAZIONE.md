# Matrix calculus e Jacobiane

## L'intuizione

Finora: input un numero, output un numero, derivata un numero. Poi: input un vettore, output un numero, derivata un vettore (il gradiente). Manca l'ultimo caso: input un vettore, output un vettore. Che forma ha la derivata?

Pensa a un sistema con 2 ingressi e 3 uscite. Ogni uscita può essere sensibile a ognuno degli ingressi. Quante sensibilità ci sono in tutto? 3 per 2, cioè 6. La derivata è quindi una tabella 3x2 di pendenze: una riga per uscita, una colonna per ingresso. Quella tabella si chiama **Jacobiana**.

Non c'è nessuna idea nuova qui. Solo contabilità: tante derivate parziali, impacchettate in una matrice con le forme giuste. Il matrix calculus è per il 90 per cento una questione di tenere in ordine le forme.

## L'idea formale, in parole semplici

Se f prende un vettore di n numeri e restituisce un vettore di m numeri, la sua Jacobiana J è una matrice m x n:

    J[i, j] = quanto l'uscita i sente l'ingresso j = la derivata parziale di f_i rispetto a x_j

Casi particolari che già conosci:

* m = 1, n = 1: la Jacobiana è 1x1, un numero. La derivata della lezione 01.
* m = 1, n qualsiasi: la Jacobiana è una riga sola. È il gradiente della lezione 02, sdraiato.

Il caso più bello: se la funzione è lineare, f(x) = W @ x, allora la Jacobiana è esattamente W. La matrice dei pesi È la tabella delle sensibilità. Per questo i layer lineari delle reti sono così comodi da derivare.

E la chain rule? Funziona ancora, ma i prodotti diventano prodotti tra matrici: la Jacobiana di una composizione è la matmul delle Jacobiane. Stessa regola della lezione 03, gli anelli ora sono matrici.

## Esempio numerico a mano

Prendiamo f(x, y) = [x², x*y], due ingressi e due uscite, nel punto (2, 3).

Quattro derivate parziali, una per cella:

    riga 0 (uscita x²):   df0/dx = 2x = 4     df0/dy = 0
    riga 1 (uscita x*y):  df1/dx = y  = 3     df1/dy = x = 2

    J(2, 3) = | 4  0 |
              | 3  2 |

Lettura per righe: la prima uscita sente solo x (velocità 4). La seconda uscita sente x con velocità 3 e y con velocità 2.

Regola pratica per non perdersi con le forme: la Jacobiana ha sempre forma (numero di uscite) x (numero di ingressi). Se il tuo calcolo produce una forma diversa, hai sbagliato qualcosa.

## Riferimenti

* Mathematics for Machine Learning: capitolo 5, sezione 5.3 (gradients of vector valued functions) e sezione 5.4 (gradients of matrices).
* Parr, Howard, The Matrix Calculus You Need For Deep Learning: la sezione sulla Jacobiana, che è il centro del paper.

## E adesso

Esegui `python lesson.py`. Costruisce la Jacobiana dell'esempio in tre modi (formula a mano, differenze finite colonna per colonna, `torch.autograd.functional.jacobian`), verifica che per un layer lineare la Jacobiana è W, e chiude con le forme dei gradienti della loss delle case.
