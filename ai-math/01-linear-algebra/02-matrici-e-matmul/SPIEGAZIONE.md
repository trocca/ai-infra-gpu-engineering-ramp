# Matrici e moltiplicazione tra matrici

## L'intuizione

Se un vettore è un record, una matrice è una tabella: righe e colonne, come un foglio di calcolo o l'output di una query. Il nostro dataset di 5 case diventa una matrice con 5 righe (le case) e 2 colonne (metri quadri, stanze). Si scrive che la matrice è 5x2, sempre prima le righe e poi le colonne.

La moltiplicazione tra matrici, in inglese matmul, sembra strana la prima volta. Ma è solo questo: tanti prodotti scalari fatti in blocco. Ogni cella del risultato è il prodotto scalare tra una riga della prima matrice e una colonna della seconda. Tutto qui. Se nella lezione 01 hai capito il prodotto scalare, la matmul è quel concetto in versione batch.

## L'idea formale, in parole semplici

Una matrice A di forma m x n ha m righe e n colonne. L'elemento alla riga i e colonna j si scrive A[i, j].

**Trasposta**: si scrive Aᵀ, con una T in alto che si legge "trasposta". Scambia righe e colonne: la riga 0 diventa la colonna 0, e così via. Una matrice 5x2 trasposta diventa 2x5.

**Moltiplicazione** (matmul, simbolo `@` in Python): per calcolare C = A @ B, la cella C[i, j] è il prodotto scalare tra la riga i di A e la colonna j di B. Regola delle forme: le colonne di A devono essere quante le righe di B. Una (m x n) per una (n x p) dà una (m x p). I due numeri interni devono combaciare, i due esterni danno la forma del risultato.

Perché è importante: una predizione lineare su una casa è un prodotto scalare. Con la matmul fai la predizione su tutte le case in un colpo solo: `X @ w`. Le GPU esistono praticamente per fare questo alla massima velocità.

## Esempio numerico a mano

Prendiamo due matrici 2x2:

    A = | 1  2 |      B = | 5  6 |
        | 3  4 |          | 7  8 |

Calcolo C = A @ B cella per cella. Riga di A, colonna di B, prodotto scalare:

    C[0,0] = riga 0 di A · colonna 0 di B = 1*5 + 2*7 = 19
    C[0,1] = riga 0 di A · colonna 1 di B = 1*6 + 2*8 = 22
    C[1,0] = riga 1 di A · colonna 0 di B = 3*5 + 4*7 = 43
    C[1,1] = riga 1 di A · colonna 1 di B = 3*6 + 4*8 = 50

    C = | 19  22 |
        | 43  50 |

Attenzione: A @ B e B @ A in generale danno risultati diversi. L'ordine conta.

## Riferimenti

* Mathematics for Machine Learning: capitolo 2, sezione 2.2 (matrices) e sezione 2.7 (linear mappings).
* Strang, Introduction to Linear Algebra: capitolo 2.
* MIT 18.06: lezione 3, dedicata proprio alla moltiplicazione tra matrici.

## E adesso

Esegui `python lesson.py`. Vedrai la matmul scritta a mano con tre cicli for, il confronto con `torch.matmul`, e una gara di velocità tra le due versioni. Poi le predizioni su tutte le case con una riga sola.
