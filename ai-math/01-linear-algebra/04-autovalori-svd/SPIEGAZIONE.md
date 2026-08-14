# Autovalori, autovettori e SVD

## L'intuizione

Una matrice quadrata, moltiplicata per un vettore, lo trasforma: lo ruota, lo allunga, lo schiaccia. Prova a moltiplicare la stessa matrice per tanti vettori diversi: quasi tutti cambiano direzione.

Ma quasi ogni matrice ha delle direzioni speciali. Se ci metti un vettore lì, la matrice non lo ruota: lo allunga o lo accorcia soltanto, lasciandolo sulla sua retta. Quelle direzioni si chiamano **autovettori**. Il fattore di allungamento si chiama **autovalore**. È come trovare "l'asse naturale" di un sistema: la direzione lungo cui il comportamento diventa semplice.

La **SVD** (Singular Value Decomposition, decomposizione ai valori singolari) generalizza l'idea a qualunque matrice, anche rettangolare come il nostro dataset 5x2. Dice: ogni matrice, per quanto complicata, è la sequenza di tre operazioni semplici. Una rotazione, un allungamento lungo gli assi, un'altra rotazione. Sempre. E ti dice quali direzioni contano di più: è la base della compressione e della PCA.

## L'idea formale, in parole semplici

**Autovettori e autovalori**: v è un autovettore di A, con autovalore λ (lambda, una lettera greca che qui è solo un numero), se

    A @ v = λ * v

Applicare la matrice a v equivale a moltiplicarlo per un numero. Direzione invariata, solo scala.

**Decomposizione agli autovalori**: una matrice simmetrica (uguale alla sua trasposta) si può riscrivere come Q @ Λ @ Qᵀ, dove Q ha gli autovettori nelle colonne e Λ (lambda maiuscola) è diagonale con gli autovalori. In PyTorch la calcola `torch.linalg.eigh`.

**SVD**: qualunque matrice A si riscrive come U @ S @ Vᵀ. U e V contengono direzioni (ortogonali tra loro), S è diagonale e contiene i **valori singolari**, numeri positivi in ordine decrescente. Il primo valore singolare indica la direzione lungo cui la matrice "contiene più informazione". Tenendo solo i primi k valori ottieni la migliore approssimazione di rango k: stessa matrice, meno numeri, minima perdita.

## Esempio numerico a mano

Prendiamo la matrice simmetrica

    A = | 2  1 |
        | 1  2 |

Provo v = [1, 1]:

    A @ v = [2*1 + 1*1, 1*1 + 2*1] = [3, 3] = 3 * [1, 1]

Direzione identica, lunghezza per 3. Quindi [1, 1] è un autovettore con autovalore 3.

Provo v = [1, -1]:

    A @ v = [2*1 + 1*(-1), 1*1 + 2*(-1)] = [1, -1] = 1 * [1, -1]

Autovettore con autovalore 1. Provo invece v = [1, 0]:

    A @ v = [2, 1]

Direzione cambiata: non è un autovettore. Le direzioni speciali sono davvero speciali.

## Riferimenti

* Mathematics for Machine Learning: capitolo 4, sezioni 4.2 (eigenvalues and eigenvectors), 4.4 (eigendecomposition and diagonalization) e 4.5 (singular value decomposition).
* Strang, Introduction to Linear Algebra: capitolo 6 per gli autovalori, capitolo 7 per la SVD.
* MIT 18.06: lezione 21 per gli autovalori, lezione 29 per la SVD.

## E adesso

Esegui `python lesson.py`. Verifica a mano gli autovettori qui sopra, ricostruisce A dai suoi pezzi, poi fa la SVD del dataset delle case e mostra quanto bene si comprime a rango 1.
