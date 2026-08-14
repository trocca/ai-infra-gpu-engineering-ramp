# Norme e distanze

## L'intuizione

Quanto è "grande" un vettore? Dipende da come misuri. Immagina di dover andare da casa a un punto della città che sta 3 isolati a est e 4 a nord.

* In taxi, tra i palazzi, percorri 3 + 4 = 7 isolati. Questa è la **norma L1**, detta anche distanza di Manhattan.
* In volo, in linea d'aria, percorri 5 isolati (il teorema di Pitagora delle superiori). Questa è la **norma L2**, la lunghezza classica.
* Se ti interessa solo il tratto peggiore, il massimo tra 3 e 4 è 4. Questa è la **norma L infinito**.

Stesso vettore, tre misure diverse. Nessuna è "quella giusta": sono strumenti diversi. In machine learning le userai tutte: la L2 per le distanze e per la loss, la L1 per rendere i modelli più semplici, la L infinito per i casi peggiori.

## L'idea formale, in parole semplici

La norma di un vettore v si scrive ‖v‖, due barrette verticali per lato, e si legge "norma di v". È un numero che misura la lunghezza del vettore.

* **Norma L1**: somma dei valori assoluti di ogni elemento.
* **Norma L2**: radice quadrata della somma dei quadrati. Se non c'è pedice, ‖v‖ vuol dire quasi sempre L2.
* **Norma L infinito**: il massimo dei valori assoluti.

La **distanza** tra due vettori u e v è la norma della loro differenza: ‖u − v‖. Prima sottrai, poi misuri la lunghezza di quello che resta.

La **similarità coseno** misura invece l'angolo tra due vettori: prodotto scalare diviso per il prodotto delle due norme L2. Vale 1 se puntano nella stessa direzione, 0 se sono perpendicolari, −1 se opposti. Ignora la lunghezza, guarda solo la direzione. È la misura standard per confrontare gli embedding.

## Esempio numerico a mano

Prendiamo v = [3, 4].

    L1:  |3| + |4| = 7
    L2:  radice di (3*3 + 4*4) = radice di 25 = 5
    Linf: max(|3|, |4|) = 4

Distanza L2 tra u = [1, 1] e v = [4, 5]:

    u - v = [1 - 4, 1 - 5] = [-3, -4]
    distanza = radice di (9 + 16) = radice di 25 = 5

Una trappola pratica che vedrai nello script: se misuri la distanza tra due case usando i dati grezzi, i metri quadri (numeri grandi) schiacciano completamente le stanze (numeri piccoli). La distanza dice solo "chi ha più metri quadri". La soluzione è normalizzare le colonne prima di misurare. Questo problema tornerà identico quando addestreremo le reti.

## Riferimenti

* Mathematics for Machine Learning: capitolo 3, sezioni 3.1 (norms), 3.2 (inner products), 3.3 (lengths and distances) e 3.4 (angles and orthogonality).
* Strang, Introduction to Linear Algebra: capitolo 1, dove lunghezze e prodotto scalare vengono introdotti insieme.

## E adesso

Esegui `python lesson.py`. Calcola le tre norme a mano e con `torch.linalg.norm`, mostra la trappola delle scale sul dataset delle case, e salva in `figures/` un grafico con le "palle unitarie": la forma di tutti i punti a distanza 1 dall'origine, secondo ciascuna norma.
