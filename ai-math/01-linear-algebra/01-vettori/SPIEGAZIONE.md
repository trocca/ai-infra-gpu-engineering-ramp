# Vettori

## L'intuizione

Pensa a un record in un log strutturato: un evento con dentro campi numerici, sempre nello stesso ordine. Un vettore è esattamente questo: una lista ordinata di numeri. Niente di più.

Una casa del nostro dataset, per esempio, è descritta da due numeri: metri quadri e numero di stanze. La casa "50 metri quadri, 2 stanze" diventa il vettore `[50, 2]`. L'ordine conta: la posizione 0 è sempre i metri quadri, la posizione 1 è sempre le stanze. Come i campi di una struct.

Un vettore di 2 numeri si può anche disegnare: è una freccia che parte dall'origine e arriva al punto con quelle coordinate. Con 3 numeri è una freccia nello spazio. Con 100 numeri il disegno non si può più fare, ma le regole di calcolo restano identiche.

## L'idea formale, in parole semplici

Un vettore di dimensione n è una lista ordinata di n numeri. Si scrive **v** in grassetto, oppure v con una freccina sopra. Le operazioni fondamentali sono tre:

1. **Somma**: si sommano i numeri posizione per posizione. Serve che i due vettori abbiano la stessa dimensione.
2. **Moltiplicazione per uno scalare**: uno scalare è un numero singolo. Si moltiplica ogni elemento del vettore per quel numero. La freccia si allunga o si accorcia, la direzione non cambia.
3. **Prodotto scalare** (in inglese dot product, simbolo `·`, un puntino): si moltiplicano i numeri posizione per posizione e poi si somma tutto. Il risultato è un numero singolo, non un vettore.

Il prodotto scalare è l'operazione più importante di tutto il percorso. Una predizione lineare, il cuore di ogni rete neurale, è un prodotto scalare: dati per pesi, tutto sommato.

## Esempio numerico a mano

Prendiamo `u = [2, 1]` e `v = [1, 3]`.

Somma, posizione per posizione:

    u + v = [2 + 1, 1 + 3] = [3, 4]

Moltiplicazione per lo scalare 2:

    2 * u = [2 * 2, 2 * 1] = [4, 2]

Prodotto scalare, moltiplico e poi sommo:

    u · v = 2 * 1 + 1 * 3 = 2 + 3 = 5

Tutto qui. Se sai fare queste tre operazioni a mano su vettori piccoli, sai già leggere metà delle formule del deep learning.

## Riferimenti

* Mathematics for Machine Learning: capitolo 2, sezione 2.4 (vector spaces); il prodotto scalare è nel capitolo 3, sezione 3.2 (inner products).
* Strang, Introduction to Linear Algebra: capitolo 1.
* MIT 18.06: lezione 1.

## E adesso

Apri `lesson.py` ed eseguilo con `python lesson.py`. Rifà questi stessi calcoli in PyTorch, stampando ogni valore, e alla fine usa un prodotto scalare per fare la prima predizione sul prezzo di una casa.
