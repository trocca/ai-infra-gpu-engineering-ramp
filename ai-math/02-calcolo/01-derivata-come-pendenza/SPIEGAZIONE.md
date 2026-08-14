# La derivata come pendenza

## L'intuizione

Guarda una curva qualunque su un grafico. Adesso zooma su un punto. Zooma ancora. A un certo punto la curva, vista così da vicino, sembra un segmento dritto. La pendenza di quel segmento è la derivata della funzione in quel punto.

Pendenza vuol dire: se mi sposto un pochino a destra, di quanto sale (o scende) la funzione? Derivata 6 significa che, vicino a quel punto, spostarsi di 0.01 a destra fa salire la funzione di circa 0.06. Derivata negativa: la funzione scende. Derivata zero: sei su un punto piatto, un fondovalle o una cima.

Per un debugger è un concetto familiare: è l'analisi di sensibilità. Tocco questo input di un epsilon, quanto si muove l'output?

## L'idea formale, in parole semplici

La derivata di f nel punto x si scrive f'(x), che si legge "f primo di x", oppure df/dx, che si legge "de f su de x" e ricorda una frazione: variazione di f diviso variazione di x.

La ricetta pratica si chiama **differenza finita**: scegli un passo piccolo h, e calcoli

    pendenza approssimata = (f(x + h) - f(x)) / h

Più h è piccolo, più l'approssimazione è buona. La derivata vera è il valore a cui tende questa frazione quando h diventa piccolissimo.

Per le funzioni comuni esistono formule esatte. L'unica che ci serve subito: la derivata di x² è 2x. E una regola: la derivata di una somma è la somma delle derivate.

PyTorch calcola le derivate esatte da solo, con un meccanismo chiamato **autograd**. Chiami `loss.backward()` e ogni tensor coinvolto riceve la sua pendenza nel campo `.grad`. Nel modulo 05 apriremo il cofano di questo meccanismo.

## Esempio numerico a mano

Prendiamo f(x) = x² nel punto x = 3, con passo h = 0.01:

    f(3)    = 9
    f(3.01) = 9.0601
    pendenza approssimata = (9.0601 - 9) / 0.01 = 6.01

La formula esatta dice f'(x) = 2x, quindi f'(3) = 6. La differenza finita ci è andata vicinissimo: 6.01 contro 6. Con h ancora più piccolo l'errore si riduce.

Perché ci interessa: nel nostro dataset delle case, la loss (il punteggio di errore del modello) è una funzione del peso w. La derivata della loss rispetto a w ci dice se aumentare w peggiora o migliora l'errore, e di quanto. È l'informazione che serve per imparare.

## Riferimenti

* Mathematics for Machine Learning: capitolo 5, sezione 5.1 (differentiation of univariate functions).
* Parr, Howard, The Matrix Calculus You Need For Deep Learning: la sezione introduttiva, dove si parte proprio dalle derivate scalari.

## E adesso

Esegui `python lesson.py`. Calcola la derivata di x² in tre modi (differenza finita, formula esatta, autograd), verifica che coincidono, poi disegna la loss delle case in funzione del peso w: vedrai una parabola, con la tangente che indica la direzione della discesa.
