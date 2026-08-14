# Variabili casuali e distribuzioni

## L'intuizione

Una variabile casuale è un numero prodotto da un processo che non controlli del tutto. Il lancio di un dado. Il tempo di risposta di un server sotto carico. Il rumore in una misura. Non sai quale valore uscirà la prossima volta, ma il processo non è caos puro: alcuni valori escono più spesso di altri.

La **distribuzione** è la carta d'identità del processo: l'elenco dei valori possibili con la probabilità di ciascuno. Conoscere la distribuzione non ti dice il prossimo valore, ma ti dice tutto sul comportamento a lungo termine.

## L'idea formale, in parole semplici

Una variabile casuale discreta ha un elenco finito di valori possibili, ognuno con la sua probabilità. Le probabilità sono numeri tra 0 e 1 e sommano a 1. Una variabile continua (come il rumore) può assumere qualunque valore in un intervallo, e la distribuzione diventa una curva di densità.

Tre distribuzioni che incontrerai ovunque:

* **Bernoulli(p)**: vale 1 con probabilità p, 0 altrimenti. Una moneta, un click, un bit.
* **Uniforme**: tutti i valori di un intervallo sono ugualmente probabili. Il dado onesto.
* **Normale (gaussiana)**: la campana. Descrive il rumore di misura e mille altre cose. Ha due parametri: μ (mu, il centro) e σ (sigma, la larghezza).

Il **valore atteso** E[X] (si legge "valore atteso di X") è la media pesata dei valori, ognuno pesato con la sua probabilità. È il numero attorno a cui si stabilizza la media di tante ripetizioni. La **varianza** misura quanto i valori ballano attorno al valore atteso.

Fatto chiave, quasi magico ma verificabile al computer: la media di tanti campioni converge al valore atteso. Si chiama legge dei grandi numeri, e la vedrai succedere davanti agli occhi nello script.

## Esempio numerico a mano

Dado onesto, valori 1 fino a 6, ognuno con probabilità 1/6:

    E[X] = 1*(1/6) + 2*(1/6) + 3*(1/6) + 4*(1/6) + 5*(1/6) + 6*(1/6)
         = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 21 / 6 = 3.5

Nota: 3.5 non è nemmeno un valore possibile del dado. Il valore atteso non è "il valore tipico", è il baricentro.

Moneta truccata Bernoulli con p = 0.7:

    E[X] = 1 * 0.7 + 0 * 0.3 = 0.7
    varianza = p * (1 - p) = 0.7 * 0.3 = 0.21

## Riferimenti

* Mathematics for Machine Learning: capitolo 6, sezioni 6.1 e 6.2 per probabilità e distribuzioni, sezione 6.4 per media e varianza, sezione 6.5 per la gaussiana.
* Blitzstein, Hwang, Introduction to Probability: capitolo 3 (random variables), capitolo 4 (expectation), capitolo 5 (continuous random variables).

## E adesso

Esegui `python lesson.py`. Simula dadi, monete truccate e gaussiane con `torch.distributions`, verifica che le medie empiriche convergono ai valori attesi calcolati a mano, e salva in `figures/` la legge dei grandi numeri in azione e l'istogramma della campana.
