# Modulo 03: probabilità

I dati veri sono rumorosi, i modelli veri sono incerti. La probabilità è il linguaggio per ragionare bene sull'incertezza. In questo modulo scoprirai anche un segreto del deep learning: quasi tutte le loss che si usano davvero (MSE, cross entropy) non sono inventate, sono conseguenze dirette di idee probabilistiche.

## Lezioni

| Cartella | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| `01-variabili-casuali-distribuzioni` | Variabili casuali, distribuzioni, valore atteso | Simulare processi casuali e prevedere il loro comportamento medio |
| `02-verosimiglianza-mle` | Likelihood, log likelihood, MLE | Stimare parametri dai dati massimizzando la verosimiglianza |
| `03-entropia-kl-cross-entropy` | Sorpresa, entropia, KL, cross entropy | Capire da dove viene davvero la loss di classificazione |

Le lezioni vanno fatte in ordine. La lezione 03 chiude il cerchio: la cross entropy che troverai in ogni rete neurale è l'incontro tra la likelihood della lezione 02 e l'entropia.

## Riferimenti ai libri

* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), il testo principale:
  * Capitolo 6, Probability and Distributions: sezioni 6.1 e 6.2 per la lezione 01, sezione 6.4 per valore atteso e varianza, sezione 6.5 per la gaussiana.
  * Capitolo 8, When Models Meet Data: sezione 8.3 per la maximum likelihood della lezione 02.
* **Introduction to Probability** (Blitzstein, Hwang), di supporto:
  * Capitolo 3 per le variabili casuali e le distribuzioni, capitolo 4 per il valore atteso, capitolo 5 per le variabili continue.
* **Understanding Deep Learning** (Prince), di supporto:
  * Capitolo 5, Loss Functions: il ponte tra probabilità e loss, usato nella lezione 03.

## Tempo stimato

Circa 1 settimana e mezza a 4 o 5 ore a settimana. La lezione 01 è leggera, le altre due meritano calma: sono i concetti che rendono leggibili i paper.

## Il filo conduttore

Il punteggio di errore cambia faccia: nella lezione 02 scoprirai che minimizzare la MSE del modello delle case equivale a massimizzare una verosimiglianza gaussiana. E nella lezione 03 vedrai che la loss di classificazione è la sorpresa media del modello davanti alla verità. Le loss non si inventano: si derivano.
