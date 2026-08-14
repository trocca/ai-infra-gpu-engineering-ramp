# Modulo 04: ottimizzazione

Qui tutto quello che hai costruito si mette in moto. Sai misurare l'errore (la loss), sai calcolare la pendenza (il gradiente). Ottimizzare vuol dire usare quelle pendenze per scendere verso l'errore minimo, un passo alla volta. È il motore dell'addestramento di qualunque rete neurale, da un modello a due pesi a un LLM.

La metafora che accompagna tutto il modulo: una pallina che rotola giù per una valle nella nebbia. Non vede la valle intera, sente solo la pendenza sotto di sé. Eppure, passo dopo passo, trova il fondo.

## Lezioni

| Cartella | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| `01-gradient-descent-a-mano` | Gradient descent, learning rate | Scrivere il ciclo di discesa a mano e scegliere il passo giusto |
| `02-sgd-minibatch` | SGD, minibatch, epoche | Addestrare su dati a pezzi, capendo il rumore che ne deriva |
| `03-momentum-adam` | Momentum, Adam | Implementare i due optimizer piu' usati e verificarli contro torch.optim |
| `04-loss-landscape` | Superfici di loss, traiettorie | Disegnare la valle intera e guardare la pallina scendere |

Le lezioni vanno fatte in ordine: ognuna aggiunge un pezzo al ciclo di training della precedente.

## Riferimenti ai libri

* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), il testo principale:
  * Capitolo 7, Continuous Optimization: sezione 7.1 per gradient descent, momentum e SGD, sezione 7.3 per l'idea di convessità.
* **Convex Optimization** (Boyd, Vandenberghe), di supporto e solo per curiosità:
  * Capitolo 2 (convex sets) e capitolo 3 (convex functions), per chi vuole vedere la teoria delle valli con un solo fondo.
* **Understanding Deep Learning** (Prince), di supporto:
  * Capitolo 6, Fitting Models: gradient descent, SGD, momentum e Adam raccontati con ottime figure.

## Tempo stimato

Circa 2 settimane a 4 o 5 ore a settimana. La lezione 03 è la più densa: implementare Adam a mano e vederlo coincidere con `torch.optim.Adam` ripaga ogni minuto speso.

## Il filo conduttore

Le 5 case tornano da protagoniste: il modello lineare che nel modulo 02 sapeva solo misurare la propria pendenza, qui impara davvero. Nella lezione 04 vedrai l'intera valle della sua loss e la traiettoria della discesa sopra di essa, nebbia tolta.
