---
title: "04 · Ottimizzazione"
parent: "AI Math"
nav_order: 4
permalink: /docs/ai-math/04-ottimizzazione/
---

# Modulo 04 · Ottimizzazione

Qui tutto quello che hai costruito si mette in moto. Sai misurare l'errore (la
loss), sai calcolare la pendenza (il gradiente). Ottimizzare vuol dire usare quelle
pendenze per scendere verso l'errore minimo, un passo alla volta. È il motore
dell'addestramento di qualunque rete neurale, da un modello a due pesi a un LLM.

La metafora che accompagna tutto il modulo: una pallina che rotola giù per una valle
nella nebbia. Non vede la valle intera, sente solo la pendenza sotto di sé. Eppure,
passo dopo passo, trova il fondo.

![La superficie di loss e le traiettorie degli optimizer che scendono verso il minimo](figures/loss_landscape.png)

## Lezioni

| Lezione | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| [01 · Gradient descent a mano](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/04-ottimizzazione/01-gradient-descent-a-mano) | Gradient descent, learning rate | Scrivere il ciclo di discesa a mano e scegliere il passo giusto |
| [02 · SGD e minibatch](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/04-ottimizzazione/02-sgd-minibatch) | SGD, minibatch, epoche | Addestrare su dati a pezzi, capendo il rumore che ne deriva |
| [03 · Momentum e Adam](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/04-ottimizzazione/03-momentum-adam) | Momentum, Adam | Implementare i due optimizer più usati e verificarli contro `torch.optim` |
| [04 · Loss landscape](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/04-ottimizzazione/04-loss-landscape) | Superfici di loss, traiettorie | Disegnare la valle intera e guardare la pallina scendere |

La lezione 03 è verifica pura in codice: il tuo Adam scritto a mano deve produrre,
passo per passo, gli stessi numeri di `torch.optim.Adam`.

![Confronto fra optimizer: gradient descent, momentum e Adam sulla stessa valle](figures/optimizers.png)

## Riferimenti ai libri

- **Mathematics for Machine Learning**: capitolo 7 (Continuous Optimization) —
  sezione 7.1 per gradient descent, momentum e SGD; sezione 7.3 per la convessità.
- **Understanding Deep Learning** (Prince): capitolo 6 (Fitting Models).
- **Convex Optimization** (Boyd, Vandenberghe): capitoli 2–3, solo per curiosità.

## Il ponte verso il resto del percorso

Il ciclo di training che scrivi qui a mano è lo stesso che il
[C++ ↔ CUDA track](../../track/) accelera: SGD sui minibatch è il motivo per cui
l'addestramento è fatto di matmul ripetute, e la
[reduction](../../track/04-reduction/) che ottimizzi lì è la somma che calcola la
loss qui.
