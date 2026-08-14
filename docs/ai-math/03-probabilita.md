---
title: "03 · Probabilità"
parent: "AI Math"
nav_order: 3
permalink: /docs/ai-math/03-probabilita/
---

# Modulo 03 · Probabilità

I dati veri sono rumorosi, i modelli veri sono incerti. La probabilità è il
linguaggio per ragionare bene sull'incertezza. In questo modulo scoprirai anche un
segreto del deep learning: quasi tutte le loss che si usano davvero (MSE, cross
entropy) non sono inventate, sono conseguenze dirette di idee probabilistiche.

![Entropia di una moneta al variare della probabilità di testa: massima incertezza a 0,5](figures/entropia_moneta.png)

## Lezioni

| Lezione | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| [01 · Variabili casuali e distribuzioni](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/03-probabilita/01-variabili-casuali-distribuzioni) | Variabili casuali, distribuzioni, valore atteso | Simulare processi casuali e prevedere il loro comportamento medio |
| [02 · Verosimiglianza e MLE](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/03-probabilita/02-verosimiglianza-mle) | Likelihood, log likelihood, MLE | Stimare parametri dai dati massimizzando la verosimiglianza |
| [03 · Entropia, KL, cross entropy](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/03-probabilita/03-entropia-kl-cross-entropy) | Sorpresa, entropia, KL, cross entropy | Capire da dove viene davvero la loss di classificazione |

La lezione 03 chiude il cerchio: la cross entropy che troverai in ogni rete neurale
è l'incontro tra la likelihood della lezione 02 e l'entropia.

## Riferimenti ai libri

- **Mathematics for Machine Learning**: capitolo 6 (Probability and Distributions),
  sezioni 6.1–6.2 per la lezione 01, 6.4 per valore atteso e varianza, 6.5 per la
  gaussiana; capitolo 8, sezione 8.3 per la maximum likelihood.
- **Introduction to Probability** (Blitzstein, Hwang): capitoli 3–5 di supporto.
- **Understanding Deep Learning** (Prince): capitolo 5 (Loss Functions), il ponte
  tra probabilità e loss.

## Il filo conduttore

Il punteggio di errore cambia faccia: nella lezione 02 scoprirai che minimizzare la
MSE del modello delle case equivale a massimizzare una verosimiglianza gaussiana. E
nella lezione 03 vedrai che la loss di classificazione è la sorpresa media del
modello davanti alla verità. Le loss non si inventano: si derivano.
