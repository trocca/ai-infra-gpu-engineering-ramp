---
title: "02 · Calcolo"
parent: "AI Math"
nav_order: 2
permalink: /docs/ai-math/02-calcolo/
---

# Modulo 02 · Calcolo

Il calcolo differenziale risponde a una sola domanda, ripetuta in mille forme: se
muovo questa manopola di un pelo, quanto cambia il risultato? Quella "sensibilità"
si chiama derivata. Le reti neurali imparano esattamente così: misurano quanto ogni
peso fa cambiare l'errore, e girano le manopole nella direzione giusta.

![Campo di gradienti: in ogni punto la freccia indica la direzione di salita più ripida](figures/campo_gradienti.png)

## Lezioni

| Lezione | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| [01 · Derivata come pendenza](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/02-calcolo/01-derivata-come-pendenza) | Derivata, differenze finite, autograd | Calcolare la pendenza di una funzione in tre modi diversi e verificare che coincidono |
| [02 · Derivate parziali e gradiente](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/02-calcolo/02-derivate-parziali-gradiente) | Derivate parziali, gradiente | Misurare la pendenza rispetto a ogni manopola, e leggere ∇ senza paura |
| [03 · Chain rule](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/02-calcolo/03-chain-rule) | Regola della catena | Derivare funzioni composte moltiplicando le pendenze anello per anello |
| [04 · Matrix calculus e Jacobiane](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/02-calcolo/04-matrix-calculus-jacobiani) | Jacobiane, gradienti di vettori | Gestire le forme delle derivate quando input e output sono vettori |

La lezione 03 è la più importante del modulo: se la chain rule ti è chiara,
backpropagation (modulo 05) sarà una formalità.

## Riferimenti ai libri

- **Mathematics for Machine Learning**: capitolo 5 (Vector Calculus) — sezione 5.1
  per la lezione 01, sezione 5.2 per le lezioni 02 e 03, sezione 5.3 per la lezione
  04, sezione 5.6 come anteprima di backpropagation.
- **The Matrix Calculus You Need For Deep Learning** (Parr, Howard): derivate
  parziali, chain rule e Jacobiana raccontate per chi fa deep learning.

## Il filo conduttore

Il dataset delle 5 case torna qui con un ruolo nuovo: la loss — il punteggio di
errore del modello — vista come funzione di un peso. Nella lezione 01 scoprirai che
è una parabola, e che la sua pendenza ti dice da che parte correggere il peso. È il
seme del gradient descent del [modulo 04](../04-ottimizzazione/).
