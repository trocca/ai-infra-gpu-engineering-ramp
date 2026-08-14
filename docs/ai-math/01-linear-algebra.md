---
title: "01 · Algebra lineare"
parent: "AI Math"
nav_order: 1
permalink: /docs/ai-math/01-linear-algebra/
---

# Modulo 01 · Algebra lineare

L'algebra lineare è il linguaggio dei dati. Un dataset è una matrice. Un esempio è
un vettore. Una rete neurale, sotto il cofano, è quasi solo moltiplicazione di
matrici. Questo modulo costruisce quel linguaggio pezzo per pezzo.

![Vettori come frecce nel piano: somma e prodotto scalare visti geometricamente](figures/vettori.png)

## Lezioni

| Lezione | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| [01 · Vettori](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/01-linear-algebra/01-vettori) | Vettori, somma, prodotto scalare | Descrivere un dato come lista di numeri e fare una predizione con un prodotto scalare |
| [02 · Matrici e matmul](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/01-linear-algebra/02-matrici-e-matmul) | Matrici, trasposta, matmul | Fare predizioni su tutto il dataset in un colpo solo con `X @ w` |
| [03 · Norme e distanze](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/01-linear-algebra/03-norme-e-distanze) | Norme L1, L2, L∞, distanze, coseno | Misurare quanto due dati sono simili, e capire perché serve normalizzare |
| [04 · Autovalori e SVD](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/01-linear-algebra/04-autovalori-svd) | Autovalori, autovettori, SVD | Trovare le direzioni importanti di una matrice e comprimerla |

Le lezioni vanno fatte in ordine: ognuna usa i concetti della precedente.
Ogni lezione si chiude con `pytest`: quando i test passano, la lezione è fatta.

Le "palle unitarie" delle norme L1, L2 e L∞ — la stessa distanza 1 dal centro,
tre geometrie diverse (figura generata da `03-norme-e-distanze/lesson.py`):

![Palle unitarie delle norme L1, L2 e L-infinito](figures/unit_balls.png)

## Riferimenti ai libri

- **Mathematics for Machine Learning**: capitolo 2 (Linear Algebra), sezioni 2.1–2.4
  per le lezioni 01 e 02; capitolo 3 (Analytic Geometry), sezioni 3.1–3.4 per la
  lezione 03; capitolo 4 (Matrix Decompositions) per la lezione 04.

## Il ponte verso il resto del percorso

La matmul `X @ w` che impari qui è la stessa operazione che nel
[C++ ↔ CUDA track](../../track/06-matmul-tiling/) ottimizzerai con il tiling in
shared memory: prima capisci *cosa* calcola, poi *come* calcolarla in fretta.
