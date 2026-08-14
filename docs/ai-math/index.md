---
title: "AI Math"
nav_order: 3
has_children: true
permalink: /docs/ai-math/
---

# AI Math — la matematica dietro le reti neurali
{: .fs-8 }

Percorso extra-curriculare, ma obbligatorio per chi vuole andare a fondo:
tutta la matematica che serve per capire davvero cosa fanno i modelli, verificata
riga per riga con codice PyTorch eseguibile.
{: .fs-5 .fw-300 }

[Modulo 1 · Algebra lineare](01-linear-algebra/){: .btn .btn-primary }
[Sorgenti su GitHub](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math){: .btn }

---

## L'idea di fondo

> Se puoi mettere un breakpoint su un numero, quel numero non ti fa più paura.

Niente dimostrazioni astratte. Ogni concetto arriva in coppia: una spiegazione
semplice in italiano e uno script PyTorch che puoi eseguire riga per riga, anche
dentro un debugger. Il percorso è pensato per chi viene dal mondo del software
(debugging, sistemi, codice) e vuole ricostruire le basi partendo dalla matematica
delle superiori. Ogni simbolo viene spiegato a parole la prima volta che appare.

**Ogni verifica passa dal codice.** Non ci sono quiz: ogni lezione ha esercizi
`# TODO` da completare e una suite `pytest` che decide quando la lezione è fatta.
E i grafici non sono decorazione: ogni figura che vedi in queste pagine è generata
dagli script delle lezioni, e puoi rigenerarla (e modificarla) tu stesso.

![La superficie di loss come una valle: il gradient descent è una pallina che rotola verso il fondo](figures/loss_landscape.png)

## Come si incastra con il resto del sito

Il [C++ ↔ CUDA track](../track/) risponde a *come* le macchine calcolano in fretta.
Questo percorso risponde a *cosa* stanno calcolando e *perché* funziona. I due si
rinforzano: quando nel track incontri una matmul tiled, qui scopri perché le reti
neurali sono quasi solo matmul; quando il capitolo
[How Machines Learn from Data](../how-machines-learn/) ti dice che "il modello scende
lungo il gradiente", qui scrivi quel gradiente a mano e lo verifichi al decimale.

Percorso consigliato:

- **In parallelo al track**, un modulo alla volta: 10–12 settimane a 4–5 ore a settimana.
- **Come immersione dedicata** prima dei moduli avanzati del track (reduction, matmul
  tiling), se i simboli ∇, Σ e le Jacobiane ti frenano nella lettura dei paper.

## La mappa del percorso

| Modulo | Argomento | Lezioni | Tempo |
|---|---|---|---|
| [01 · Algebra lineare](01-linear-algebra/) | Vettori, matrici, matmul, norme, autovalori, SVD | 4 | ~2 settimane |
| [02 · Calcolo](02-calcolo/) | Derivate, gradienti, chain rule, Jacobiane | 4 | ~2 settimane |
| [03 · Probabilità](03-probabilita/) | Variabili casuali, verosimiglianza, entropia, KL | 3 | ~1,5 settimane |
| [04 · Ottimizzazione](04-ottimizzazione/) | Gradient descent, SGD, momentum, Adam | 4 | ~2 settimane |
| [05 · Reti neurali](05-reti-neurali/) | Da regressione lineare ad attention, tutto da zero | 5 | ~3 settimane |

I moduli vanno in ordine: ognuno usa i concetti del precedente. Un piccolo dataset
di 5 case (metri quadri, stanze, prezzo) ritorna in ogni modulo sotto una luce
diversa — prima come matrice, poi come loss, poi come likelihood, infine come primo
modello addestrato.

## Il ciclo di lavoro di ogni lezione

Ogni cartella di lezione contiene sempre gli stessi cinque file:

1. **Leggi** `SPIEGAZIONE.md` — intuizione, definizione in parole semplici, esempio numerico fatto a mano.
2. **Esegui** `python lesson.py` — lo script rifà gli stessi passi stampando ogni valore intermedio. Ancora meglio: aprilo nel debugger e mettici breakpoint.
3. **Completa** le funzioni `# TODO` in `exercises.py`.
4. **Verifica** con `pytest` dalla cartella della lezione: quando i test passano, la lezione è fatta.
5. **Confronta** con `solutions.py` — solo dopo aver provato sul serio.

Le lezioni con grafici li salvano come PNG in `figures/`: non serve un display.

## Setup

Serve Python 3.10+. Dalla cartella [`ai-math/`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math):

```bash
python -m venv .venv
.venv\Scripts\activate      # su Windows
pip install -r requirements.txt
```

Per questo percorso basta PyTorch CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Testi di riferimento

Il percorso è autosufficiente, ma ogni modulo indica i capitoli di approfondimento in:

- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong) — [mml-book.github.io](https://mml-book.github.io/), il testo principale.
- **Understanding Deep Learning** (Prince) — [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/), per il modulo 05.
- **The Matrix Calculus You Need For Deep Learning** (Parr, Howard) — [arxiv.org/abs/1802.01528](https://arxiv.org/abs/1802.01528), per il modulo 02.
