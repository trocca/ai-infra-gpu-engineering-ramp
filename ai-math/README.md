# math-for-ai

Un percorso personale per ricostruire la matematica che sta dietro le reti neurali moderne. Niente dimostrazioni astratte. Ogni concetto arriva in coppia: una spiegazione semplice in italiano e uno script PyTorch che puoi eseguire riga per riga, anche dentro un debugger.

L'idea di fondo: se puoi mettere un breakpoint su un numero, quel numero non ti fa più paura.

## A chi serve

A chi viene dal mondo del software (debugging, sistemi, codice) e vuole ripartire dalle basi della matematica. Non serve nessun prerequisito oltre la matematica delle superiori. Ogni simbolo viene spiegato a parole la prima volta che appare.

## La mappa del percorso

| Modulo | Argomento | Lezioni |
|---|---|---|
| `01-linear-algebra` | Vettori, matrici, matmul, norme, autovalori, SVD | 4 |
| `02-calcolo` | Derivate, gradienti, chain rule, Jacobiani | 4 |
| `03-probabilita` | Variabili casuali, verosimiglianza, entropia, KL | 3 |
| `04-ottimizzazione` | Gradient descent, SGD, momentum, Adam | 4 |
| `05-reti-neurali` | Da regressione lineare ad attention, tutto da zero | 5 |

Tutti i moduli sono completi: 20 lezioni, ognuna con spiegazione, script eseguibile, esercizi e test. Tempo stimato per l'intero percorso: 10 o 12 settimane a 4 o 5 ore a settimana.

I moduli vanno in ordine. Ognuno usa i concetti del precedente. Alcuni esempi ritornano lungo tutto il percorso, in particolare un piccolo dataset di 5 case (metri quadri, stanze, prezzo) che rivedrai in ogni modulo sotto una luce diversa.

## Come si usa una lezione

Ogni cartella di lezione contiene sempre gli stessi cinque file. Il flusso di lavoro è questo:

1. Leggi `SPIEGAZIONE.md`. Contiene l'intuizione, la definizione in parole semplici e un esempio numerico fatto a mano.
2. Esegui `python lesson.py`. Lo script rifà gli stessi passi della spiegazione, stampando ogni valore intermedio. Ancora meglio: aprilo nel debugger e mettici dei breakpoint.
3. Apri `exercises.py` e completa le funzioni marcate con `# TODO`. Le docstring spiegano cosa deve fare ogni funzione.
4. Esegui `pytest` dalla cartella della lezione. Quando tutti i test passano, la lezione è fatta.
5. `solutions.py` contiene le soluzioni complete. Guardalo solo dopo aver provato sul serio, oppure per confrontare il tuo approccio con quello proposto.

Alcune lezioni salvano grafici nella sottocartella `figures/`. Non serve un display: i grafici vengono scritti su file PNG.

## Setup

Serve Python 3.10 o superiore.

```
python -m venv .venv
.venv\Scripts\activate      # su Windows
pip install -r requirements.txt
```

Nota su PyTorch: per questo percorso basta la versione CPU. Se vuoi quella più leggera:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verifica rapida che tutto funzioni:

```
python -c "import torch; print(torch.__version__)"
```

## Testi di riferimento

Il percorso è ancorato a testi gratuiti e legali. Ogni modulo cita capitolo e sezione precisi.

* Deisenroth, Faisal, Ong, **Mathematics for Machine Learning**: la spina dorsale di tutto il percorso. Gratis su [mml-book.github.io](https://mml-book.github.io).
* Strang, **Introduction to Linear Algebra** e le lezioni video **MIT 18.06**: supporto per il modulo 01. Le lezioni sono su [MIT OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
* Parr, Howard, **The Matrix Calculus You Need For Deep Learning**: supporto per il modulo 02. Gratis su [arxiv.org/abs/1802.01528](https://arxiv.org/abs/1802.01528).
* Blitzstein, Hwang, **Introduction to Probability** (Harvard Stat 110): supporto per il modulo 03. Gratis su [projects.iq.harvard.edu/stat110](https://projects.iq.harvard.edu/stat110).
* Boyd, Vandenberghe, **Convex Optimization** (solo i primi capitoli): supporto per il modulo 04. Gratis su [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/).
* Prince, **Understanding Deep Learning**: testo principale per il modulo 05. Gratis su [udlbook.github.io](https://udlbook.github.io/udlbook/).

Leggere i libri non è obbligatorio. Sono lì per quando vuoi andare più a fondo su un tema.
