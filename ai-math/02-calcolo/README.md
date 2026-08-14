# Modulo 02: calcolo

Il calcolo differenziale risponde a una sola domanda, ripetuta in mille forme: se muovo questa manopola di un pelo, quanto cambia il risultato? Quella "sensibilità" si chiama derivata. Le reti neurali imparano esattamente così: misurano quanto ogni peso fa cambiare l'errore, e girano le manopole nella direzione giusta.

## Lezioni

| Cartella | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| `01-derivata-come-pendenza` | Derivata, differenze finite, autograd | Calcolare la pendenza di una funzione in tre modi diversi e verificare che coincidono |
| `02-derivate-parziali-gradiente` | Derivate parziali, gradiente | Misurare la pendenza rispetto a ogni manopola, e leggere ∇ senza paura |
| `03-chain-rule` | Regola della catena | Derivare funzioni composte moltiplicando le pendenze anello per anello |
| `04-matrix-calculus-jacobiani` | Jacobiane, gradienti di vettori | Gestire le forme di derivate quando input e output sono vettori |

Le lezioni vanno fatte in ordine: la chain rule della lezione 03 è il cuore di backpropagation, e la lezione 04 sistema le forme delle matrici.

## Riferimenti ai libri

* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), il testo principale:
  * Capitolo 5, Vector Calculus: sezione 5.1 per la lezione 01, sezione 5.2 per le lezioni 02 e 03, sezione 5.3 per la lezione 04, sezione 5.6 come anteprima di backpropagation.
* **The Matrix Calculus You Need For Deep Learning** (Parr, Howard), di supporto:
  * Le sezioni introduttive sulle derivate parziali per la lezione 02, la sezione sulla chain rule per la lezione 03, la sezione sulla Jacobiana per la lezione 04.

## Tempo stimato

Circa 2 settimane a 4 o 5 ore a settimana. La lezione 03 è la più importante del modulo: se la chain rule ti è chiara, backpropagation (modulo 05) sarà una formalità.

## Il filo conduttore

Il dataset delle 5 case torna qui con un ruolo nuovo: la loss, il punteggio di errore del modello, vista come funzione di un peso. Nella lezione 01 scoprirai che è una parabola, e che la sua pendenza ti dice da che parte correggere il peso. È il seme del gradient descent del modulo 04.
