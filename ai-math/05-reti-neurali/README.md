# Modulo 05: reti neurali dalla matematica

La sintesi. Ogni pezzo costruito nei moduli precedenti (matrici, gradienti, chain rule, likelihood, cross entropy, gradient descent, Adam) si incastra qui, e alla fine del modulo avrai scritto a mano: una regressione lineare che impara, un classificatore, una rete neurale vera senza `nn.Module`, backpropagation verificata anello per anello, e la matematica dell'attention che fa funzionare i transformer.

Nessuna magia rimasta: solo matmul, pendenze e sorprese medie.

## Lezioni

| Cartella | Argomento | Cosa saprai fare alla fine |
|---|---|---|
| `01-regressione-lineare-da-zero` | Il ciclo di training completo | Addestrare il primo modello vero sulle 5 case |
| `02-regressione-logistica` | Sigmoid, binary cross entropy | Costruire un classificatore e leggerne le probabilità |
| `03-mlp-da-zero-senza-nn` | Hidden layer, ReLU, XOR | Costruire una rete che risolve un problema impossibile per i modelli lineari |
| `04-autograd-sotto-il-cofano` | Il grafo di calcolo, backprop a mano | Rifare il lavoro di backward() a mano e verificarlo al decimale |
| `05-attention-e-transformer` | Query, key, value, softmax, maschera causale | Calcolare a mano il meccanismo che fa funzionare gli LLM |

Le lezioni vanno fatte in ordine: sono una scala, e ogni gradino usa il precedente.

## Riferimenti ai libri

* **Understanding Deep Learning** (Prince), il testo principale di questo modulo:
  * Capitolo 2 (supervised learning) per la lezione 01.
  * Capitolo 5 (loss functions) per la lezione 02.
  * Capitoli 3 e 4 (shallow e deep networks) per la lezione 03.
  * Capitolo 7 (gradients and initialization) per la lezione 04.
  * Capitolo 12 (transformers) per la lezione 05.
* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), di supporto:
  * Capitolo 9 (linear regression) per la lezione 01, capitolo 5 sezione 5.6 (backpropagation) per la lezione 04.

## Tempo stimato

Circa 3 settimane a 4 o 5 ore a settimana. Le lezioni 04 e 05 sono il traguardo del percorso: prenditi il tempo di eseguirle nel debugger, riga per riga.

## Il filo conduttore

Le 5 case aprono il modulo: il modello lineare che nel modulo 01 era una matmul, nel 02 una pendenza e nel 04 una discesa, qui diventa un training loop completo, e poi un classificatore. Dalla lezione 03 in poi si sale: XOR, backprop, attention. Alla fine, quando leggerai "multi head self attention" in un paper, saprai che è la matmul del modulo 01 con un vestito elegante.
