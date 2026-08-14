# Autograd sotto il cofano

## L'intuizione

Per quattro moduli hai chiamato `backward()` e i gradienti sono apparsi. Oggi apriamo il cofano. La sorpresa è quanto è poco misterioso: autograd è un registratore più la chain rule.

L'analogia giusta per chi fa debugging da una vita: **il backward è camminare uno stack trace all'indietro**. Durante il forward, ogni operazione su un tensor tracciato viene registrata in un grafo: chi sono gli input, quale operazione è stata fatta, come si calcola la sua derivata locale. È il call stack del calcolo. Quando chiami `backward()`, PyTorch percorre quel grafo dalla loss all'indietro, moltiplicando le pendenze locali anello per anello: la chain rule del modulo 02, automatizzata.

Puoi letteralmente ispezionare il grafo: ogni tensor intermedio ha un campo `grad_fn` che dice quale operazione l'ha creato, e da lì si risale tutta la catena. Lo faremo nello script, come si esplora uno stack.

## L'idea formale, in parole semplici

Tre regole compongono tutto il meccanismo:

1. **Registrazione**: ogni operazione sa calcolare la propria derivata locale. Moltiplicazione, somma, ReLU, exp: ognuna conosce solo se stessa. Nessuna sa niente del modello intero.
2. **Chain rule all'indietro**: il gradiente che arriva dall'alto viene moltiplicato per la derivata locale e passato agli input dell'operazione. Si parte dalla loss (gradiente 1 rispetto a se stessa) e si scende fino ai pesi.
3. **Accumulo**: se un tensor è usato in più punti del grafo, i gradienti che gli arrivano dai vari rami si **sommano**. È il motivo per cui serve `grad.zero_()` tra un passo e l'altro: l'accumulo è una feature del grafo, non un bug.

Quando la lezione parla di "backpropagation" nei paper, è esattamente questo: forward per calcolare i valori, backward per distribuire le pendenze. Il costo è circa il doppio del solo forward, indipendentemente dal numero di parametri. È questa efficienza che rende addestrabili i modelli giganti.

## Esempio numerico a mano

La catena della lezione 03 del modulo 02, con numeri nuovi: x = 2, w = 3, target t = 10.

Forward, salvando ogni valore intermedio:

    y    = w * x    = 6
    err  = y - t    = -4
    loss = err²     = 16

Backward, dalla loss verso w, moltiplicando le derivate locali:

    dloss/dloss = 1                        (si parte sempre da 1)
    dloss/derr  = 2 * err       = -8       (derivata locale di err²)
    dloss/dy    = -8 * 1        = -8       (derivata locale di y - t rispetto a y)
    dloss/dw    = -8 * x        = -16      (derivata locale di w*x rispetto a w)
    dloss/dx    = -8 * w        = -24      (e volendo anche verso x)

Nello script rifaremo questi conti su una vera rete a uno strato nascosto, formula per formula, e ogni numero verrà confrontato con quello che `backward()` deposita nei `.grad`. Devono coincidere al decimale. Coincideranno.

## Riferimenti

* Prince, Understanding Deep Learning: capitolo 7 (gradients and initialization), le sezioni sull'algoritmo di backpropagation.
* Mathematics for Machine Learning: capitolo 5, sezione 5.6 (backpropagation and automatic differentiation).

## E adesso

Esegui `python lesson.py`, meglio ancora: aprilo nel debugger. Prima esplora il grafo con `grad_fn` come fosse uno stack trace, poi fa backpropagation a mano su un MLP vero e verifica ogni singolo gradiente contro autograd. Dopo questa lezione, `backward()` non è più una scatola nera.
