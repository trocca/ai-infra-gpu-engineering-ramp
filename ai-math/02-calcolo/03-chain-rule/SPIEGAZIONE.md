# La chain rule

## L'intuizione

Cambi valuta due volte: euro in dollari, poi dollari in yen. Se un euro vale 1.1 dollari e un dollaro vale 150 yen, quanto vale un euro in yen? Le due conversioni si moltiplicano: 1.1 * 150 = 165 yen.

La chain rule (regola della catena) dice che le derivate funzionano allo stesso modo. Se x influenza u, e u influenza y, allora la sensibilità di y rispetto a x è il prodotto delle due sensibilità lungo la catena:

    quanto y sente x = (quanto y sente u) * (quanto u sente x)

Una rete neurale è una catena lunghissima: input, primo layer, secondo layer, predizione, loss. Per sapere quanto la loss sente un peso sepolto all'inizio, moltiplichi le pendenze anello per anello, all'indietro. Questo processo ha un nome famoso: backpropagation. La chain rule è tutto quello che c'è dentro.

## L'idea formale, in parole semplici

Se y = f(g(x)), cioè prima applichi g e poi f, la derivata della composizione è:

    dy/dx = f'(g(x)) * g'(x)

A parole: derivata della funzione esterna, calcolata nel punto interno, per la derivata della funzione interna. La notazione con le frazioni rende l'idea quasi ovvia:

    dy/dx = dy/du * du/dx

dove u = g(x) è il valore intermedio. Le "du" sembrano semplificarsi come in una frazione. Non è una dimostrazione, ma è un ottimo modo per ricordarla. E la catena può essere lunga quanto vuoi: tre, dieci, cento anelli. Si moltiplica tutto.

## Esempio numerico a mano

Prendiamo y = (3x + 1)² nel punto x = 2. La catena è: u = 3x + 1 (interna), y = u² (esterna).

Passo 1, valore interno:

    u = 3 * 2 + 1 = 7

Passo 2, le due pendenze separate:

    du/dx = 3           (derivata di 3x + 1)
    dy/du = 2u = 14     (derivata di u^2, calcolata in u = 7)

Passo 3, moltiplico gli anelli:

    dy/dx = 14 * 3 = 42

Verifica indipendente: espandendo, y = 9x² + 6x + 1, la cui derivata è 18x + 6, che in x = 2 vale 42. Torna.

Ora la stessa idea sul modello delle case, con una casa sola. La catena è:

    pred = w * x        (predizione)
    err  = pred - y     (errore)
    loss = err²         (punteggio di errore)

Le pendenze degli anelli: dloss/derr = 2 * err, derr/dpred = 1, dpred/dw = x. Quindi:

    dloss/dw = 2 * err * 1 * x

Questa formuletta, generalizzata, è quello che `backward()` calcola per ogni peso di una rete.

## Riferimenti

* Mathematics for Machine Learning: capitolo 5, sezione 5.1.2 (differentiation rules) e sezione 5.2.2 (chain rule).
* Parr, Howard, The Matrix Calculus You Need For Deep Learning: le sezioni sulla chain rule, il cuore del paper.

## E adesso

Esegui `python lesson.py`. Rifà la catena di (3x + 1)² numero per numero, poi smonta la catena della loss di una casa e confronta ogni anello calcolato a mano con quello che trova autograd. Sono identici.
