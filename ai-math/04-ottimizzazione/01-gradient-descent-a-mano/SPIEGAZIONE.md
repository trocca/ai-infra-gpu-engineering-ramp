# Gradient descent a mano

## L'intuizione

Sei una pallina su un fianco di una valle, di notte, nella nebbia. Non vedi il fondo. Vedi solo il pezzetto di terreno sotto di te, e senti da che parte pende. La strategia ovvia è anche quella giusta: fai un passo in discesa. Poi risenti la pendenza, e fai un altro passo. Ripeti.

Questo è il gradient descent, tutto qui. La valle è la loss, il punteggio di errore in funzione dei pesi. La pendenza sotto i piedi è il gradiente (modulo 02). Il passo è l'aggiornamento dei pesi. La nebbia è il fatto che nessuno può permettersi di calcolare la loss dappertutto: solo dove si trova ora.

## L'idea formale, in parole semplici

L'aggiornamento, ripetuto fino a convergenza:

    w nuovo = w - lr * gradiente

Il segno meno c'è perché il gradiente punta in salita (lezione 02 del modulo 02) e noi vogliamo scendere. Il numero lr si chiama **learning rate**: la lunghezza del passo. È il primo iperparametro vero che incontri, e domina tutto:

* lr troppo piccolo: passi da formica. Scendi, ma servono ere geologiche.
* lr giusto: scendi rapidamente e ti assesti sul fondo.
* lr troppo grande: scavalchi il fondo e rimbalzi da un fianco all'altro, sempre più in alto. La loss diverge, spesso fino a NaN.

Quando la valle ha un fondo solo (funzione **convessa**, come una parabola), il gradient descent con lr sensato arriva al minimo globale. Le loss delle reti profonde non sono convesse, hanno valli secondarie, ma sorprendentemente il metodo funziona lo stesso: al modulo 05 ne riparliamo.

## Esempio numerico a mano

Prendiamo loss(w) = (w - 3)², una valle con il fondo in w = 3. Il gradiente è 2(w - 3). Parto da w = 0 con lr = 0.25:

    passo 0: gradiente = 2*(0 - 3)     = -6      w = 0 - 0.25*(-6)     = 1.5
    passo 1: gradiente = 2*(1.5 - 3)   = -3      w = 1.5 - 0.25*(-3)   = 2.25
    passo 2: gradiente = 2*(2.25 - 3)  = -1.5    w = 2.25 - 0.25*(-1.5) = 2.625
    passo 3: gradiente = 2*(2.625 - 3) = -0.75   w = 2.8125

Ogni passo dimezza la distanza dal fondo: 3, 1.5, 0.75, 0.375. Convergenza pulita.

Ora il caso patologico: stesso punto di partenza, lr = 1.1:

    passo 0: gradiente = -6      w = 0 - 1.1*(-6)   = 6.6     (scavalcato il fondo)
    passo 1: gradiente = 7.2     w = 6.6 - 1.1*7.2  = -1.32   (scavalcato di nuovo, piu' in la')

Le oscillazioni si allargano a ogni passo: divergenza. Stessa valle, stesso algoritmo, passo sbagliato.

## Riferimenti

* Mathematics for Machine Learning: capitolo 7, sezione 7.1 (optimization using gradient descent) e sezione 7.3 per la convessità.
* Prince, Understanding Deep Learning: capitolo 6 (fitting models).
* Boyd, Vandenberghe, Convex Optimization: capitoli 2 e 3, solo se vuoi la teoria delle valli a fondo unico.

## E adesso

Esegui `python lesson.py`. Rifà i conti qui sopra passo per passo, poi addestra il modello delle case con autograd e confronta tre learning rate: formica, giusto, e divergente. Le tre traiettorie finiscono in `figures/`.
