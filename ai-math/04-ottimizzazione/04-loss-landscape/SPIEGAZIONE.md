# Il loss landscape

## L'intuizione

Per tutto il modulo la pallina è scesa nella nebbia. In questa lezione la nebbia si alza. Con un modello a due soli parametri (un peso w e un bias b) possiamo permetterci il lusso proibito ai modelli grandi: calcolare la loss DAPPERTUTTO, su una griglia di combinazioni (w, b), e guardare la valle intera dall'alto.

Lo strumento è la mappa topografica: le **curve di livello** (contour lines). Ogni curva unisce i punti con la stessa loss, come le isoipse di una carta escursionistica. Curve fitte vuol dire pendenza ripida, curve rade vuol dire pianura. Il centro degli anelli è il fondo della valle: la combinazione (w, b) perfetta.

Sopra questa mappa disegneremo le traiettorie degli algoritmi delle lezioni scorse. Vedrai il gradient descent tagliare le curve di livello sempre perpendicolarmente (il gradiente è ortogonale alle curve di livello), zigzagare dove la valle è stretta, e il momentum filare dritto lungo il fondovalle.

## L'idea formale, in parole semplici

La superficie di loss (loss landscape) del modello delle case è la funzione

    loss(w, b) = MSE dei prezzi con peso w e bias b

vista come paesaggio: due coordinate orizzontali (w e b), l'altitudine è la loss. Per disegnarla si valuta la loss su una griglia di punti e si tracciano le curve di livello.

Due proprietà della nostra valle:

1. È **convessa**: la MSE di un modello lineare è una parabola in ogni direzione, quindi c'è un solo fondo, niente conche secondarie. È il motivo per cui la regressione lineare converge sempre.
2. La sua **forma dipende dalle unità dei dati**. Con feature non normalizzate la valle è un canyon strettissimo e obliquo, e il gradient descent soffre. Con feature standardizzate (modulo 01, lezione 03) la valle è quasi circolare e la discesa fila. Stessa matematica del confronto GD contro momentum della lezione 03.

Le reti profonde del modulo 05 hanno paesaggi non convessi, con milioni di dimensioni, che nessuno può disegnare per intero. Ma l'intuizione costruita qui su due dimensioni è quella che i ricercatori usano davvero per ragionarci.

## Esempio numerico a mano

Versione ridotta: una sola casa, x = 1 (feature normalizzata), prezzo y = 250. La loss è

    loss(w, b) = (w * 1 + b - 250)²

Calcolo l'altitudine in tre punti della mappa:

    (w=100, b=100):  (100 + 100 - 250)² = (-50)²  = 2500
    (w=200, b=50):   (200 + 50 - 250)²  = 0       (fondovalle)
    (w=250, b=50):   (250 + 50 - 250)²  = 2500

Nota: anche (w=100, b=150) dà zero. Con una casa sola il fondo non è un punto ma una linea intera di soluzioni: dati insufficienti per fissare due parametri. Con le 5 case dello script il fondo torna a essere un punto solo.

## Riferimenti

* Mathematics for Machine Learning: capitolo 7, sezione 7.1 per la discesa e sezione 7.3 per la convessità.
* Prince, Understanding Deep Learning: capitolo 6, dove i loss landscape sono disegnati proprio così.

## E adesso

Esegui `python lesson.py`. Costruisce la griglia, verifica che il minimo della mappa coincide con la soluzione esatta dei minimi quadrati, e salva in `figures/` la valle con sopra le traiettorie di GD e momentum, nebbia tolta.
