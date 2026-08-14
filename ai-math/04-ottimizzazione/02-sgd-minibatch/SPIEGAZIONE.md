# SGD e minibatch

## L'intuizione

Il gradient descent della lezione 01 ha un costo nascosto: per fare UN passo deve calcolare l'errore su TUTTO il dataset. Con 5 case è gratis. Con 10 milioni di immagini, ogni singolo passo costerebbe una scansione completa dei dati. Impraticabile.

L'idea salvavita: non serve la pendenza esatta, basta una stima onesta. Prendi una manciata di esempi a caso, un **minibatch**, e calcola la pendenza solo su quelli. È rumorosa: ogni manciata dice una cosa un po' diversa. Ma in media punta nella direzione giusta, e costa pochissimo. Questo è lo **stochastic gradient descent**, SGD.

La pallina nella nebbia ora è anche un po' ubriaca: ogni passo ha una direzione leggermente sbagliata. Ma i passi sono tanti, economici, e gli errori si compensano. E c'è un bonus inaspettato: quel rumore aiuta a non restare intrappolati in conche secondarie della valle.

## L'idea formale, in parole semplici

Vocabolario del training, che ritroverai in ogni paper e in ogni log di addestramento:

* **batch size**: quanti esempi entrano in un minibatch. Tipici: 32, 64, 256.
* **epoca** (epoch): un giro completo su tutto il dataset, un minibatch alla volta.
* **shuffle**: a ogni epoca si rimescola l'ordine degli esempi, così i minibatch cambiano composizione e il rumore non si ripete uguale.

Il ciclo completo: mescola i dati, tagliali a minibatch, e per ogni minibatch calcola la loss solo su quei dati, poi backward e aggiornamento. Fine dei minibatch: è passata un'epoca. Ricomincia.

Il gradiente di minibatch è una **stima non distorta** di quello vero: in media, su tante estrazioni, coincide con il gradiente calcolato su tutto il dataset. È lo stesso patto della lezione 01 del modulo 03: la media empirica converge al valore atteso.

## Esempio numerico a mano

Modello a un peso, pred = w * x, con w = 1, e tre esempi:

    x = [1, 2, 3]    y = [2, 4, 6]    (regola vera: y = 2x)

Gradiente per singolo esempio: 2 * (w*x - y) * x. Con w = 1 gli errori sono -1, -2, -3, quindi:

    esempio 1: 2 * (-1) * 1 = -2
    esempio 2: 2 * (-2) * 2 = -8
    esempio 3: 2 * (-3) * 3 = -18

Gradiente esatto (media su tutti): (-2 - 8 - 18) / 3 = -9.33.

Minibatch di 2 esempi:

    batch {1, 2}: (-2 - 8) / 2  = -5
    batch {1, 3}: (-2 - 18) / 2 = -10
    batch {2, 3}: (-8 - 18) / 2 = -13

Nessuno vale esattamente -9.33, tutti hanno il segno giusto, e la loro media (-5 - 10 - 13) / 3 = -9.33 è proprio il gradiente esatto. Rumore sì, distorsione no.

## Riferimenti

* Mathematics for Machine Learning: capitolo 7, la parte di sezione 7.1 dedicata allo stochastic gradient descent.
* Prince, Understanding Deep Learning: capitolo 6, le sezioni su SGD e minibatch.

## E adesso

Esegui `python lesson.py`. Genera 200 case sintetiche con la stessa regola delle nostre 5, confronta il gradiente esatto con quelli dei minibatch, e addestra il modello con SGD vero: shuffle, epoche, minibatch. Le curve di loss finiscono in `figures/`.
