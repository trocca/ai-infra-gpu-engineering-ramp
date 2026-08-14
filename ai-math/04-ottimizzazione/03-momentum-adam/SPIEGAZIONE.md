# Momentum e Adam

## L'intuizione

La pallina del gradient descent ha un difetto: non ha peso. A ogni passo riparte da ferma, guarda solo la pendenza del momento, e in una valle stretta e lunga finisce a zigzagare tra i fianchi invece di correre lungo il fondo.

**Momentum** le dà inerzia, come una vera palla che rotola. Ogni passo è una miscela del passo precedente e della pendenza attuale. Gli zigzag laterali si cancellano tra loro (un colpo a destra, uno a sinistra), mentre la componente costante lungo il fondovalle si accumula e prende velocità.

**Adam** aggiunge la seconda idea chiave: un passo su misura per ogni parametro. Tiene due medie mobili per ciascun peso: la direzione media del gradiente (come momentum) e la grandezza media del gradiente al quadrato. Poi divide la prima per la radice della seconda: i parametri con gradienti abitualmente enormi ricevono passi prudenti, quelli con gradienti minuscoli ricevono passi amplificati. È l'optimizer di default del deep learning moderno.

## L'idea formale, in parole semplici

**Momentum**, con β (beta, tipicamente 0.9) che decide quanta inerzia conservare:

    v = beta * v + gradiente
    w = w - lr * v

v è la velocità accumulata. Con beta = 0 torna il gradient descent puro.

**Adam**, con le costanti standard beta1 = 0.9, beta2 = 0.999, eps = 1e-8:

    m = beta1 * m + (1 - beta1) * gradiente          (direzione media)
    v = beta2 * v + (1 - beta2) * gradiente^2        (grandezza media)
    m_hat = m / (1 - beta1^t)                        (correzione: m parte da 0)
    v_hat = v / (1 - beta2^t)
    w = w - lr * m_hat / (sqrt(v_hat) + eps)

Le due righe "hat" correggono un difetto di gioventù: m e v partono da zero e nei primi passi sottostimano tutto; la divisione per (1 - beta^t), con t il numero del passo, li rimette in scala. La eps evita divisioni per zero. Niente magia: sei righe di aritmetica.

## Esempio numerico a mano

Momentum con beta = 0.9, lr = 0.1, e un gradiente che vale sempre 1 (fondovalle costante):

    passo 1: v = 0.9*0 + 1 = 1        aggiornamento = 0.1 * 1    = 0.1
    passo 2: v = 0.9*1 + 1 = 1.9      aggiornamento = 0.1 * 1.9  = 0.19
    passo 3: v = 0.9*1.9 + 1 = 2.71   aggiornamento = 0.271

La palla accelera: stessi gradienti, passi sempre più lunghi, fino a un massimo di 10 volte il passo base. Ora gradienti alternati +1, -1, +1 (zigzag tra i fianchi):

    passo 1: v = 1        passo 2: v = 0.9*1 - 1 = -0.1      passo 3: v = 0.9*(-0.1) + 1 = 0.91

La velocità resta piccola e oscilla attorno allo zero: gli zigzag si smorzano da soli. Stesso meccanismo, due comportamenti giusti.

## Riferimenti

* Mathematics for Machine Learning: capitolo 7, sezione 7.1.2 (gradient descent with momentum) e in generale la sezione 7.1.
* Prince, Understanding Deep Learning: capitolo 6, le sezioni su momentum e Adam.

## E adesso

Esegui `python lesson.py`. Implementa momentum e Adam da zero su una valle stretta, li confronta con il gradient descent puro, e poi la prova del nove: il tuo Adam scritto a mano e `torch.optim.Adam` devono produrre esattamente la stessa traiettoria, numero per numero.
