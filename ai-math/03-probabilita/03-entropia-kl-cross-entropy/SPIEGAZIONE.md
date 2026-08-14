# Entropia, KL divergence e cross entropy

## L'intuizione

Parti dalla **sorpresa**. Un evento raro, quando accade, sorprende molto. Un evento quasi certo non sorprende nessuno. La misura matematica è -log p: probabilità 1 dà sorpresa 0, probabilità piccola dà sorpresa grande. Useremo il logaritmo naturale, come fa PyTorch.

L'**entropia** di una distribuzione è la sorpresa media: quanto ti sorprende, in media, il prossimo campione. Una moneta onesta è massimamente imprevedibile: entropia alta. Una moneta che dà quasi sempre testa è noiosa: entropia bassa.

La **KL divergence** risponde a: quanto costa usare il modello sbagliato? Se il mondo segue la distribuzione P ma tu ragioni con la distribuzione Q, subisci una sorpresa extra rispetto a chi conosce P. La KL misura quella sorpresa extra media. Vale zero solo se Q è identica a P.

La **cross entropy** è la sorpresa media totale che provi usando Q in un mondo governato da P: la sorpresa inevitabile (entropia di P) più quella extra per il modello sbagliato (la KL).

## L'idea formale, in parole semplici

Per distribuzioni discrete, con la somma su tutti i valori possibili:

    sorpresa di un evento:  -log p
    entropia:               H(P)    = somma di p * (-log p)
    KL divergence:          KL(P‖Q) = somma di p * (log p - log q)
    cross entropy:          H(P,Q)  = somma di p * (-log q) = H(P) + KL(P‖Q)

Il simbolo ‖ nella KL è solo un separatore: si legge "KL da P a Q". Attenzione: non è simmetrica, KL(P‖Q) e KL(Q‖P) in generale differiscono.

Perché è il cuore della classificazione: la verità di un esempio è una distribuzione tutta concentrata sulla classe giusta (probabilità 1 lì, 0 altrove). In quel caso la cross entropy collassa in un solo termine: **meno log della probabilità che il modello dà alla classe giusta**. Modello sicuro e corretto: loss quasi zero. Modello sicuro e sbagliato: loss enorme. Minimizzare la cross entropy è la MLE della lezione 02 applicata a un classificatore.

## Esempio numerico a mano

Entropia di una moneta onesta, p = [0.5, 0.5]:

    H = 0.5 * (-log 0.5) + 0.5 * (-log 0.5) = log 2 ≈ 0.693

Moneta truccata, p = [0.9, 0.1]:

    H = 0.9 * 0.105 + 0.1 * 2.303 ≈ 0.095 + 0.230 = 0.325

Meno della metà: più prevedibile, meno sorpresa media.

Cross entropy di un classificatore: la verità è la classe 0, il modello dice q = [0.8, 0.2]:

    CE = -log 0.8 ≈ 0.223

Se il modello avesse detto q = [0.2, 0.8], cioè sicuro della classe sbagliata:

    CE = -log 0.2 ≈ 1.609

Sette volte peggio. La cross entropy punisce ferocemente la sicurezza mal riposta.

## Riferimenti

* Prince, Understanding Deep Learning: capitolo 5 (loss functions), dove cross entropy e likelihood vengono unite esattamente come qui.
* Mathematics for Machine Learning: capitolo 6 per le distribuzioni su cui questi concetti si appoggiano.

## E adesso

Esegui `python lesson.py`. Calcola entropia, KL e cross entropy a mano e le confronta con `F.kl_div`, `F.softmax` e `F.cross_entropy`, inclusa la softmax scritta da zero. Chiude con la curva dell'entropia della moneta al variare di p, salvata in `figures/`.
