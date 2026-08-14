# Regressione lineare da zero

## L'intuizione

Questa lezione non introduce niente di nuovo. Fa una cosa migliore: prende tutti i pezzi che hai già costruito e li assembla nel **training loop**, il ciclo di quattro battute che addestra qualunque rete neurale, da questo modellino a GPT:

1. **Forward**: il modello fa le predizioni con i pesi attuali (una matmul, modulo 01).
2. **Loss**: si misura il punteggio di errore, la MSE (che dal modulo 03 sai essere una likelihood gaussiana sotto mentite spoglie).
3. **Backward**: autograd calcola il gradiente della loss rispetto a ogni peso (chain rule, modulo 02).
4. **Update**: un passo di discesa: `w -= lr * grad` (modulo 04).

Ripeti finché la loss smette di scendere. Tutto l'addestramento del deep learning, dal primo perceptron all'ultimo LLM, è questo giro di giostra con modelli sempre più grossi in mezzo.

## L'idea formale, in parole semplici

Il modello: pred = X @ w + b, con X la matrice delle case (feature standardizzate, lezione 03 del modulo 01), w i due pesi, b il bias.

La loss: MSE, media dei quadrati degli errori.

Due dettagli pratici che vedrai nel codice e che sono fonte di bug classici in PyTorch:

* L'update va fatto dentro `torch.no_grad()`: stai modificando i pesi, non calcolando qualcosa da derivare. Senza, autograd proverebbe a tracciare anche l'update.
* Dopo ogni passo serve `grad.zero_()`: PyTorch **accumula** i gradienti invece di sovrascriverli. Se non azzeri, il passo successivo usa la somma di tutti i gradienti passati. È il bug numero uno dei principianti.

Per questo modello esiste anche la soluzione esatta in forma chiusa (i minimi quadrati, `torch.linalg.lstsq`). La useremo come prova del nove: il training iterativo deve arrivare esattamente lì. Per le reti vere la forma chiusa non esiste, e resta solo il loop.

## Esempio numerico a mano

Primo giro del loop, versione ridotta a una casa: x = 1.26 (la quinta casa, standardizzata), y = 350, partenza w = 0, b = 0, lr = 0.1.

    forward:  pred = 0 * 1.26 + 0 = 0
    loss:     (0 - 350)² = 122500
    backward: dloss/dw = 2 * (pred - y) * x = 2 * (-350) * 1.26 = -882
              dloss/db = 2 * (pred - y)     = -700
    update:   w = 0 - 0.1 * (-882) = 88.2
              b = 0 - 0.1 * (-700) = 70

Dopo un solo passo il modello è già molto meno sbagliato: pred = 88.2 * 1.26 + 70 ≈ 181. Il loop non fa che ripetere questa correzione, sempre più fine.

## Riferimenti

* Prince, Understanding Deep Learning: capitolo 2 (supervised learning).
* Mathematics for Machine Learning: capitolo 9 (linear regression), in particolare il legame tra minimi quadrati e maximum likelihood.

## E adesso

Esegui `python lesson.py`. Il training loop completo sulle 5 case, la loss che crolla, le predizioni finali confrontate coi prezzi veri, e la verifica contro la soluzione esatta dei minimi quadrati. La curva di training finisce in `figures/`.
