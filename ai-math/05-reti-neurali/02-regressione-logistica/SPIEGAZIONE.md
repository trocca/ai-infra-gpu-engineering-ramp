# Regressione logistica

## L'intuizione

Cambio di domanda. Finora: "quanto costa questa casa?". Adesso: "questa casa è cara, sì o no?". Da un numero continuo a una classe. Serve un modello che risponda con una **probabilità**: "cara al 92 per cento".

Il trucco è minimo. Il modello lineare produce già un punteggio: alto per le case grandi, basso per le piccole. Basta schiacciare quel punteggio nell'intervallo tra 0 e 1. La funzione che lo fa si chiama **sigmoid**: una S morbida che manda i punteggi molto negativi verso 0, i molto positivi verso 1, e lo zero esattamente a 0.5.

Nonostante il nome, la regressione logistica è un classificatore. Ed è anche un neurone: punteggio lineare più funzione di schiacciamento è esattamente il mattone con cui, nella prossima lezione, costruiremo la rete.

## L'idea formale, in parole semplici

Il modello, in due righe:

    z = x · w + b            (il punteggio, detto logit)
    p = sigmoid(z) = 1 / (1 + e^(-z))     (la probabilita' della classe 1)

La loss è la **binary cross entropy** (BCE), figlia diretta della lezione 03 del modulo 03: la sorpresa media del modello davanti alla verità.

    BCE = media di [ -log(p) se la verita' e' 1, -log(1-p) se la verita' e' 0 ]

Che è anche, di nuovo, una negative log likelihood: massimizzare la probabilità delle etichette osservate. MLE, entropia e training loop si incontrano tutti qui.

Perché non usare la MSE? Si può, ma la BCE punisce molto più duramente l'errore sicuro di sé (ricordi: -log di una probabilità piccola esplode) e produce gradienti più sani per questa forma di modello. È lo standard, e ora sai da dove viene.

## Esempio numerico a mano

La sigmoid su tre punteggi:

    z = 0:    p = 1 / (1 + e^0)  = 1 / 2      = 0.5     (indeciso)
    z = 2:    p = 1 / (1 + e^-2) = 1 / 1.135  ≈ 0.88    (abbastanza sicuro: classe 1)
    z = -2:   p ≈ 0.12                                  (abbastanza sicuro: classe 0)

La BCE su un esempio con verità 1:

    modello dice p = 0.88:  loss = -log(0.88) ≈ 0.128   (bene)
    modello dice p = 0.12:  loss = -log(0.12) ≈ 2.12    (sbagliato e sicuro: castigo)

Nello script etichettiamo le 5 case: care quelle da 250 mila in su, cioè y = [0, 0, 1, 1, 1], e il modello impara il confine da solo.

## Riferimenti

* Prince, Understanding Deep Learning: capitolo 5 (loss functions), le sezioni sulla classificazione binaria.
* Mathematics for Machine Learning: capitolo 8, sezione 8.3, per il legame generale tra likelihood e loss.

## E adesso

Esegui `python lesson.py`. Sigmoid e BCE scritte a mano e verificate contro `torch.sigmoid` e `F.binary_cross_entropy`, poi il training loop (identico a ieri, cambia solo la loss) e le probabilità finali casa per casa. La curva a S finisce in `figures/`.
