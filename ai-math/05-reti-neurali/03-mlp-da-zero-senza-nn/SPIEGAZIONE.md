# MLP da zero, senza nn

## L'intuizione

I modelli delle lezioni 01 e 02 hanno un limite strutturale: tracciano confini dritti. Una retta per separare, un piano per predire. Ma esistono problemi dove nessuna linea dritta funziona. Il più famoso è **XOR**: quattro punti, (0,0) e (1,1) in una classe, (0,1) e (1,0) nell'altra. Prova a separarli con una riga: impossibile. Fu l'argomento che congelò le reti neurali per anni.

La soluzione: impilare. Un primo strato lineare inventa nuove feature, una piega (la **ReLU**) le rende non lineari, un secondo strato lineare combina le feature piegate. Il risultato è un **MLP** (multi layer perceptron), e può tracciare confini spezzati, curvi, complicati a piacere. Questa è una rete neurale. Non c'è altro nel mattone di base.

## L'idea formale, in parole semplici

Il forward di un MLP a uno strato nascosto:

    h   = ReLU(x @ W1 + b1)      (lo strato nascosto: nuove feature)
    out = h @ W2 + b2            (lo strato di uscita: la combinazione)

La ReLU (rectified linear unit) è la non linearità più semplice possibile: max(0, z). Sotto zero taglia, sopra zero lascia passare. Una piega, letteralmente.

Perché la piega è indispensabile: due strati lineari in fila si fondono in uno solo (matmul di matmul è una matmul, modulo 01). Senza ReLU in mezzo, l'MLP collasserebbe in una regressione lineare travestita. È la piega che compra la potenza.

Lo strato nascosto ha una dimensione a scelta: più neuroni, più pieghe disponibili, confini più ricchi. Nello script ne useremo 16, molti più dei 2 strettamente necessari: la larghezza extra rende l'addestramento molto più affidabile, perché la superficie di loss di XOR ha vere conche secondarie (ricordi il modulo 04?) e con poche pieghe di riserva il gradient descent ci resta intrappolato spesso.

## Esempio numerico a mano

Costruiamo a mano un MLP che risolve XOR, con 2 neuroni nascosti:

    h1 = ReLU(x1 + x2)          (conta quanti input sono accesi)
    h2 = ReLU(x1 + x2 - 1)      (si accende solo se ENTRAMBI sono accesi)
    out = h1 - 2 * h2

Verifica su tutti e quattro i casi:

    (0,0): h1 = 0, h2 = 0        out = 0 - 0 = 0    corretto
    (1,0): h1 = 1, h2 = ReLU(0) = 0    out = 1 - 0 = 1    corretto
    (0,1): h1 = 1, h2 = 0        out = 1        corretto
    (1,1): h1 = 2, h2 = 1        out = 2 - 2 = 0    corretto

Leggi cosa è successo: h1 e h2 sono feature inventate ("almeno uno acceso", "tutti e due accesi") e l'uscita le combina in "almeno uno, ma non tutti e due". Che è la definizione di XOR. Le reti addestrate fanno esattamente questo, solo che le feature le scoprono da sole con il gradient descent.

## Riferimenti

* Prince, Understanding Deep Learning: capitolo 3 (shallow neural networks) e capitolo 4 (deep neural networks).
* Mathematics for Machine Learning: capitolo 5, sezione 5.6, per l'anticipo su come si deriva una composizione di strati.

## E adesso

Esegui `python lesson.py`. Prima verifica l'MLP costruito a mano sulla tabella XOR, poi ne addestra uno con pesi casuali e guarda il gradient descent riscoprire la soluzione. Solo tensor grezzi: niente `nn.Module`, niente `nn.Linear`. La mappa del confine di decisione finisce in `figures/`.
