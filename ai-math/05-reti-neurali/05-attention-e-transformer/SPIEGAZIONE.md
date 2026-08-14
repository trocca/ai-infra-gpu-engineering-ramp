# Attention e la matematica dei transformer

## L'intuizione

Ultima lezione, ultimo mattone: il meccanismo che fa funzionare gli LLM. E la buona notizia è che conosci già tutti i pezzi.

L'**attention** è un dizionario sfumato. In un dizionario vero cerchi una chiave esatta e prendi un valore. Nell'attention ogni parola della frase pone una domanda (la sua **query**), ogni parola espone un'etichetta (la sua **key**) e un contenuto (il suo **value**). La ricerca non trova UNA chiave: misura quanto la domanda somiglia a ogni etichetta (prodotto scalare, modulo 01), trasforma le somiglianze in percentuali (softmax, modulo 03), e restituisce la media pesata dei contenuti. Ogni parola esce arricchita da un mix delle altre, pesato per rilevanza.

Esempio mentale: nella frase "il gatto dorme perché è stanco", la parola "è" chiede "di chi sto parlando?", la key di "gatto" risponde forte, e il value di "gatto" fluisce dentro "è". Questo è tutto il segreto del contesto negli LLM.

## L'idea formale, in parole semplici

Ogni token (parola o pezzo di parola) arriva come vettore. Tre matrici di pesi, imparate col training, lo proiettano in tre ruoli: Q = query, K = key, V = value. Poi:

    attention(Q, K, V) = softmax(Q @ Kᵀ / √d) @ V

Letta da dentro a fuori, con le forme accanto:

1. `Q @ Kᵀ`: tutti i prodotti scalari domanda per etichetta. Con n token, una matrice n x n di punteggi di affinità. Una matmul del modulo 01.
2. `/ √d`: si divide per la radice della dimensione d dei vettori, perché con d grande i prodotti scalari crescono e la softmax saturerebbe sputando quasi solo 0 e 1. Un fattore di scala, niente di più.
3. `softmax` riga per riga: ogni riga diventa una distribuzione di probabilità: "il token i distribuisce così la sua attenzione".
4. `@ V`: la media pesata dei contenuti. Un'altra matmul.

Per i modelli che generano testo serve un dettaglio: la **maschera causale**. Un token non deve vedere il futuro, quindi i punteggi verso i token successivi vengono messi a meno infinito prima della softmax, che li trasforma in probabilità zero. Un transformer è questo blocco, più un MLP (lezione 03), ripetuti in pila decine di volte, addestrati con cross entropy (modulo 03) e Adam (modulo 04). Fine della lista degli ingredienti.

## Esempio numerico a mano

Due token, dimensione 2. Il token 1 ha query q = [1, 0]. Le key sono k1 = [1, 0] e k2 = [0, 1], i value v1 = [10, 0] e v2 = [0, 10].

    punteggi:  q · k1 = 1,   q · k2 = 0
    scala:     √d = √2 ≈ 1.41, quindi [0.71, 0]
    softmax:   e^0.71 = 2.03, e^0 = 1: percentuali [2.03, 1] / 3.03 = [0.67, 0.33]
    uscita:    0.67 * [10, 0] + 0.33 * [0, 10] = [6.7, 3.3]

Il token 1 somiglia di più alla key 1, quindi pesca due terzi del contenuto da v1 e un terzo da v2. Nessun passaggio nuovo: dot product, softmax, media pesata.

## Riferimenti

* Prince, Understanding Deep Learning: capitolo 12 (transformers), in particolare le sezioni su self attention e scaled dot product attention.

## E adesso

Esegui `python lesson.py`. Costruisce l'attention a mano su 4 token, la verifica contro `F.scaled_dot_product_attention` di PyTorch, aggiunge la maschera causale, e stampa la matrice di attenzione: chi guarda chi. La heatmap finisce in `figures/`. È l'ultima lezione: alla fine, guarda indietro e conta i mattoni.
