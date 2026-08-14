# Derivate parziali e gradiente

## L'intuizione

Nella lezione 01 la funzione aveva una sola manopola. Ma un modello vero ne ha tante: il nostro modello delle case ha un peso per i metri quadri, uno per le stanze, un bias. La loss dipende da tutte insieme.

Come si misura la pendenza quando le manopole sono tante? Una alla volta. Congeli tutte le manopole tranne una, muovi quella di un pelo, guardi quanto cambia il risultato. Quella è la **derivata parziale** rispetto a quella manopola. Poi ripeti per ognuna.

Il **gradiente** è semplicemente l'elenco completo: una derivata parziale per manopola, impacchettate in un vettore. Se sei su una collina, il gradiente è la freccia che punta verso la salita più ripida. E quindi, il suo opposto punta verso la discesa più ripida: è la direzione che useremo per imparare.

## L'idea formale, in parole semplici

La derivata parziale di f rispetto a x si scrive ∂f/∂x. Il simbolo ∂ è una "d arrotondata" e si legge come una normale d: ricorda che le altre variabili sono tenute ferme.

Il gradiente si scrive ∇f, dove ∇ (si legge "nabla") indica il gradiente, cioè l'elenco delle pendenze:

    ∇f = [∂f/∂x, ∂f/∂y, ...]

È un vettore con tante componenti quante sono le variabili. Due fatti da ricordare:

1. Il gradiente punta nella direzione di **massima salita** di f.
2. La sua lunghezza dice quanto è ripida quella salita.

In PyTorch non cambia nulla rispetto alla lezione 01: metti le variabili in un tensor con `requires_grad=True`, calcoli f, chiami `backward()`, e in `.grad` trovi l'intero gradiente in un colpo solo. Autograd fa le derivate parziali per te.

## Esempio numerico a mano

Prendiamo f(x, y) = x² + 3y nel punto (2, 1).

Derivata parziale rispetto a x: tengo ferma y, quindi il termine 3y è una costante e sparisce. Resta la derivata di x², che è 2x:

    ∂f/∂x = 2x = 4

Derivata parziale rispetto a y: tengo ferma x, quindi x² sparisce. La derivata di 3y è 3:

    ∂f/∂y = 3

Il gradiente nel punto (2, 1) è quindi:

    ∇f(2, 1) = [4, 3]

Lettura: aumentare x di un pelo fa salire f a velocità 4, aumentare y di un pelo la fa salire a velocità 3. E la direzione [4, 3], presa tutta insieme, è la salita più rapida da quel punto.

## Riferimenti

* Mathematics for Machine Learning: capitolo 5, sezione 5.2 (partial differentiation and gradients).
* Parr, Howard, The Matrix Calculus You Need For Deep Learning: la sezione sulle derivate parziali e sul gradiente.

## E adesso

Esegui `python lesson.py`. Calcola le derivate parziali a mano e con autograd, poi disegna il campo dei gradienti: una mappa di frecce che, in ogni punto, indicano la salita. Infine misura il gradiente della loss delle case rispetto a peso e bias insieme.
