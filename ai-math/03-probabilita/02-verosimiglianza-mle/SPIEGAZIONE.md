# Verosimiglianza e maximum likelihood

## L'intuizione

Lavori a un'escalation: hai un log di eventi (i dati) e diverse ipotesi su cosa li ha prodotti (i parametri). Alcune ipotesi rendono quel log plausibile, altre lo renderebbero un miracolo. La **verosimiglianza** (likelihood) misura esattamente questo: quanto è probabile osservare proprio i dati che hai, se l'ipotesi fosse vera.

La **stima di massima verosimiglianza** (MLE, maximum likelihood estimation) è la regola più naturale del mondo: tra tutte le ipotesi, scegli quella sotto cui i tuoi dati sono meno miracolosi. Non garantisce la verità. Sceglie la spiegazione che rende i fatti più ordinari.

## L'idea formale, in parole semplici

Hai dei dati osservati e un modello con un parametro θ (theta, il nome generico del parametro da stimare). La likelihood L(θ) è la probabilità dei dati calcolata come se il parametro valesse θ. Attenzione al cambio di prospettiva: i dati sono fissi, è il parametro che varia.

In pratica si usa sempre il **logaritmo** della likelihood, per due motivi concreti:

1. Le probabilità di tanti dati indipendenti si moltiplicano, e un prodotto di cento numeri piccoli va in underflow. Il log trasforma il prodotto in somma, numericamente stabile.
2. Il log non sposta il massimo: il θ migliore resta lo stesso.

Ultimo passo, il ponte verso il deep learning: massimizzare la log likelihood equivale a minimizzare il suo negativo, la **negative log likelihood** (NLL). Quel negativo è una loss, un punteggio di errore. Le loss delle reti neurali nascono quasi tutte così: la MSE è la NLL di un modello con rumore gaussiano, la cross entropy è la NLL di un classificatore. Le vedrai nella lezione 03 e nel modulo 05.

## Esempio numerico a mano

Lancio una moneta di parametro ignoto p per 10 volte: escono 7 teste e 3 croci. La likelihood di una sequenza con 7 teste e 3 croci è

    L(p) = p^7 * (1 - p)^3

Provo due ipotesi:

    L(0.5) = 0.5^7 * 0.5^3 = 0.5^10 ≈ 0.00098
    L(0.7) = 0.7^7 * 0.3^3 ≈ 0.0823 * 0.027 ≈ 0.00222

Sotto p = 0.7 i dati sono circa il doppio più plausibili che sotto p = 0.5. Provando tutti i p, il massimo cade su p = 0.7, cioè 7 su 10, la frequenza osservata. La formula generale conferma: per la moneta, la MLE è sempre teste diviso lanci.

Nota di prudenza da ingegnere: con 10 lanci la stima 0.7 è fragile. La MLE dice qual è il parametro più plausibile, non quanto fidarsi. Servono più dati perché si stabilizzi, come hai visto con la legge dei grandi numeri.

## Riferimenti

* Mathematics for Machine Learning: capitolo 8, sezione 8.3 (parameter estimation e maximum likelihood); capitolo 6, sezione 6.5 per la gaussiana usata nello script.
* Blitzstein, Hwang, Introduction to Probability: capitoli 3 e 4 come base sulle distribuzioni usate qui.

## E adesso

Esegui `python lesson.py`. Calcola L(p) su una griglia di ipotesi e trova il massimo, verifica che il log non sposta il massimo, e poi stima il centro di una gaussiana massimizzando la likelihood con il gradiente, in anteprima sul modulo 04.
