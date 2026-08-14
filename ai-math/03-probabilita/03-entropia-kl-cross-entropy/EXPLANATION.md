# Entropy, KL divergence and cross entropy

## The intuition

Start from **surprise**. A rare event, when it happens, is very surprising. A near-certain event surprises nobody. The mathematical measure is -log p: probability 1 gives surprise 0, small probability gives large surprise. We'll use the natural logarithm, like PyTorch does.

The **entropy** of a distribution is the average surprise: how much, on average, the next sample surprises you. A fair coin is maximally unpredictable: high entropy. A coin that almost always lands heads is boring: low entropy.

The **KL divergence** answers: how much does it cost to use the wrong model? If the world follows distribution P but you reason with distribution Q, you suffer extra surprise compared to someone who knows P. The KL measures that average extra surprise. It's zero only if Q is identical to P.

The **cross entropy** is the total average surprise you experience using Q in a world governed by P: the unavoidable surprise (the entropy of P) plus the extra surprise for the wrong model (the KL).

## The formal idea, in plain words

For discrete distributions, summing over all possible values:

    surprise of an event:   -log p
    entropy:                H(P)    = sum of p * (-log p)
    KL divergence:          KL(P‖Q) = sum of p * (log p - log q)
    cross entropy:          H(P,Q)  = sum of p * (-log q) = H(P) + KL(P‖Q)

The ‖ symbol in the KL is just a separator: it reads "KL from P to Q". Careful: it's not symmetric — KL(P‖Q) and KL(Q‖P) generally differ.

Why this is the heart of classification: the truth of an example is a distribution entirely concentrated on the right class (probability 1 there, 0 elsewhere). In that case the cross entropy collapses into a single term: **minus the log of the probability the model gives to the right class**. Confident and correct model: loss near zero. Confident and wrong model: enormous loss. Minimizing the cross entropy is the MLE from lesson 02 applied to a classifier.

## Worked example by hand

Entropy of a fair coin, p = [0.5, 0.5]:

    H = 0.5 * (-log 0.5) + 0.5 * (-log 0.5) = log 2 ≈ 0.693

Loaded coin, p = [0.9, 0.1]:

    H = 0.9 * 0.105 + 0.1 * 2.303 ≈ 0.095 + 0.230 = 0.325

Less than half: more predictable, less average surprise.

Cross entropy of a classifier: the truth is class 0, the model says q = [0.8, 0.2]:

    CE = -log 0.8 ≈ 0.223

If the model had said q = [0.2, 0.8], that is confident in the wrong class:

    CE = -log 0.2 ≈ 1.609

Seven times worse. Cross entropy ferociously punishes misplaced confidence.

## References

* Prince, Understanding Deep Learning: chapter 5 (loss functions), where cross entropy and likelihood are joined exactly as here.
* Mathematics for Machine Learning: chapter 6 for the distributions these concepts rest on.

## What's next

Run `python lesson.py`. It computes entropy, KL and cross entropy by hand and compares them against `F.kl_div`, `F.softmax` and `F.cross_entropy`, including a softmax written from scratch. It closes with the curve of the coin's entropy as p varies, saved to `figures/`.
