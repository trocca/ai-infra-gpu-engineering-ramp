# Likelihood and maximum likelihood

## The intuition

You're working an escalation: you have a log of events (the data) and several hypotheses about what produced them (the parameters). Some hypotheses make that log plausible; others would make it a miracle. The **likelihood** measures exactly this: how probable it is to observe precisely the data you have, if the hypothesis were true.

**Maximum likelihood estimation** (MLE) is the most natural rule in the world: among all hypotheses, pick the one under which your data is least miraculous. It doesn't guarantee the truth. It picks the explanation that makes the facts most ordinary.

## The formal idea, in plain words

You have observed data and a model with a parameter θ (theta, the generic name for the parameter to estimate). The likelihood L(θ) is the probability of the data computed as if the parameter were θ. Watch out for the change of perspective: the data is fixed, it's the parameter that varies.

In practice you always work with the **logarithm** of the likelihood, for two concrete reasons:

1. The probabilities of many independent data points multiply, and a product of a hundred small numbers underflows. The log turns the product into a sum, which is numerically stable.
2. The log doesn't move the maximum: the best θ stays the same.

Last step, the bridge to deep learning: maximizing the log likelihood is equivalent to minimizing its negative, the **negative log likelihood** (NLL). That negative is a loss, an error score. Nearly all neural network losses are born this way: MSE is the NLL of a model with Gaussian noise, cross entropy is the NLL of a classifier. You'll see them in lesson 03 and in module 05.

## Worked example by hand

I flip a coin with unknown parameter p 10 times: 7 heads and 3 tails come up. The likelihood of a sequence with 7 heads and 3 tails is

    L(p) = p^7 * (1 - p)^3

Let's try two hypotheses:

    L(0.5) = 0.5^7 * 0.5^3 = 0.5^10 ≈ 0.00098
    L(0.7) = 0.7^7 * 0.3^3 ≈ 0.0823 * 0.027 ≈ 0.00222

Under p = 0.7 the data is roughly twice as plausible as under p = 0.5. Trying every p, the maximum lands on p = 0.7, that is 7 out of 10, the observed frequency. The general formula confirms it: for the coin, the MLE is always heads divided by flips.

An engineer's note of caution: with 10 flips the estimate 0.7 is fragile. The MLE tells you which parameter is most plausible, not how much to trust it. It takes more data for it to stabilize, as you saw with the law of large numbers.

## References

* Mathematics for Machine Learning: chapter 8, section 8.3 (parameter estimation and maximum likelihood); chapter 6, section 6.5 for the Gaussian used in the script.
* Blitzstein, Hwang, Introduction to Probability: chapters 3 and 4 as background on the distributions used here.

## What's next

Run `python lesson.py`. It computes L(p) on a grid of hypotheses and finds the maximum, verifies that the log doesn't move the maximum, and then estimates the center of a Gaussian by maximizing the likelihood with the gradient — a preview of module 04.
