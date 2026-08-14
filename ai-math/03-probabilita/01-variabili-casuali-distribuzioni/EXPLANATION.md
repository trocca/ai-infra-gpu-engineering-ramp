# Random variables and distributions

## The intuition

A random variable is a number produced by a process you don't fully control. The roll of a die. The response time of a server under load. The noise in a measurement. You don't know which value will come out next time, but the process isn't pure chaos: some values come out more often than others.

The **distribution** is the process's ID card: the list of possible values with the probability of each one. Knowing the distribution doesn't tell you the next value, but it tells you everything about the long-run behavior.

## The formal idea, in plain words

A discrete random variable has a finite list of possible values, each with its own probability. The probabilities are numbers between 0 and 1 and sum to 1. A continuous variable (like noise) can take any value in an interval, and the distribution becomes a density curve.

Three distributions you'll run into everywhere:

* **Bernoulli(p)**: equals 1 with probability p, 0 otherwise. A coin, a click, a bit.
* **Uniform**: every value in an interval is equally likely. The fair die.
* **Normal (Gaussian)**: the bell curve. It describes measurement noise and a thousand other things. It has two parameters: μ (mu, the center) and σ (sigma, the width).

The **expected value** E[X] (read "expected value of X") is the weighted average of the values, each weighted by its probability. It's the number the average of many repetitions settles around. The **variance** measures how much the values bounce around the expected value.

Key fact, almost magical but verifiable on a computer: the average of many samples converges to the expected value. It's called the law of large numbers, and you'll watch it happen before your eyes in the script.

## Worked example by hand

Fair die, values 1 through 6, each with probability 1/6:

    E[X] = 1*(1/6) + 2*(1/6) + 3*(1/6) + 4*(1/6) + 5*(1/6) + 6*(1/6)
         = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 21 / 6 = 3.5

Note: 3.5 isn't even a possible value of the die. The expected value isn't "the typical value" — it's the center of mass.

Biased Bernoulli coin with p = 0.7:

    E[X] = 1 * 0.7 + 0 * 0.3 = 0.7
    variance = p * (1 - p) = 0.7 * 0.3 = 0.21

## References

* Mathematics for Machine Learning: chapter 6, sections 6.1 and 6.2 for probability and distributions, section 6.4 for mean and variance, section 6.5 for the Gaussian.
* Blitzstein, Hwang, Introduction to Probability: chapter 3 (random variables), chapter 4 (expectation), chapter 5 (continuous random variables).

## What's next

Run `python lesson.py`. It simulates dice, biased coins and Gaussians with `torch.distributions`, verifies that the empirical means converge to the expected values computed by hand, and saves to `figures/` the law of large numbers in action and the histogram of the bell curve.
