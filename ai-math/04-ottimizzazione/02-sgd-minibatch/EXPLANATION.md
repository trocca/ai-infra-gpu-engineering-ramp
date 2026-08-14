# SGD and minibatches

## The intuition

The gradient descent of lesson 01 has a hidden cost: to take ONE step it must compute the error on the ENTIRE dataset. With 5 houses that's free. With 10 million images, every single step would cost a full scan of the data. Unworkable.

The lifesaving idea: you don't need the exact slope, an honest estimate is enough. Grab a handful of examples at random, a **minibatch**, and compute the slope only on those. It's noisy: each handful says something slightly different. But on average it points in the right direction, and it costs almost nothing. This is **stochastic gradient descent**, SGD.

The ball in the fog is now also a little drunk: every step points in a slightly wrong direction. But the steps are many, they're cheap, and the errors cancel out. And there's an unexpected bonus: that noise helps it avoid getting trapped in secondary hollows of the valley.

## The formal idea, in plain words

Training vocabulary you'll find in every paper and every training log:

* **batch size**: how many examples go into a minibatch. Typical values: 32, 64, 256.
* **epoch**: one complete pass over the whole dataset, one minibatch at a time.
* **shuffle**: at every epoch the order of the examples is reshuffled, so the minibatches change composition and the noise never repeats the same way.

The full loop: shuffle the data, cut it into minibatches, and for each minibatch compute the loss only on that data, then backward and update. Out of minibatches: an epoch has passed. Start over.

The minibatch gradient is an **unbiased estimate** of the true one: on average, over many draws, it matches the gradient computed on the whole dataset. It's the same pact as lesson 01 of module 03: the empirical mean converges to the expected value.

## A numerical example by hand

One-weight model, pred = w * x, with w = 1, and three examples:

    x = [1, 2, 3]    y = [2, 4, 6]    (true rule: y = 2x)

Gradient for a single example: 2 * (w*x - y) * x. With w = 1 the errors are -1, -2, -3, so:

    example 1: 2 * (-1) * 1 = -2
    example 2: 2 * (-2) * 2 = -8
    example 3: 2 * (-3) * 3 = -18

Exact gradient (mean over all): (-2 - 8 - 18) / 3 = -9.33.

Minibatches of 2 examples:

    batch {1, 2}: (-2 - 8) / 2  = -5
    batch {1, 3}: (-2 - 18) / 2 = -10
    batch {2, 3}: (-8 - 18) / 2 = -13

None of them equals exactly -9.33, all have the right sign, and their mean (-5 - 10 - 13) / 3 = -9.33 is precisely the exact gradient. Noise yes, bias no.

## References

* Mathematics for Machine Learning: chapter 7, the part of section 7.1 devoted to stochastic gradient descent.
* Prince, Understanding Deep Learning: chapter 6, the sections on SGD and minibatches.

## What's next

Run `python lesson.py`. It generates 200 synthetic houses with the same rule as our 5, compares the exact gradient with the minibatch ones, and trains the model with real SGD: shuffle, epochs, minibatches. The loss curves end up in `figures/`.
