# An MLP from scratch, no nn

## The intuition

The models from lessons 01 and 02 have a structural limit: they draw straight boundaries. A line to separate, a plane to predict. But there are problems where no straight line works. The most famous is **XOR**: four points, (0,0) and (1,1) in one class, (0,1) and (1,0) in the other. Try separating them with a line: impossible. It was the argument that froze neural networks for years.

The solution: stacking. A first linear layer invents new features, a bend (the **ReLU**) makes them nonlinear, a second linear layer combines the bent features. The result is an **MLP** (multi layer perceptron), and it can draw boundaries that are jagged, curved, as complicated as you like. This is a neural network. There is nothing else in the basic brick.

## The formal idea, in plain words

The forward pass of a single-hidden-layer MLP:

    h   = ReLU(x @ W1 + b1)      (the hidden layer: new features)
    out = h @ W2 + b2            (the output layer: the combination)

The ReLU (rectified linear unit) is the simplest possible nonlinearity: max(0, z). Below zero it cuts, above zero it lets through. A bend, literally.

Why the bend is indispensable: two linear layers in a row fuse into one (a matmul of a matmul is a matmul, module 01). Without the ReLU in between, the MLP would collapse into a linear regression in disguise. It's the bend that buys the power.

The hidden layer has a size of your choosing: more neurons, more bends available, richer boundaries. In the script we'll use 16, far more than the 2 strictly necessary: the extra width makes training much more reliable, because XOR's loss surface has genuine secondary basins (remember module 04?) and with too few spare bends gradient descent gets trapped in them often.

## A numerical example by hand

Let's hand-build an MLP that solves XOR, with 2 hidden neurons:

    h1 = ReLU(x1 + x2)          (counts how many inputs are on)
    h2 = ReLU(x1 + x2 - 1)      (fires only if BOTH are on)
    out = h1 - 2 * h2

Check on all four cases:

    (0,0): h1 = 0, h2 = 0        out = 0 - 0 = 0    correct
    (1,0): h1 = 1, h2 = ReLU(0) = 0    out = 1 - 0 = 1    correct
    (0,1): h1 = 1, h2 = 0        out = 1        correct
    (1,1): h1 = 2, h2 = 1        out = 2 - 2 = 0    correct

Read what happened: h1 and h2 are invented features ("at least one on", "both on") and the output combines them into "at least one, but not both". Which is the definition of XOR. Trained networks do exactly this, except they discover the features on their own via gradient descent.

## References

* Prince, Understanding Deep Learning: chapter 3 (shallow neural networks) and chapter 4 (deep neural networks).
* Mathematics for Machine Learning: chapter 5, section 5.6, for a preview of how to differentiate a composition of layers.

## What's next

Run `python lesson.py`. First it verifies the hand-built MLP on the XOR truth table, then it trains one from random weights and watches gradient descent rediscover the solution. Raw tensors only: no `nn.Module`, no `nn.Linear`. The decision boundary map lands in `figures/`.
