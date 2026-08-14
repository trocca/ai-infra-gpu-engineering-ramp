# The chain rule

## The intuition

You exchange currency twice: euros to dollars, then dollars to yen. If one euro is worth 1.1 dollars and one dollar is worth 150 yen, how much is one euro in yen? The two conversions multiply: 1.1 * 150 = 165 yen.

The chain rule says derivatives work the same way. If x influences u, and u influences y, then the sensitivity of y to x is the product of the two sensitivities along the chain:

    how much y feels x = (how much y feels u) * (how much u feels x)

A neural network is one very long chain: input, first layer, second layer, prediction, loss. To know how much the loss feels a weight buried at the start, you multiply the slopes link by link, backwards. That process has a famous name: backpropagation. The chain rule is all there is inside it.

## The formal idea, in plain words

If y = f(g(x)), that is, you apply g first and then f, the derivative of the composition is:

    dy/dx = f'(g(x)) * g'(x)

In words: derivative of the outer function, evaluated at the inner value, times the derivative of the inner function. The fraction notation makes the idea almost obvious:

    dy/dx = dy/du * du/dx

where u = g(x) is the intermediate value. The "du"s appear to cancel like in a fraction. That's not a proof, but it's a great way to remember it. And the chain can be as long as you like: three, ten, a hundred links. You multiply everything.

## A numerical example by hand

Take y = (3x + 1)² at the point x = 2. The chain is: u = 3x + 1 (inner), y = u² (outer).

Step 1, the inner value:

    u = 3 * 2 + 1 = 7

Step 2, the two slopes separately:

    du/dx = 3           (derivative of 3x + 1)
    dy/du = 2u = 14     (derivative of u^2, evaluated at u = 7)

Step 3, multiply the links:

    dy/dx = 14 * 3 = 42

Independent check: expanding, y = 9x² + 6x + 1, whose derivative is 18x + 6, which at x = 2 equals 42. It matches.

Now the same idea on the house model, with a single house. The chain is:

    pred = w * x        (prediction)
    err  = pred - y     (error)
    loss = err²         (error score)

The slopes of the links: dloss/derr = 2 * err, derr/dpred = 1, dpred/dw = x. So:

    dloss/dw = 2 * err * 1 * x

This little formula, generalized, is what `backward()` computes for every weight of a network.

## References

* Mathematics for Machine Learning: chapter 5, section 5.1.2 (differentiation rules) and section 5.2.2 (chain rule).
* Parr, Howard, The Matrix Calculus You Need For Deep Learning: the chain rule sections, the heart of the paper.

## What's next

Run `python lesson.py`. It redoes the chain of (3x + 1)² number by number, then takes apart the loss chain of a single house and compares every link computed by hand against what autograd finds. They're identical.
