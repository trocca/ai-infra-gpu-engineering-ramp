# The derivative as a slope

## The intuition

Look at any curve on a graph. Now zoom in on one point. Zoom in again. At some point the curve, seen this close, looks like a straight segment. The slope of that segment is the derivative of the function at that point.

Slope means: if I move a tiny bit to the right, how much does the function go up (or down)? A derivative of 6 means that, near that point, moving 0.01 to the right raises the function by about 0.06. Negative derivative: the function goes down. Zero derivative: you're on a flat spot, a valley floor or a peak.

For a debugger this is a familiar concept: it's sensitivity analysis. I poke this input by an epsilon — how much does the output move?

## The formal idea, in plain words

The derivative of f at the point x is written f'(x), read "f prime of x", or df/dx, read "d f over d x", which looks like a fraction on purpose: change in f divided by change in x.

The practical recipe is called a **finite difference**: pick a small step h, and compute

    approximate slope = (f(x + h) - f(x)) / h

The smaller h is, the better the approximation. The true derivative is the value this fraction approaches as h becomes vanishingly small.

For common functions there are exact formulas. The only one we need right away: the derivative of x² is 2x. And one rule: the derivative of a sum is the sum of the derivatives.

PyTorch computes exact derivatives on its own, with a mechanism called **autograd**. You call `loss.backward()` and every tensor involved receives its slope in the `.grad` field. In module 05 we'll pop the hood on this mechanism.

## A numerical example by hand

Take f(x) = x² at the point x = 3, with step h = 0.01:

    f(3)    = 9
    f(3.01) = 9.0601
    approximate slope = (9.0601 - 9) / 0.01 = 6.01

The exact formula says f'(x) = 2x, so f'(3) = 6. The finite difference got extremely close: 6.01 versus 6. With an even smaller h the error shrinks.

Why we care: in our house dataset, the loss (the model's error score) is a function of the weight w. The derivative of the loss with respect to w tells us whether increasing w makes the error worse or better, and by how much. That's the information needed for learning.

## References

* Mathematics for Machine Learning: chapter 5, section 5.1 (differentiation of univariate functions).
* Parr, Howard, The Matrix Calculus You Need For Deep Learning: the introductory section, which starts precisely from scalar derivatives.

## What's next

Run `python lesson.py`. It computes the derivative of x² in three ways (finite difference, exact formula, autograd), checks that they agree, then plots the house loss as a function of the weight w: you'll see a parabola, with the tangent line pointing the way downhill.
