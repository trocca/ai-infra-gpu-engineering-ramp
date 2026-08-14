# Partial derivatives and the gradient

## The intuition

In lesson 01 the function had a single knob. But a real model has many: our house model has one weight for the square meters, one for the rooms, one bias. The loss depends on all of them at once.

How do you measure slope when there are many knobs? One at a time. Freeze every knob except one, nudge that one a tiny bit, watch how much the result changes. That is the **partial derivative** with respect to that knob. Then repeat for each one.

The **gradient** is simply the complete list: one partial derivative per knob, packed into a vector. If you're standing on a hill, the gradient is the arrow pointing up the steepest climb. Which means its opposite points down the steepest descent: that's the direction we'll use for learning.

## The formal idea, in plain words

The partial derivative of f with respect to x is written ∂f/∂x. The symbol ∂ is a "rounded d" and is read like an ordinary d: it reminds you that the other variables are held fixed.

The gradient is written ∇f, where ∇ (read "nabla") denotes the gradient, i.e. the list of slopes:

    ∇f = [∂f/∂x, ∂f/∂y, ...]

It's a vector with as many components as there are variables. Two facts to remember:

1. The gradient points in the direction of **steepest ascent** of f.
2. Its length tells you how steep that climb is.

In PyTorch nothing changes compared to lesson 01: put the variables in a tensor with `requires_grad=True`, compute f, call `backward()`, and in `.grad` you find the entire gradient in one shot. Autograd does the partial derivatives for you.

## A numerical example by hand

Take f(x, y) = x² + 3y at the point (2, 1).

Partial derivative with respect to x: I hold y fixed, so the 3y term is a constant and vanishes. What remains is the derivative of x², which is 2x:

    ∂f/∂x = 2x = 4

Partial derivative with respect to y: I hold x fixed, so x² vanishes. The derivative of 3y is 3:

    ∂f/∂y = 3

The gradient at the point (2, 1) is therefore:

    ∇f(2, 1) = [4, 3]

Reading it: nudging x up makes f rise at rate 4, nudging y up makes it rise at rate 3. And the direction [4, 3], taken as a whole, is the fastest way up from that point.

## References

* Mathematics for Machine Learning: chapter 5, section 5.2 (partial differentiation and gradients).
* Parr, Howard, The Matrix Calculus You Need For Deep Learning: the section on partial derivatives and the gradient.

## What's next

Run `python lesson.py`. It computes the partial derivatives by hand and with autograd, then draws the gradient field: a map of arrows that, at every point, show the way uphill. Finally it measures the gradient of the house loss with respect to weight and bias together.
