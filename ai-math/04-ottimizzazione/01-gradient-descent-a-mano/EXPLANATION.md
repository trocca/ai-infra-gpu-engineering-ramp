# Gradient descent by hand

## The intuition

You are a ball on the side of a valley, at night, in the fog. You can't see the bottom. You only see the little patch of ground beneath you, and you can feel which way it slopes. The obvious strategy is also the right one: take a step downhill. Then feel the slope again, and take another step. Repeat.

That's gradient descent, all of it. The valley is the loss, the error score as a function of the weights. The slope under your feet is the gradient (module 02). The step is the weight update. The fog is the fact that nobody can afford to compute the loss everywhere: only where you are standing right now.

## The formal idea, in plain words

The update, repeated until convergence:

    new w = w - lr * gradient

The minus sign is there because the gradient points uphill (lesson 02 of module 02) and we want to go down. The number lr is called the **learning rate**: the length of the step. It's the first real hyperparameter you meet, and it dominates everything:

* lr too small: ant steps. You descend, but it takes geological eras.
* lr just right: you descend quickly and settle at the bottom.
* lr too large: you overshoot the bottom and bounce from one side to the other, higher each time. The loss diverges, often all the way to NaN.

When the valley has a single bottom (a **convex** function, like a parabola), gradient descent with a sensible lr reaches the global minimum. The losses of deep networks are not convex — they have secondary valleys — but surprisingly the method works anyway: we'll come back to that in module 05.

## A numerical example by hand

Take loss(w) = (w - 3)², a valley with its bottom at w = 3. The gradient is 2(w - 3). I start from w = 0 with lr = 0.25:

    step 0: gradient = 2*(0 - 3)     = -6      w = 0 - 0.25*(-6)     = 1.5
    step 1: gradient = 2*(1.5 - 3)   = -3      w = 1.5 - 0.25*(-3)   = 2.25
    step 2: gradient = 2*(2.25 - 3)  = -1.5    w = 2.25 - 0.25*(-1.5) = 2.625
    step 3: gradient = 2*(2.625 - 3) = -0.75   w = 2.8125

Every step halves the distance to the bottom: 3, 1.5, 0.75, 0.375. Clean convergence.

Now the pathological case: same starting point, lr = 1.1:

    step 0: gradient = -6      w = 0 - 1.1*(-6)   = 6.6     (overshot the bottom)
    step 1: gradient = 7.2     w = 6.6 - 1.1*7.2  = -1.32   (overshot again, further out)

The oscillations widen with every step: divergence. Same valley, same algorithm, wrong step size.

## References

* Mathematics for Machine Learning: chapter 7, section 7.1 (optimization using gradient descent) and section 7.3 for convexity.
* Prince, Understanding Deep Learning: chapter 6 (fitting models).
* Boyd, Vandenberghe, Convex Optimization: chapters 2 and 3, only if you want the theory of single-bottom valleys.

## What's next

Run `python lesson.py`. It redoes the computations above step by step, then trains the house model with autograd and compares three learning rates: ant, right, and divergent. The three trajectories end up in `figures/`.
