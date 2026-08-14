# Momentum and Adam

## The intuition

The gradient descent ball has a flaw: it's weightless. At every step it starts from a standstill, looks only at the slope of the moment, and in a long, narrow valley it ends up zigzagging between the walls instead of racing along the floor.

**Momentum** gives it inertia, like a real rolling ball. Each step is a blend of the previous step and the current slope. The sideways zigzags cancel each other out (one kick to the right, one to the left), while the constant component along the valley floor accumulates and picks up speed.

**Adam** adds the second key idea: a step tailored to each parameter. It keeps two running averages for every weight: the average direction of the gradient (like momentum) and the average magnitude of the squared gradient. Then it divides the first by the square root of the second: parameters with habitually huge gradients get cautious steps, those with tiny gradients get amplified ones. It's the default optimizer of modern deep learning.

## The formal idea, in plain words

**Momentum**, with β (beta, typically 0.9) deciding how much inertia to keep:

    v = beta * v + gradient
    w = w - lr * v

v is the accumulated velocity. With beta = 0 you're back to plain gradient descent.

**Adam**, with the standard constants beta1 = 0.9, beta2 = 0.999, eps = 1e-8:

    m = beta1 * m + (1 - beta1) * gradient          (average direction)
    v = beta2 * v + (1 - beta2) * gradient^2        (average magnitude)
    m_hat = m / (1 - beta1^t)                       (correction: m starts at 0)
    v_hat = v / (1 - beta2^t)
    w = w - lr * m_hat / (sqrt(v_hat) + eps)

The two "hat" lines fix a growing pain: m and v start from zero and underestimate everything in the first steps; dividing by (1 - beta^t), with t the step number, puts them back on scale. The eps avoids division by zero. No magic: six lines of arithmetic.

## A numerical example by hand

Momentum with beta = 0.9, lr = 0.1, and a gradient that is always 1 (a constant valley floor):

    step 1: v = 0.9*0 + 1 = 1        update = 0.1 * 1    = 0.1
    step 2: v = 0.9*1 + 1 = 1.9      update = 0.1 * 1.9  = 0.19
    step 3: v = 0.9*1.9 + 1 = 2.71   update = 0.271

The ball accelerates: same gradients, ever longer steps, up to a maximum of 10 times the base step. Now alternating gradients +1, -1, +1 (zigzagging between the walls):

    step 1: v = 1        step 2: v = 0.9*1 - 1 = -0.1      step 3: v = 0.9*(-0.1) + 1 = 0.91

The velocity stays small and oscillates around zero: the zigzags damp themselves out. Same mechanism, two correct behaviors.

## References

* Mathematics for Machine Learning: chapter 7, section 7.1.2 (gradient descent with momentum) and section 7.1 in general.
* Prince, Understanding Deep Learning: chapter 6, the sections on momentum and Adam.

## What's next

Run `python lesson.py`. It implements momentum and Adam from scratch on a narrow valley, compares them with plain gradient descent, and then the acid test: your hand-written Adam and `torch.optim.Adam` must produce exactly the same trajectory, number for number.
