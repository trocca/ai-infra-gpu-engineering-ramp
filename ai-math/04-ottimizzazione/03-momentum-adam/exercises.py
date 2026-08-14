"""Exercises for lesson 03: momentum and Adam.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def passo_momentum(w, v, grad, lr, beta):
    """Apply one gradient descent step with momentum.

    Formulas (the same as torch.optim.SGD with momentum):
    new v = beta * v + grad
    new w = w - lr * new v
    Return the tuple (w_new, v_new).
    """
    # TODO
    raise NotImplementedError


def velocita_a_regime(lr, beta):
    """Compute the effective steady-state step with constant gradient 1.

    With a gradient always equal to 1, the velocity v converges to a
    geometric sum: 1 + beta + beta^2 + ... = 1 / (1 - beta).
    The effective steady-state step is therefore lr / (1 - beta).
    Return that number as a float. With beta = 0.9 momentum
    amplifies the step 10 times: that's why it accelerates.
    """
    # TODO
    raise NotImplementedError


def passo_adam(w, m, v, grad, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Apply one Adam step. t is the step number, starting from 1.

    The six lines from EXPLANATION.md:
    new m = beta1 * m + (1 - beta1) * grad
    new v = beta2 * v + (1 - beta2) * grad^2
    m_hat = new m / (1 - beta1^t)
    v_hat = new v / (1 - beta2^t)
    new w = w - lr * m_hat / (sqrt(v_hat) + eps)
    Return the tuple (w_new, m_new, v_new).
    The test compares your implementation against torch.optim.Adam.
    """
    # TODO
    raise NotImplementedError


def allena_con_adam(f, w0, lr, passi):
    """Minimize f starting from w0 using YOUR passo_adam.

    f takes a tensor and returns a scalar tensor. For each step
    t from 1 to passi: compute the gradient of f (with autograd, as in
    the previous lessons), then update with passo_adam.
    m and v start as zeros with the same shape as w0.
    Return the final point (a tensor without requires_grad).
    """
    # TODO
    raise NotImplementedError
