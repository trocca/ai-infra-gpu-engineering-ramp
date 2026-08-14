"""Exercises for lesson 01: gradient descent by hand.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def passo_gd(w, grad, lr):
    """Apply a single gradient descent step.

    The update rule: new w = w - lr * grad.
    w and grad may be floats or tensors of the same shape.
    Return the new w.
    """
    # TODO
    raise NotImplementedError


def gd_su_parabola(w0, lr, passi):
    """Run gradient descent on the parabola loss(w) = (w - 3)^2.

    The exact gradient is 2 * (w - 3): write it by hand, no
    autograd. Start from w0 (a float) and apply the rule for the
    requested number of steps. Return the final w as a float.
    With lr = 0.25 and enough steps it must get close to 3.
    """
    # TODO
    raise NotImplementedError


def gd_autograd(f, w0, lr, passi):
    """Generic gradient descent on any function, with autograd.

    f takes a tensor and returns a scalar tensor. w0 is the starting
    tensor (without requires_grad). The loop for each step:
    1. clone the current point with requires_grad_(True)
    2. compute f and call backward
    3. update: new point = point - lr * gradient (use .detach())
    Return the final point (a tensor without requires_grad).
    """
    # TODO
    raise NotImplementedError


def gd_con_storia(f, w0, lr, passi):
    """Like gd_autograd, but also record the loss history.

    Return a tuple (final_point, history) where history is the list
    of f values (floats) at every step, BEFORE updating.
    The history is what you watch to tell whether training is working:
    it must go down. If it goes up, the learning rate is too large.
    """
    # TODO
    raise NotImplementedError
