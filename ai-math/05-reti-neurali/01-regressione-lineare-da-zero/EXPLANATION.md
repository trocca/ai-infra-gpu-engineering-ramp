# Linear regression from scratch

## The intuition

This lesson introduces nothing new. It does something better: it takes every piece you've already built and assembles them into the **training loop**, the four-beat cycle that trains every neural network, from this toy model to GPT:

1. **Forward**: the model makes predictions with the current weights (a matmul, module 01).
2. **Loss**: measure the error score, the MSE (which since module 03 you know is a Gaussian likelihood in disguise).
3. **Backward**: autograd computes the gradient of the loss with respect to every weight (chain rule, module 02).
4. **Update**: one descent step: `w -= lr * grad` (module 04).

Repeat until the loss stops dropping. All of deep learning training, from the first perceptron to the latest LLM, is this same merry-go-round with ever bigger models in the middle.

## The formal idea, in plain words

The model: pred = X @ w + b, with X the matrix of houses (standardized features, module 01 lesson 03), w the two weights, b the bias.

The loss: MSE, the mean of the squared errors.

Two practical details you'll see in the code, both classic sources of PyTorch bugs:

* The update must happen inside `torch.no_grad()`: you're modifying the weights, not computing something to differentiate. Without it, autograd would try to trace the update too.
* After every step you need `grad.zero_()`: PyTorch **accumulates** gradients instead of overwriting them. If you don't zero them, the next step uses the sum of all past gradients. It's the number one beginner bug.

For this model an exact closed-form solution also exists (least squares, `torch.linalg.lstsq`). We'll use it as the acid test: the iterative training must land exactly there. For real networks no closed form exists, and the loop is all there is.

## A numerical example by hand

First turn of the loop, single-house version: x = 1.26 (the fifth house, standardized), y = 350, starting from w = 0, b = 0, lr = 0.1.

    forward:  pred = 0 * 1.26 + 0 = 0
    loss:     (0 - 350)² = 122500
    backward: dloss/dw = 2 * (pred - y) * x = 2 * (-350) * 1.26 = -882
              dloss/db = 2 * (pred - y)     = -700
    update:   w = 0 - 0.1 * (-882) = 88.2
              b = 0 - 0.1 * (-700) = 70

After a single step the model is already much less wrong: pred = 88.2 * 1.26 + 70 ≈ 181. The loop does nothing but repeat this correction, ever more finely.

## References

* Prince, Understanding Deep Learning: chapter 2 (supervised learning).
* Mathematics for Machine Learning: chapter 9 (linear regression), especially the link between least squares and maximum likelihood.

## What's next

Run `python lesson.py`. The complete training loop on the 5-house dataset, the loss collapsing, the final predictions compared against the real prices, and the check against the exact least squares solution. The training curve lands in `figures/`.
