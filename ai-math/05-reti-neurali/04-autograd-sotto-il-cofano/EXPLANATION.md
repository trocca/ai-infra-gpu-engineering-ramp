# Autograd under the hood

## The intuition

For four modules you've called `backward()` and gradients appeared. Today we open the hood. The surprise is how unmysterious it is: autograd is a recorder plus the chain rule.

The right analogy for someone who's been debugging their whole life: **the backward pass is walking a stack trace backwards**. During the forward pass, every operation on a tracked tensor gets recorded into a graph: which inputs went in, which operation ran, how to compute its local derivative. It's the call stack of the computation. When you call `backward()`, PyTorch walks that graph backwards from the loss, multiplying the local slopes link by link: the chain rule from module 02, automated.

You can literally inspect the graph: every intermediate tensor has a `grad_fn` field saying which operation created it, and from there you can climb the whole chain. We'll do it in the script, the way you explore a stack.

## The formal idea, in plain words

Three rules make up the entire mechanism:

1. **Recording**: every operation knows how to compute its own local derivative. Multiplication, addition, ReLU, exp: each one knows only itself. None of them knows anything about the whole model.
2. **Chain rule backwards**: the gradient arriving from above is multiplied by the local derivative and passed to the operation's inputs. You start from the loss (gradient 1 with respect to itself) and descend all the way to the weights.
3. **Accumulation**: if a tensor is used in more than one place in the graph, the gradients arriving from the various branches get **summed**. That's why you need `grad.zero_()` between steps: accumulation is a feature of the graph, not a bug.

When papers say "backpropagation", it's exactly this: forward to compute the values, backward to distribute the slopes. The cost is roughly twice the forward pass alone, regardless of the number of parameters. That efficiency is what makes giant models trainable.

## A numerical example by hand

The chain from module 02 lesson 03, with fresh numbers: x = 2, w = 3, target t = 10.

Forward, saving every intermediate value:

    y    = w * x    = 6
    err  = y - t    = -4
    loss = err²     = 16

Backward, from the loss toward w, multiplying the local derivatives:

    dloss/dloss = 1                        (you always start from 1)
    dloss/derr  = 2 * err       = -8       (local derivative of err²)
    dloss/dy    = -8 * 1        = -8       (local derivative of y - t wrt y)
    dloss/dw    = -8 * x        = -16      (local derivative of w*x wrt w)
    dloss/dx    = -8 * w        = -24      (and, if you want, toward x too)

In the script we'll redo these calculations on a real single-hidden-layer network, formula by formula, and every number will be compared against what `backward()` deposits into the `.grad` fields. They must match to the decimal. They will.

## References

* Prince, Understanding Deep Learning: chapter 7 (gradients and initialization), the sections on the backpropagation algorithm.
* Mathematics for Machine Learning: chapter 5, section 5.6 (backpropagation and automatic differentiation).

## What's next

Run `python lesson.py`, or better still: open it in the debugger. First it explores the graph via `grad_fn` as if it were a stack trace, then it does backpropagation by hand through a real MLP and verifies every single gradient against autograd. After this lesson, `backward()` is no longer a black box.
