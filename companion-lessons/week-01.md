# Week 01 Companion - AI Math Notation, Backprop, and First Rust Contact

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-1-nca-aiio/week-1/plan.md) · [build project](../gpu-engineering-lab/01-foundations/week-01-autograd-from-scratch/README.md)

## Prerequisite Checklist

- You can read scalar, vector, matrix, and tensor notation without panic.
- You can explain a loss function as a number that scores how wrong a model is.
- You can apply the chain rule to a two-step function.
- You have a working Python environment and can run `pytest`.
- You know the Rust words ownership, borrow, reference, `Result`, and `Cargo`, even if you are not fluent yet.

## Mini Lesson

The week is about turning "training" from a vague word into a mechanical loop:

1. compute predictions with a forward pass;
2. compare predictions to labels with a loss;
3. compute gradients with backpropagation;
4. update parameters with gradient descent;
5. repeat over batches.

The useful notation:

```text
y_hat = f(x; theta)
L = loss(y_hat, y)
theta <- theta - eta * grad_theta L
```

`theta` means "all trainable parameters." `eta` is the learning rate. The gradient tells you which direction increases the loss fastest, so the update subtracts it.

## Math Insight

Backprop is just the chain rule with cached intermediate values. For:

```text
z = w * x
L = (z - y)^2
```

the gradient is:

```text
dL/dw = dL/dz * dz/dw
      = 2 * (z - y) * x
```

This tiny example is the whole week in miniature: every bigger network is a graph of these local derivatives.

## Programming Primer

- Python: shape bugs dominate beginner ML code. Print shapes early and use assertions.
- PyTorch: tensors carry both data and shape; autograd records operations when `requires_grad=True`.
- Rust: ownership means one clear owner of memory; borrowing lets code read or mutate without taking ownership; `Result<T, E>` forces errors into the type system.

## 25-Minute Gate

1. By hand, compute `dL/dw` for `x=3`, `w=2`, `y=10`, and `L=(w*x-y)^2`.
2. In Python, create two NumPy arrays and verify matrix multiplication shapes for `(4, 3) @ (3, 2)`.
3. Run one Rust `cargo new scratch_week_01` locally or read the existing Rust hello project if setup is not ready.

If any of these take more than 45 minutes, slow week 1 down: the whole curriculum rests on this mental model.
