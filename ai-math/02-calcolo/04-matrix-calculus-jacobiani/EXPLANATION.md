# Matrix calculus and Jacobians

## The intuition

So far: input a number, output a number, derivative a number. Then: input a vector, output a number, derivative a vector (the gradient). One case is missing: input a vector, output a vector. What shape does the derivative have?

Think of a system with 2 inputs and 3 outputs. Every output can be sensitive to every input. How many sensitivities are there in total? 3 times 2, that is 6. The derivative is therefore a 3x2 table of slopes: one row per output, one column per input. That table is called the **Jacobian**.

There is no new idea here. Just bookkeeping: lots of partial derivatives, packed into a matrix with the right shapes. Matrix calculus is 90 percent a matter of keeping the shapes straight.

## The formal idea, in plain words

If f takes a vector of n numbers and returns a vector of m numbers, its Jacobian J is an m x n matrix:

    J[i, j] = how much output i feels input j = the partial derivative of f_i with respect to x_j

Special cases you already know:

* m = 1, n = 1: the Jacobian is 1x1, a single number. The derivative of lesson 01.
* m = 1, any n: the Jacobian is a single row. It's the gradient of lesson 02, lying on its side.

The nicest case: if the function is linear, f(x) = W @ x, then the Jacobian is exactly W. The weight matrix IS the sensitivity table. That's why the linear layers of a network are so convenient to differentiate.

And the chain rule? It still works, but the products become matrix products: the Jacobian of a composition is the matmul of the Jacobians. Same rule as lesson 03, the links are now matrices.

## A numerical example by hand

Take f(x, y) = [x², x*y], two inputs and two outputs, at the point (2, 3).

Four partial derivatives, one per cell:

    row 0 (output x²):   df0/dx = 2x = 4     df0/dy = 0
    row 1 (output x*y):  df1/dx = y  = 3     df1/dy = x = 2

    J(2, 3) = | 4  0 |
              | 3  2 |

Reading by rows: the first output feels only x (at rate 4). The second output feels x at rate 3 and y at rate 2.

Practical rule to avoid getting lost with shapes: the Jacobian always has shape (number of outputs) x (number of inputs). If your calculation produces a different shape, something went wrong.

## References

* Mathematics for Machine Learning: chapter 5, section 5.3 (gradients of vector valued functions) and section 5.4 (gradients of matrices).
* Parr, Howard, The Matrix Calculus You Need For Deep Learning: the Jacobian section, which is the centerpiece of the paper.

## What's next

Run `python lesson.py`. It builds the Jacobian of the example in three ways (hand formula, finite differences column by column, `torch.autograd.functional.jacobian`), verifies that for a linear layer the Jacobian is W, and closes with the shapes of the house loss gradients.
