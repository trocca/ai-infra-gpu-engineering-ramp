# Matrices and matrix multiplication

## The intuition

If a vector is a record, a matrix is a table: rows and columns, like a spreadsheet or the output of a query. Our 5-house dataset becomes a matrix with 5 rows (the houses) and 2 columns (square meters, rooms). We say the matrix is 5x2, always rows first, then columns.

Matrix multiplication, matmul for short, looks strange the first time. But it's just this: lots of dot products done in bulk. Each cell of the result is the dot product of a row of the first matrix and a column of the second. That's all. If you understood the dot product in lesson 01, matmul is that same concept in batch mode.

## The formal idea, in plain words

A matrix A of shape m x n has m rows and n columns. The element at row i, column j is written A[i, j].

**Transpose**: written Aᵀ, with a raised T read as "transpose". It swaps rows and columns: row 0 becomes column 0, and so on. A 5x2 matrix transposed becomes 2x5.

**Multiplication** (matmul, `@` operator in Python): to compute C = A @ B, the cell C[i, j] is the dot product of row i of A and column j of B. Shape rule: the columns of A must equal the rows of B. An (m x n) times an (n x p) gives an (m x p). The two inner numbers must match; the two outer ones give the shape of the result.

Why it matters: a linear prediction on one house is a dot product. With matmul you make the prediction on all the houses in one shot: `X @ w`. GPUs exist practically for the sole purpose of doing this as fast as possible.

## Numeric example by hand

Take two 2x2 matrices:

    A = | 1  2 |      B = | 5  6 |
        | 3  4 |          | 7  8 |

Compute C = A @ B cell by cell. Row of A, column of B, dot product:

    C[0,0] = row 0 of A · column 0 of B = 1*5 + 2*7 = 19
    C[0,1] = row 0 of A · column 1 of B = 1*6 + 2*8 = 22
    C[1,0] = row 1 of A · column 0 of B = 3*5 + 4*7 = 43
    C[1,1] = row 1 of A · column 1 of B = 3*6 + 4*8 = 50

    C = | 19  22 |
        | 43  50 |

Watch out: A @ B and B @ A give different results in general. Order matters.

## References

* Mathematics for Machine Learning: chapter 2, section 2.2 (matrices) and section 2.7 (linear mappings).
* Strang, Introduction to Linear Algebra: chapter 2.
* MIT 18.06: lecture 3, dedicated precisely to matrix multiplication.

## What's next

Run `python lesson.py`. You'll see matmul written by hand with three for loops, the comparison against `torch.matmul`, and a speed race between the two versions. Then predictions on all the houses in a single line.
