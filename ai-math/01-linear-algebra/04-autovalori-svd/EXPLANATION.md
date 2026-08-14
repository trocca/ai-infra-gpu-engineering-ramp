# Eigenvalues, eigenvectors and SVD

## The intuition

A square matrix, multiplied by a vector, transforms it: it rotates it, stretches it, squashes it. Try multiplying the same matrix by lots of different vectors: almost all of them change direction.

But almost every matrix has some special directions. If you place a vector along one of them, the matrix doesn't rotate it: it only stretches or shrinks it, keeping it on its own line. Those directions are called **eigenvectors**. The stretching factor is called the **eigenvalue**. It's like finding the "natural axis" of a system: the direction along which the behavior becomes simple.

The **SVD** (Singular Value Decomposition) generalizes the idea to any matrix, even a rectangular one like our 5x2 dataset. It says: every matrix, no matter how complicated, is a sequence of three simple operations. A rotation, a stretch along the axes, another rotation. Always. And it tells you which directions matter most: it's the foundation of compression and of PCA.

## The formal idea, in plain words

**Eigenvectors and eigenvalues**: v is an eigenvector of A, with eigenvalue λ (lambda, a Greek letter that here is just a number), if

    A @ v = λ * v

Applying the matrix to v is the same as multiplying it by a number. Direction unchanged, only the scale.

**Eigendecomposition**: a symmetric matrix (equal to its transpose) can be rewritten as Q @ Λ @ Qᵀ, where Q has the eigenvectors in its columns and Λ (capital lambda) is diagonal with the eigenvalues. In PyTorch, `torch.linalg.eigh` computes it.

**SVD**: any matrix A can be rewritten as U @ S @ Vᵀ. U and V contain directions (orthogonal to each other), S is diagonal and contains the **singular values**, positive numbers in descending order. The first singular value marks the direction along which the matrix "carries the most information". By keeping only the first k values you get the best rank-k approximation: same matrix, fewer numbers, minimal loss.

## Numeric example by hand

Take the symmetric matrix

    A = | 2  1 |
        | 1  2 |

Try v = [1, 1]:

    A @ v = [2*1 + 1*1, 1*1 + 2*1] = [3, 3] = 3 * [1, 1]

Identical direction, length times 3. So [1, 1] is an eigenvector with eigenvalue 3.

Try v = [1, -1]:

    A @ v = [2*1 + 1*(-1), 1*1 + 2*(-1)] = [1, -1] = 1 * [1, -1]

An eigenvector with eigenvalue 1. Now try v = [1, 0] instead:

    A @ v = [2, 1]

Direction changed: not an eigenvector. The special directions really are special.

## References

* Mathematics for Machine Learning: chapter 4, sections 4.2 (eigenvalues and eigenvectors), 4.4 (eigendecomposition and diagonalization) and 4.5 (singular value decomposition).
* Strang, Introduction to Linear Algebra: chapter 6 for eigenvalues, chapter 7 for the SVD.
* MIT 18.06: lecture 21 for eigenvalues, lecture 29 for the SVD.

## What's next

Run `python lesson.py`. It verifies the eigenvectors above by hand, rebuilds A from its pieces, then runs the SVD on the house dataset and shows how well it compresses to rank 1.
