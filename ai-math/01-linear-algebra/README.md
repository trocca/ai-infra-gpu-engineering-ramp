# Module 01: linear algebra

Linear algebra is the language of data. A dataset is a matrix. An example is a vector. A neural network, under the hood, is almost nothing but matrix multiplication. This module builds that language piece by piece.

## Lessons

| Lesson | Topic | What you'll be able to do |
|---|---|---|
| [01-vettori](01-vettori/) | Vectors, addition, dot product | Describe a data point as a list of numbers and make a prediction with a dot product |
| [02-matrici-e-matmul](02-matrici-e-matmul/) | Matrices, transpose, matmul | Make predictions on the whole dataset in one shot with `X @ w` |
| [03-norme-e-distanze](03-norme-e-distanze/) | L1, L2, L-infinity norms, distances, cosine | Measure how similar two data points are, and understand why normalization matters |
| [04-autovalori-svd](04-autovalori-svd/) | Eigenvalues, eigenvectors, SVD | Find the important directions of a matrix and compress it |

Do the lessons in order: each one builds on the concepts of the previous one.

## Book references

* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), the main text:
  * Chapter 2, Linear Algebra: sections 2.1 through 2.4 for lessons 01 and 02.
  * Chapter 3, Analytic Geometry: sections 3.1 through 3.4 for lesson 03.
  * Chapter 4, Matrix Decompositions: sections 4.2, 4.4 and 4.5 for lesson 04.
* **Introduction to Linear Algebra** (Strang), supporting text:
  * Chapter 1 for vectors, chapter 2 for matrices, chapters 6 and 7 for eigenvalues and the SVD.
* **MIT 18.06** (Strang's video lectures), supporting material:
  * Lecture 1 to get started, lecture 3 for matrix multiplication, lecture 21 for eigenvalues, lecture 29 for the SVD.

## Estimated time

About 2 weeks at a pace of 4 to 5 hours per week. The first two lessons go quickly. The last two deserve more time, especially the SVD.

## The common thread

This module is where the 5-house dataset first appears: for each house we know square meters, number of rooms, and price. Here we'll use it as a matrix. In later modules we'll take derivatives on it, run gradient descent on it, and train a neural network on it. Same data, increasingly powerful tools.
