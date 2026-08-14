# Vectors

## The intuition

Think of a record in a structured log: an event with numeric fields inside, always in the same order. A vector is exactly that: an ordered list of numbers. Nothing more.

A house from our dataset, for example, is described by two numbers: square meters and number of rooms. The house "50 square meters, 2 rooms" becomes the vector `[50, 2]`. Order matters: position 0 is always the square meters, position 1 is always the rooms. Like the fields of a struct.

A vector of 2 numbers can also be drawn: it's an arrow that starts at the origin and ends at the point with those coordinates. With 3 numbers it's an arrow in space. With 100 numbers you can't draw it anymore, but the rules of computation stay exactly the same.

## The formal idea, in plain words

A vector of dimension n is an ordered list of n numbers. It's written **v** in bold, or v with a small arrow on top. There are three fundamental operations:

1. **Addition**: add the numbers position by position. The two vectors must have the same dimension.
2. **Scalar multiplication**: a scalar is a single number. Multiply every element of the vector by that number. The arrow gets longer or shorter; the direction doesn't change.
3. **Dot product** (symbol `·`, a small dot): multiply the numbers position by position, then add everything up. The result is a single number, not a vector.

The dot product is the single most important operation of this whole journey. A linear prediction, the heart of every neural network, is a dot product: data times weights, all summed up.

## Numeric example by hand

Take `u = [2, 1]` and `v = [1, 3]`.

Addition, position by position:

    u + v = [2 + 1, 1 + 3] = [3, 4]

Multiplication by the scalar 2:

    2 * u = [2 * 2, 2 * 1] = [4, 2]

Dot product, multiply then sum:

    u · v = 2 * 1 + 1 * 3 = 2 + 3 = 5

That's all there is to it. If you can do these three operations by hand on small vectors, you can already read half the formulas in deep learning.

## References

* Mathematics for Machine Learning: chapter 2, section 2.4 (vector spaces); the dot product is in chapter 3, section 3.2 (inner products).
* Strang, Introduction to Linear Algebra: chapter 1.
* MIT 18.06: lecture 1.

## What's next

Open `lesson.py` and run it with `python lesson.py`. It redoes these same computations in PyTorch, printing every value, and at the end uses a dot product to make the first prediction of a house price.
