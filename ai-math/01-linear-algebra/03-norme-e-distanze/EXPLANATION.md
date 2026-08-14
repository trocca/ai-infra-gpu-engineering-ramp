# Norms and distances

## The intuition

How "big" is a vector? It depends on how you measure. Imagine you have to get from home to a spot in the city that's 3 blocks east and 4 blocks north.

* By taxi, weaving between the buildings, you travel 3 + 4 = 7 blocks. This is the **L1 norm**, also known as the Manhattan distance.
* By air, as the crow flies, you travel 5 blocks (the Pythagorean theorem from high school). This is the **L2 norm**, the classic notion of length.
* If you only care about the worst leg of the trip, the maximum of 3 and 4 is 4. This is the **L-infinity norm**.

Same vector, three different measurements. None of them is "the right one": they're different tools. In machine learning you'll use them all: L2 for distances and for the loss, L1 to make models simpler, L-infinity for worst cases.

## The formal idea, in plain words

The norm of a vector v is written ‖v‖, two vertical bars on each side, and read "norm of v". It's a number that measures the length of the vector.

* **L1 norm**: sum of the absolute values of every element.
* **L2 norm**: square root of the sum of the squares. When there's no subscript, ‖v‖ almost always means L2.
* **L-infinity norm**: the maximum of the absolute values.

The **distance** between two vectors u and v is the norm of their difference: ‖u − v‖. First subtract, then measure the length of what's left.

**Cosine similarity** instead measures the angle between two vectors: dot product divided by the product of the two L2 norms. It's 1 if they point in the same direction, 0 if they're perpendicular, −1 if they're opposite. It ignores length and looks only at direction. It's the standard measure for comparing embeddings.

## Numeric example by hand

Take v = [3, 4].

    L1:  |3| + |4| = 7
    L2:  square root of (3*3 + 4*4) = square root of 25 = 5
    Linf: max(|3|, |4|) = 4

L2 distance between u = [1, 1] and v = [4, 5]:

    u - v = [1 - 4, 1 - 5] = [-3, -4]
    distance = square root of (9 + 16) = square root of 25 = 5

A practical trap you'll see in the script: if you measure the distance between two houses using the raw data, the square meters (big numbers) completely crush the rooms (small numbers). The distance only says "who has more square meters". The fix is to normalize the columns before measuring. This exact same problem will come back when we train networks.

## References

* Mathematics for Machine Learning: chapter 3, sections 3.1 (norms), 3.2 (inner products), 3.3 (lengths and distances) and 3.4 (angles and orthogonality).
* Strang, Introduction to Linear Algebra: chapter 1, where lengths and the dot product are introduced together.

## What's next

Run `python lesson.py`. It computes the three norms by hand and with `torch.linalg.norm`, shows the scale trap on the house dataset, and saves to `figures/` a plot of the "unit balls": the shape of all the points at distance 1 from the origin, according to each norm.
