# The loss landscape

## The intuition

For the whole module the ball has been descending in the fog. In this lesson the fog lifts. With a model of only two parameters (a weight w and a bias b) we can afford the luxury forbidden to large models: computing the loss EVERYWHERE, on a grid of (w, b) combinations, and looking at the whole valley from above.

The tool is the topographic map: **contour lines**. Each curve joins the points with the same loss, like the elevation lines on a hiking map. Tightly packed curves mean a steep slope, sparse curves mean a plain. The center of the rings is the bottom of the valley: the perfect (w, b) combination.

On top of this map we'll draw the trajectories of the algorithms from the previous lessons. You'll see gradient descent cutting the contour lines always at right angles (the gradient is orthogonal to the contour lines), zigzagging where the valley is narrow, and momentum sailing straight along the valley floor.

## The formal idea, in plain words

The loss surface (loss landscape) of the house model is the function

    loss(w, b) = MSE of the prices with weight w and bias b

seen as a landscape: two horizontal coordinates (w and b), with the loss as the altitude. To draw it, you evaluate the loss on a grid of points and trace the contour lines.

Two properties of our valley:

1. It is **convex**: the MSE of a linear model is a parabola in every direction, so there is a single bottom, no secondary hollows. That's why linear regression always converges.
2. Its **shape depends on the units of the data**. With unnormalized features the valley is an extremely narrow, slanted canyon, and gradient descent suffers. With standardized features (module 01, lesson 03) the valley is nearly circular and the descent flows. Same math as the GD-versus-momentum comparison of lesson 03.

The deep networks of module 05 have non-convex landscapes, with millions of dimensions, that nobody can draw in full. But the intuition built here in two dimensions is the one researchers actually use to reason about them.

## A numerical example by hand

Stripped-down version: a single house, x = 1 (normalized feature), price y = 250. The loss is

    loss(w, b) = (w * 1 + b - 250)²

I compute the altitude at three points of the map:

    (w=100, b=100):  (100 + 100 - 250)² = (-50)²  = 2500
    (w=200, b=50):   (200 + 50 - 250)²  = 0       (valley floor)
    (w=250, b=50):   (250 + 50 - 250)²  = 2500

Note: (w=100, b=150) also gives zero. With a single house the bottom is not a point but an entire line of solutions: not enough data to pin down two parameters. With the 5-house dataset of the script the bottom goes back to being a single point.

## References

* Mathematics for Machine Learning: chapter 7, section 7.1 for the descent and section 7.3 for convexity.
* Prince, Understanding Deep Learning: chapter 6, where loss landscapes are drawn exactly like this.

## What's next

Run `python lesson.py`. It builds the grid, checks that the minimum of the map coincides with the exact least squares solution, and saves in `figures/` the valley with the GD and momentum trajectories on top, fog lifted.
