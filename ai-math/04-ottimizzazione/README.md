# Module 04: optimization

This is where everything you have built gets set in motion. You know how to measure the error (the loss), and you know how to compute the slope (the gradient). Optimizing means using those slopes to descend toward the minimum error, one step at a time. It is the engine behind training any neural network, from a two-weight model to an LLM.

The metaphor that runs through the whole module: a ball rolling down a valley in the fog. It can't see the whole valley, it only feels the slope under itself. And yet, step after step, it finds the bottom.

## Lessons

| Lesson | Topic | What you'll be able to do |
|---|---|---|
| [01-gradient-descent-a-mano](01-gradient-descent-a-mano/) | Gradient descent, learning rate | Write the descent loop by hand and pick the right step size |
| [02-sgd-minibatch](02-sgd-minibatch/) | SGD, minibatches, epochs | Train on data in chunks, understanding the noise that comes with it |
| [03-momentum-adam](03-momentum-adam/) | Momentum, Adam | Implement the two most widely used optimizers and verify them against torch.optim |
| [04-loss-landscape](04-loss-landscape/) | Loss surfaces, trajectories | Draw the whole valley and watch the ball roll down |

The lessons should be done in order: each one adds a piece to the training loop of the previous one.

## Book references

* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), the main text:
  * Chapter 7, Continuous Optimization: section 7.1 for gradient descent, momentum and SGD, section 7.3 for the idea of convexity.
* **Convex Optimization** (Boyd, Vandenberghe), supporting material and purely for the curious:
  * Chapter 2 (convex sets) and chapter 3 (convex functions), for anyone who wants to see the theory of valleys with a single bottom.
* **Understanding Deep Learning** (Prince), supporting material:
  * Chapter 6, Fitting Models: gradient descent, SGD, momentum and Adam, told with excellent figures.

## Estimated time

About 2 weeks at 4 or 5 hours a week. Lesson 03 is the densest: implementing Adam by hand and watching it match `torch.optim.Adam` repays every minute spent.

## The common thread

The 5-house dataset returns as the protagonist: the linear model that in module 02 could only measure its own slope actually learns here. In lesson 04 you'll see the entire valley of its loss and the descent trajectory drawn on top of it, fog lifted.
