# Module 02: calculus

Differential calculus answers a single question, asked in a thousand different forms: if I nudge this knob a tiny bit, how much does the result change? That "sensitivity" is called the derivative. Neural networks learn exactly this way: they measure how much each weight moves the error, and turn the knobs in the right direction.

## Lessons

| Lesson | Topic | What you'll be able to do |
|---|---|---|
| [01-derivata-come-pendenza](01-derivata-come-pendenza/) | Derivative, finite differences, autograd | Compute the slope of a function in three different ways and check that they agree |
| [02-derivate-parziali-gradiente](02-derivate-parziali-gradiente/) | Partial derivatives, gradient | Measure the slope with respect to each knob, and read ∇ without fear |
| [03-chain-rule](03-chain-rule/) | Chain rule | Differentiate composed functions by multiplying slopes link by link |
| [04-matrix-calculus-jacobiani](04-matrix-calculus-jacobiani/) | Jacobians, gradients of vectors | Keep derivative shapes straight when inputs and outputs are vectors |

Do the lessons in order: the chain rule in lesson 03 is the heart of backpropagation, and lesson 04 sorts out the matrix shapes.

## Book references

* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), the main text:
  * Chapter 5, Vector Calculus: section 5.1 for lesson 01, section 5.2 for lessons 02 and 03, section 5.3 for lesson 04, section 5.6 as a preview of backpropagation.
* **The Matrix Calculus You Need For Deep Learning** (Parr, Howard), as support:
  * The introductory sections on partial derivatives for lesson 02, the chain rule section for lesson 03, the Jacobian section for lesson 04.

## Estimated time

About 2 weeks at 4 or 5 hours a week. Lesson 03 is the most important one in the module: once the chain rule is clear to you, backpropagation (module 05) will be a formality.

## The running thread

The 5-house dataset returns here in a new role: the loss, the model's error score, seen as a function of one weight. In lesson 01 you'll discover it's a parabola, and that its slope tells you which way to correct the weight. That's the seed of the gradient descent you'll meet in module 04.
