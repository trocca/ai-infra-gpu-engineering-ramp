# Module 05: neural networks from the math up

The synthesis. Every piece built in the previous modules (matrices, gradients, chain rule, likelihood, cross entropy, gradient descent, Adam) clicks into place here, and by the end of the module you will have written by hand: a linear regression that learns, a classifier, a real neural network without `nn.Module`, backpropagation verified link by link, and the attention math that makes transformers work.

No magic left: just matmuls, slopes, and average surprises.

## Lessons

| Lesson | Topic | What you'll be able to do |
|---|---|---|
| [01-regressione-lineare-da-zero](01-regressione-lineare-da-zero/) | The complete training loop | Train your first real model on the 5-house dataset |
| [02-regressione-logistica](02-regressione-logistica/) | Sigmoid, binary cross entropy | Build a classifier and read its probabilities |
| [03-mlp-da-zero-senza-nn](03-mlp-da-zero-senza-nn/) | Hidden layers, ReLU, XOR | Build a network that solves a problem impossible for linear models |
| [04-autograd-sotto-il-cofano](04-autograd-sotto-il-cofano/) | The computation graph, backprop by hand | Redo backward()'s work by hand and verify it to the decimal |
| [05-attention-e-transformer](05-attention-e-transformer/) | Query, key, value, softmax, causal mask | Compute by hand the mechanism that makes LLMs work |

The lessons must be done in order: they are a ladder, and each rung stands on the one below.

## Book references

* **Understanding Deep Learning** (Prince), the main text for this module:
  * Chapter 2 (supervised learning) for lesson 01.
  * Chapter 5 (loss functions) for lesson 02.
  * Chapters 3 and 4 (shallow and deep networks) for lesson 03.
  * Chapter 7 (gradients and initialization) for lesson 04.
  * Chapter 12 (transformers) for lesson 05.
* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), as support:
  * Chapter 9 (linear regression) for lesson 01, chapter 5 section 5.6 (backpropagation) for lesson 04.

## Estimated time

About 3 weeks at 4 or 5 hours a week. Lessons 04 and 05 are the finish line of the whole journey: take the time to step through them in the debugger, line by line.

## The common thread

The 5-house dataset opens the module: the linear model that was a matmul in module 01, a slope in module 02, and a descent in module 04 becomes a complete training loop here, and then a classifier. From lesson 03 onward the climb begins: XOR, backprop, attention. At the end, when you read "multi head self attention" in a paper, you'll know it's the matmul from module 01 in fancy clothes.
