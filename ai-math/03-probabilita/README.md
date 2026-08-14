# Module 03: probability

Real data is noisy, real models are uncertain. Probability is the language for reasoning well about uncertainty. In this module you'll also discover one of deep learning's secrets: nearly every loss actually used in practice (MSE, cross entropy) wasn't invented — each one is a direct consequence of probabilistic ideas.

## Lessons

| Lesson | Topic | What you'll be able to do |
|---|---|---|
| [01-variabili-casuali-distribuzioni](01-variabili-casuali-distribuzioni/) | Random variables, distributions, expected value | Simulate random processes and predict their average behavior |
| [02-verosimiglianza-mle](02-verosimiglianza-mle/) | Likelihood, log likelihood, MLE | Estimate parameters from data by maximizing the likelihood |
| [03-entropia-kl-cross-entropy](03-entropia-kl-cross-entropy/) | Surprise, entropy, KL, cross entropy | Understand where the classification loss really comes from |

The lessons are meant to be done in order. Lesson 03 closes the circle: the cross entropy you'll find in every neural network is where the likelihood from lesson 02 meets entropy.

## Book references

* **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), the main text:
  * Chapter 6, Probability and Distributions: sections 6.1 and 6.2 for lesson 01, section 6.4 for expected value and variance, section 6.5 for the Gaussian.
  * Chapter 8, When Models Meet Data: section 8.3 for the maximum likelihood of lesson 02.
* **Introduction to Probability** (Blitzstein, Hwang), supporting text:
  * Chapter 3 for random variables and distributions, chapter 4 for expected value, chapter 5 for continuous variables.
* **Understanding Deep Learning** (Prince), supporting text:
  * Chapter 5, Loss Functions: the bridge between probability and losses, used in lesson 03.

## Estimated time

About a week and a half at 4 to 5 hours per week. Lesson 01 is light; the other two deserve a calm pace: these are the concepts that make papers readable.

## The common thread

The error score changes face: in lesson 02 you'll discover that minimizing the MSE of the 5-house dataset model is equivalent to maximizing a Gaussian likelihood. And in lesson 03 you'll see that the classification loss is the model's average surprise when confronted with the truth. Losses aren't invented: they're derived.
