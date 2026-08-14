# Logistic regression

## The intuition

The question changes. So far: "how much does this house cost?". Now: "is this house expensive, yes or no?". From a continuous number to a class. We need a model that answers with a **probability**: "92 percent expensive".

The trick is tiny. The linear model already produces a score: high for big houses, low for small ones. All we have to do is squash that score into the interval between 0 and 1. The function that does it is called the **sigmoid**: a soft S that sends very negative scores toward 0, very positive ones toward 1, and zero exactly to 0.5.

Despite the name, logistic regression is a classifier. And it's also a neuron: linear score plus squashing function is exactly the brick we'll use to build the network in the next lesson.

## The formal idea, in plain words

The model, in two lines:

    z = x · w + b            (the score, called the logit)
    p = sigmoid(z) = 1 / (1 + e^(-z))     (the probability of class 1)

The loss is the **binary cross entropy** (BCE), a direct descendant of module 03 lesson 03: the model's average surprise when faced with the truth.

    BCE = mean of [ -log(p) if the truth is 1, -log(1-p) if the truth is 0 ]

Which is also, once again, a negative log likelihood: maximizing the probability of the observed labels. MLE, entropy, and the training loop all meet right here.

Why not use MSE? You can, but BCE punishes confident mistakes much harder (remember: -log of a small probability blows up) and produces healthier gradients for this shape of model. It's the standard, and now you know where it comes from.

## A numerical example by hand

The sigmoid on three scores:

    z = 0:    p = 1 / (1 + e^0)  = 1 / 2      = 0.5     (undecided)
    z = 2:    p = 1 / (1 + e^-2) = 1 / 1.135  ≈ 0.88    (fairly confident: class 1)
    z = -2:   p ≈ 0.12                                  (fairly confident: class 0)

The BCE on one example whose truth is 1:

    model says p = 0.88:  loss = -log(0.88) ≈ 0.128   (good)
    model says p = 0.12:  loss = -log(0.12) ≈ 2.12    (wrong and confident: punishment)

In the script we label the 5 houses: expensive from 250 thousand up, i.e. y = [0, 0, 1, 1, 1], and the model learns the boundary on its own.

## References

* Prince, Understanding Deep Learning: chapter 5 (loss functions), the sections on binary classification.
* Mathematics for Machine Learning: chapter 8, section 8.3, for the general link between likelihood and loss.

## What's next

Run `python lesson.py`. Sigmoid and BCE written by hand and checked against `torch.sigmoid` and `F.binary_cross_entropy`, then the training loop (identical to yesterday's, only the loss changes) and the final probabilities house by house. The S curve lands in `figures/`.
