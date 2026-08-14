---
title: "05 · Neural networks"
parent: "AI Math"
nav_order: 5
permalink: /docs/ai-math/05-reti-neurali/
---

# Module 05 · Neural networks from the math

The synthesis. Every piece built in the previous modules (matrices, gradients,
chain rule, likelihood, cross entropy, gradient descent, Adam) clicks together
here, and by the end of the module you'll have written by hand: a linear regression
that learns, a classifier, a real neural network without `nn.Module`,
backpropagation verified link by link, and the attention math that makes
transformers work.

No magic left: just matmuls, slopes, and average surprises.

![Decision map of the MLP on the XOR problem: a one-hidden-layer network bends space](figures/xor_map.png)

## Lessons

| Lesson | Topic | What you'll be able to do |
|---|---|---|
| [01 · Linear regression from scratch](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/05-reti-neurali/01-regressione-lineare-da-zero) | The full training loop | Train your first real model on the 5-house dataset |
| [02 · Logistic regression](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/05-reti-neurali/02-regressione-logistica) | Sigmoid, binary cross entropy | Build a classifier and read its probabilities |
| [03 · MLP from scratch, no nn](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/05-reti-neurali/03-mlp-da-zero-senza-nn) | Hidden layer, ReLU, XOR | Build a network that solves a problem impossible for linear models |
| [04 · Autograd under the hood](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/05-reti-neurali/04-autograd-sotto-il-cofano) | The compute graph, backprop by hand | Redo the work of `backward()` by hand and verify it to the decimal |
| [05 · Attention and transformers](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/ai-math/05-reti-neurali/05-attention-e-transformer) | Query, key, value, softmax, causal mask | Compute by hand the mechanism that makes LLMs work |

The lessons are a ladder: every rung uses the one before it.

![Heatmap of attention weights with a causal mask: every token looks only backward](figures/attention_heatmap.png)

## Book references

- **Understanding Deep Learning** (Prince), the module's main text: chapter 2
  (supervised learning) for lesson 01, chapter 5 (loss functions) for 02, chapters
  3–4 (shallow and deep networks) for 03, chapter 7 (gradients and initialization)
  for 04, chapter 12 (transformers) for 05.
- **Mathematics for Machine Learning**: chapter 9 (linear regression) for lesson
  01; chapter 5, section 5.6 (backpropagation) for lesson 04.

## The bridge to the rest of the path

The attention you compute by hand in lesson 05 is the same one the
[ML chapter](../../how-machines-learn/) tells in words and the repo's LLM modules
([GPT from scratch](https://github.com/trocca/ai-infra-gpu-engineering-ramp/tree/main/gpu-engineering-lab))
implement at scale. From here on, a transformer paper is just new notation over
math you've already stepped through in a debugger.
