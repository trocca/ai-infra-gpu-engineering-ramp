# Attention and transformer math

## The intuition

Last lesson, last brick: the mechanism that makes LLMs work. And the good news is that you already know all the pieces.

**Attention** is a fuzzy dictionary. In a real dictionary you look up an exact key and take a value. In attention, every word in the sentence asks a question (its **query**), every word exposes a label (its **key**) and a content (its **value**). The lookup doesn't find ONE key: it measures how much the question resembles every label (dot product, module 01), turns the similarities into percentages (softmax, module 03), and returns the weighted average of the contents. Every word comes out enriched by a mix of the others, weighted by relevance.

A mental example: in the sentence "the cat sleeps because it is tired", the word "it" asks "who am I talking about?", the key of "cat" answers loudly, and the value of "cat" flows into "it". That's the entire secret of context in LLMs.

## The formal idea, in plain words

Every token (word or piece of a word) arrives as a vector. Three weight matrices, learned through training, project it into three roles: Q = query, K = key, V = value. Then:

    attention(Q, K, V) = softmax(Q @ Kᵀ / √d) @ V

Read from the inside out, with the shapes alongside:

1. `Q @ Kᵀ`: all the question-times-label dot products. With n tokens, an n x n matrix of affinity scores. A matmul from module 01.
2. `/ √d`: divide by the square root of the vector dimension d, because with large d the dot products grow and the softmax would saturate, spitting out almost nothing but 0s and 1s. A scale factor, nothing more.
3. `softmax` row by row: each row becomes a probability distribution: "this is how token i spreads its attention".
4. `@ V`: the weighted average of the contents. Another matmul.

Models that generate text need one more detail: the **causal mask**. A token must not see the future, so the scores toward later tokens are set to minus infinity before the softmax, which turns them into zero probabilities. A transformer is this block, plus an MLP (lesson 03), stacked dozens of times, trained with cross entropy (module 03) and Adam (module 04). End of the ingredient list.

## A numerical example by hand

Two tokens, dimension 2. Token 1 has query q = [1, 0]. The keys are k1 = [1, 0] and k2 = [0, 1], the values v1 = [10, 0] and v2 = [0, 10].

    scores:   q · k1 = 1,   q · k2 = 0
    scale:    √d = √2 ≈ 1.41, so [0.71, 0]
    softmax:  e^0.71 = 2.03, e^0 = 1: percentages [2.03, 1] / 3.03 = [0.67, 0.33]
    output:   0.67 * [10, 0] + 0.33 * [0, 10] = [6.7, 3.3]

Token 1 resembles key 1 more, so it draws two thirds of its content from v1 and one third from v2. No new step anywhere: dot product, softmax, weighted average.

## References

* Prince, Understanding Deep Learning: chapter 12 (transformers), especially the sections on self attention and scaled dot product attention.

## What's next

Run `python lesson.py`. It builds attention by hand on 4 tokens, checks it against PyTorch's `F.scaled_dot_product_attention`, adds the causal mask, and prints the attention matrix: who looks at whom. The heatmap lands in `figures/`. It's the last lesson: at the end, look back and count the bricks.
