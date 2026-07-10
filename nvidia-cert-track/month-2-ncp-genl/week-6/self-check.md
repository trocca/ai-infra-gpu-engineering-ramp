# Week 6 Self-Check — Fine-Tuning + Evaluation

**1. Write the LoRA forward pass for a linear layer and explain why B is initialized to zero.**

<details><summary>Answer</summary>
h = Wx + (α/r)·B·A·x, with W frozen, A ∈ ℝ^{r×d} (random init), B ∈ ℝ^{d×r} (zero init). Zero-initializing B makes ΔW = BA = 0 at step 0, so training starts exactly at the pretrained model's behavior and adapts smoothly instead of injecting random noise.
</details>

**2. A 7B model in BF16 is ~14 GB. Why does full fine-tuning with AdamW need on the order of 100+ GB?**

<details><summary>Answer</summary>
Training stores far more than weights: BF16 weights (14 GB) + gradients (14 GB) + AdamW state — FP32 momentum and variance plus typically an FP32 master copy of weights (~12 bytes/param ≈ 84 GB) — plus activations. Roughly 16+ bytes/param total vs 2 for inference. This is also exactly the memory that ZeRO/FSDP shards across GPUs.
</details>

**3. How many trainable parameters does LoRA with r=16 add to a single 4096×4096 attention projection, and what fraction of the original matrix is that?**

<details><summary>Answer</summary>
2 × 4096 × 16 = 131,072 ≈ 0.13 M, vs 16.8 M original — about 0.78%. Across a whole 7B model with LoRA on attention projections, trainable params are typically well under 1%.
</details>

**4. Name QLoRA's three key techniques and what each buys you.**

<details><summary>Answer</summary>
(1) NF4 quantization of the frozen base weights — 4-bit NormalFloat matched to normally-distributed weights, halving memory vs INT8 with minimal quality loss. (2) Double quantization — quantizes the per-block scaling constants themselves, saving ~0.4 bits/param. (3) Paged optimizers — optimizer state pages to CPU RAM on memory spikes, preventing OOM. Together: 7–8B fine-tuning on a single 24 GB GPU.
</details>

**5. In QLoRA, which tensors are 4-bit and which are not?**

<details><summary>Answer</summary>
Only the frozen base weights are NF4. LoRA adapter weights, gradients, and optimizer states are BF16/FP32; computation dequantizes base weights to BF16 on the fly. Gradients flow *through* the frozen quantized weights into the adapters only.
</details>

**6. Your LoRA fine-tune converges but the model has forgotten general chat ability. Name three plausible causes and fixes.**

<details><summary>Answer</summary>
(1) Learning rate too high for too long → lower LR / fewer epochs. (2) Too many epochs on a narrow dataset → 1–2 epochs, early stopping on a general eval. (3) Training data too narrow/homogeneous → mix in general instruction data (replay). Also check rank/α aren't oversized for a tiny dataset, and that loss masking isn't training on prompts.
</details>

**7. Why is the typical LoRA learning rate ~10× higher than full fine-tuning's?**

<details><summary>Answer</summary>
You're training a tiny set of freshly-initialized low-rank matrices, not perturbing billions of pretrained weights. The adapters need to move far from init quickly, and the frozen base protects against catastrophic updates — so ~1e-4–3e-4 works where full FT uses ~1e-5–2e-5.
</details>

**8. Describe the three stages of RLHF and the classic failure mode of stage 3.**

<details><summary>Answer</summary>
(1) SFT on demonstrations. (2) Reward-model training on human preference pairs (chosen vs rejected). (3) PPO: optimize the policy to maximize RM score with a KL penalty anchoring it to the SFT policy. Stage-3 failure: reward hacking / RM overoptimization — the policy exploits RM blind spots (e.g., verbosity, sycophancy), scoring high on the proxy while getting worse for humans; the KL term and RM refreshes mitigate it.
</details>

**9. What does DPO eliminate from the RLHF recipe, and what does it keep?**

<details><summary>Answer</summary>
Eliminates: the explicit reward model and the online RL (PPO) loop with its sampling and instability. Keeps: preference-pair data and a KL-style anchor — the reference (SFT) model appears in the loss, with β controlling how far the policy may drift. The policy's own log-ratios act as an implicit reward.
</details>

**10. In SFT, what is loss masking and why does it matter?**

<details><summary>Answer</summary>
Setting the loss to ignore prompt/user tokens so the model is only trained to predict assistant-response tokens. Without it, the model wastes capacity learning to reproduce prompts and can pick up degenerate behaviors like echoing user text; with multi-turn data it also prevents training on injected user content.
</details>

**11. A customer wants their support bot to know this week's product changes and never miss them. Fine-tune weekly or RAG? Why?**

<details><summary>Answer</summary>
RAG. Facts that change frequently belong in a retrievable store: instant updates, attribution/citations, deletability, no retraining cost or forgetting risk. Weekly fine-tunes are slow, expensive, unauditable, and unreliable at recalling specific facts. Fine-tune only for persistent tone/format/workflow behavior.
</details>

**12. Define perplexity and give two reasons it can mislead you when comparing models.**

<details><summary>Answer</summary>
PPL = exp(mean negative log-likelihood per token) — how "surprised" the model is by the text. Misleads because: (1) it's tokenizer-dependent — different vocabularies make per-token likelihoods incomparable; (2) it measures LM fit, not task ability — instruction quality, reasoning, and safety don't show up in it (a lower-PPL model can be a worse assistant).
</details>

**13. Match each to what it measures: MMLU, GSM8K, HumanEval, MT-Bench, TruthfulQA.**

<details><summary>Answer</summary>
MMLU — broad multi-domain knowledge via multiple choice. GSM8K — grade-school math word problems (multi-step reasoning). HumanEval — Python code generation, scored pass@k by running unit tests. MT-Bench — multi-turn conversational ability scored by an LLM judge. TruthfulQA — resistance to common misconceptions/falsehoods.
</details>

**14. What is benchmark contamination and one way to detect or avoid it?**

<details><summary>Answer</summary>
Test data leaking into training corpora, inflating scores without real ability. Detect: n-gram/substring overlap scans between train data and benchmark, or check for suspiciously perfect performance on canonical splits vs freshly-written variants. Avoid: dedup training data against eval sets (a NeMo Curator use case), use private/rotating eval sets.
</details>

**15. Name three biases of LLM-as-judge and a mitigation for each.**

<details><summary>Answer</summary>
Position bias (prefers first/last answer) → evaluate both orderings and average. Verbosity bias (prefers longer answers) → rubric that scores correctness/conciseness explicitly, length-controlled comparison. Self-preference bias (judge favors its own family's outputs) → use a different/stronger judge model, or a panel of judges.
</details>

**16. Why is ROUGE a poor metric for evaluating an instruction-tuned assistant's answers, and where is it still fine?**

<details><summary>Answer</summary>
ROUGE measures n-gram overlap with a reference; open-ended answers can be excellent with near-zero overlap or terrible with high overlap. It's still fine for constrained summarization against good references, and as a cheap regression signal — not as a quality verdict.
</details>

**17. What is pass@k?**

<details><summary>Answer</summary>
For code generation: the probability that at least one of k sampled solutions passes all unit tests, estimated over many problems. pass@1 ≈ single-shot correctness; higher k measures whether the model can find a solution given multiple attempts.
</details>

**18. You must show a customer that your fine-tune improved their task without an existing labeled test set. Sketch a defensible evaluation.**

<details><summary>Answer</summary>
Build a held-out eval set from real task inputs (never trained on); collect base-model and fine-tuned outputs; run blinded pairwise LLM-as-judge with a task rubric, both answer orders, plus a human spot-check on a sample (e.g. 50 items) to validate judge agreement; also run a general-ability benchmark (e.g. MMLU subset or IFEval) to prove no regression/forgetting. Report win-rate with confidence interval.
</details>

**19. When would you pick full fine-tuning over LoRA despite the cost?**

<details><summary>Answer</summary>
Large training corpus and deep capability change (new language, major domain shift, continued pretraining), when you need every last point of quality and have the hardware, or when adapter serving complexity is unwanted and you'd merge anyway at scale. For most enterprise SFT, LoRA matches full FT within noise at ~1% of trainable params.
</details>

**20. What is NeMo-Aligner (and its NeMo 2.0 successor tooling) used for?**

<details><summary>Answer</summary>
NVIDIA's alignment toolkit inside the NeMo framework: scalable SFT, reward-model training, PPO-based RLHF, DPO, and related preference-optimization methods on Megatron-parallel models — i.e., the NVIDIA-stack answer to "how do I RLHF/DPO a large model across many GPUs."
</details>
