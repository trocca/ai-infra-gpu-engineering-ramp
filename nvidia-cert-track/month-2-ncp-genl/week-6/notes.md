# Week 6 Notes — Fine-Tuning + Evaluation

## Full fine-tuning
### Memory anatomy (weights, grads, optimizer states — bytes/param)
-
-
### Catastrophic forgetting; when full FT is still right
-

## LoRA
### Update equation, shapes, init
-
-
### r, alpha (α/r scaling), dropout, target modules
-
-
### Merging adapters; multi-adapter serving
-
### Trainable-param math example
-

## QLoRA
### NF4 — what and why
-
### Double quantization
-
### Paged optimizers
-
### What stays high-precision, where gradients flow
-

## Other PEFT (one-liners each)
### Prompt tuning / p-tuning
-
### Prefix tuning
-
### IA³ / adapters
-

## Hyperparameters
### LR for LoRA vs full FT
-
### Epochs, batch/grad-accum, warmup/schedule
-
### BF16 vs FP16
-

## Instruction tuning / SFT
### Dataset formats; loss masking on prompt tokens
-
-
### Data quality > quantity (LIMA)
-

## RLHF
### Stage 1: SFT
-
### Stage 2: Reward model (preference pairs)
-
### Stage 3: PPO + KL penalty; reward hacking
-
-

## DPO and successors
### DPO loss intuition; β; reference model
-
-
### vs PPO: pros/cons
-
### KTO / ORPO / GRPO one-liners
-

## Tooling
### HF PEFT + TRL + bitsandbytes
-
### NeMo fine-tuning, NeMo-Aligner
-

## Evaluation
### Perplexity: definition + two limitations
-
-
### Benchmark → what it measures (MMLU, HellaSwag, GSM8K, HumanEval, TruthfulQA, MT-Bench, Arena, IFEval)
-
-
### Contamination
-
### LLM-as-judge: setups, biases, mitigations
-
-
### Task metrics: ROUGE/BLEU, EM/F1, pass@k; RAG triad
-
-
### Tooling: lm-eval-harness, NeMo Evaluator
-

## Lab observations (adapter size, before/after diffs, surprises)
-
-

## Misses from self-check
-
