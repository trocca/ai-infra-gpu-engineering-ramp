# Week 1 Notes — Essential AI Knowledge

Fill these in as you study. Keep each bullet to one line — if you can't say it in one line, you don't own it yet.

## 1. AI / ML / DL fundamentals

### The hierarchy
- AI is…
- ML is…
- DL is…

### Learning paradigms
- Supervised learning:
- Unsupervised learning:
- Reinforcement learning:
- Example use case for each:

### Neural network basics
- A neural network consists of…
- Forward pass:
- Loss function:
- Backpropagation:
- Gradient descent / optimizer:
- Epoch vs batch vs iteration:

### Generative AI / LLMs
- Transformer / attention (one line):
- Foundation model:
- Fine-tuning vs prompting vs RAG:

### Common use cases (industry → example)
- Healthcare:
- Finance:
- Retail:
- Automotive:
- Manufacturing / other:

## 2. Training vs inference

| Axis | Training | Inference |
|---|---|---|
| Compute pattern | | |
| Typical precision | | |
| Memory pressure (what fills memory?) | | |
| Duration / shape of workload | | |
| Infrastructure implications | | |
| Key metric (throughput vs latency) | | |

- Why batch size matters for inference throughput:
- What the KV cache is (LLM inference):

## 3. GPU vs CPU architecture

### CPU
- Design goal:
- Core count / cache / branch prediction:
- Best at:

### GPU
- Design goal:
- SMs, CUDA cores, threading model (one line):
- Memory bandwidth story (HBM vs DDR):
- Best at:

### Tensor Cores
- What they are:
- CUDA core vs Tensor Core:

### When CPU is still the right answer
-

## 4. NVIDIA software stack (bottom-up)

- Driver:
- CUDA Toolkit:
- cuDNN:
- cuBLAS:
- NCCL:
- DALI:
- RAPIDS (cuDF / cuML):
- PyTorch / TensorFlow (relationship to the above):
- TensorRT:
- TensorRT-LLM:
- Triton Inference Server:
- NIM:
- NGC (what's in the catalog?):
- NVIDIA AI Enterprise (what does the license buy you?):

## 5. AI development lifecycle

| Stage | What happens | NVIDIA tool(s) |
|---|---|---|
| Problem definition | | |
| Data collection & prep | | |
| Model selection & training | | |
| Evaluation / validation | | |
| Optimization for deployment | | |
| Deployment / serving | | |
| Monitoring & retraining | | |

- Why data prep is often the biggest time sink:
- Model drift — what it is and why monitoring matters:

## Parking lot (questions to resolve later)
-
