# Week 1 Self-Check — Essential AI Knowledge

Answer from memory first, then expand the answer. Target: ≥ 80% (14/17) before moving to week 2.

**1. What is the relationship between AI, ML, and DL?**

<details><summary>Answer</summary>
Nested subsets: AI is the broad goal of machines performing tasks that require intelligence; ML is the subset where systems learn patterns from data instead of being explicitly programmed; DL is the subset of ML that uses multi-layer (deep) neural networks to learn representations automatically.
</details>

**2. A model learns from labeled images of defective vs good parts. A second model groups customers into segments with no labels. Which learning paradigm is each?**

<details><summary>Answer</summary>
The first is supervised learning (labeled data, learns input→label mapping). The second is unsupervised learning (finds structure/clusters in unlabeled data).
</details>

**3. Walk through what happens in one training iteration of a neural network.**

<details><summary>Answer</summary>
Forward pass: a batch of data flows through the network to produce predictions. The loss function measures the error vs the true labels. Backpropagation computes gradients of the loss with respect to every weight. The optimizer (gradient descent variant, e.g. Adam) updates the weights in the direction that reduces the loss. Repeat over many batches/epochs.
</details>

**4. Contrast training and inference in terms of the key performance metric each optimizes for.**

<details><summary>Answer</summary>
Training optimizes throughput — total samples/tokens processed per second over a long-running batch job; time-to-train is the headline. Inference typically optimizes latency (response time per request) while maintaining acceptable throughput/cost; production serving balances the two, often via batching.
</details>

**5. Why does training generally need more GPU memory than inference for the same model?**

<details><summary>Answer</summary>
Training must hold weights, activations (for backprop), gradients, and optimizer states (e.g. Adam keeps two extra values per parameter) — several times the model size. Inference needs only the weights plus working activations (and for LLMs, the KV cache).
</details>

**6. Why are GPUs better than CPUs for deep learning?**

<details><summary>Answer</summary>
DL is dominated by matrix multiplications — massively data-parallel work. GPUs have thousands of cores organized for throughput, very high memory bandwidth (HBM), and Tensor Cores that accelerate matrix math in reduced precision. CPUs have few cores optimized for serial, latency-sensitive, branchy code.
</details>

**7. What is the difference between a CUDA core and a Tensor Core?**

<details><summary>Answer</summary>
A CUDA core is a general-purpose scalar/vector arithmetic unit executing one operation (e.g. FMA) per clock per thread. A Tensor Core is a specialized unit that performs small matrix multiply-accumulate operations in one shot, in mixed/reduced precision (TF32, FP16, BF16, FP8), giving order-of-magnitude speedups for DL math.
</details>

**8. Name a scenario where the CPU is the right processor in an AI pipeline.**

<details><summary>Answer</summary>
Serial or branchy work: data ingestion and preprocessing/ETL logic, orchestration/control plane, small classical-ML models, business logic around the model. (GPU-accelerated alternatives exist — DALI, RAPIDS — but the exam expects "CPU for serial/general-purpose tasks".)
</details>

**9. What is CUDA, in one sentence?**

<details><summary>Answer</summary>
CUDA is NVIDIA's parallel computing platform and programming model (plus toolkit: compiler, runtime, libraries) that lets developers run general-purpose computation on NVIDIA GPUs.
</details>

**10. What do cuDNN and NCCL each do, and who typically calls them?**

<details><summary>Answer</summary>
cuDNN is a library of highly optimized deep learning primitives (convolutions, attention, normalization); NCCL implements fast multi-GPU/multi-node collective communication (all-reduce, broadcast, all-gather). Both are called by frameworks like PyTorch and TensorFlow rather than directly by most users.
</details>

**11. A customer has a trained PyTorch model and wants the lowest possible inference latency on NVIDIA GPUs. Which NVIDIA tool do you point them to and what does it do?**

<details><summary>Answer</summary>
TensorRT — an inference optimizer and runtime that applies layer/kernel fusion, precision reduction (FP16/FP8/INT8 with calibration), and kernel auto-tuning to produce a highly optimized engine for the target GPU. (For LLMs specifically: TensorRT-LLM; for serving it at scale: Triton or NIM.)
</details>

**12. What is Triton Inference Server and how does it differ from TensorRT?**

<details><summary>Answer</summary>
Triton is a model serving platform: it hosts models from multiple frameworks (TensorRT, PyTorch, ONNX…) behind HTTP/gRPC endpoints with dynamic batching, concurrent model execution, and metrics. TensorRT optimizes a model; Triton serves models in production. They're complementary.
</details>

**13. What is NGC and what three kinds of assets would you find there?**

<details><summary>Answer</summary>
NGC (NVIDIA GPU Cloud) is NVIDIA's catalog of GPU-optimized software: (1) containers (e.g. PyTorch, Triton, NeMo), (2) pretrained models, and (3) Helm charts (plus SDKs and other resources) — all tested and optimized for NVIDIA GPUs.
</details>

**14. A customer asks: "We can download all these frameworks for free — what does NVIDIA AI Enterprise actually give us?" Answer in two sentences.**

<details><summary>Answer</summary>
NVIDIA AI Enterprise is the supported, production-grade distribution of the NVIDIA AI stack: enterprise support with SLAs, security patching and long-term branch stability, and certification on mainstream servers, VMware, and clouds. It also licenses production components like NIM microservices — it's about running AI in production with support and stability guarantees, not about raw functionality.
</details>

**15. What is NIM?**

<details><summary>Answer</summary>
NVIDIA Inference Microservices — prebuilt, containerized inference services (model + optimized runtime like TensorRT-LLM + industry-standard API, typically OpenAI-compatible) that you deploy anywhere with a docker/helm command; part of NVIDIA AI Enterprise.
</details>

**16. List the stages of the AI development lifecycle in order.**

<details><summary>Answer</summary>
Business problem definition → data collection and preparation (cleaning, labeling, augmentation) → model selection and training → evaluation/validation → optimization for deployment (e.g. quantization, TensorRT) → deployment/serving → monitoring and retraining (handling drift). It's a loop, not a line.
</details>

**17. What is model drift and which lifecycle stage catches it?**

<details><summary>Answer</summary>
Drift is the degradation of model accuracy over time because real-world data distribution shifts away from the training data. The monitoring stage catches it (tracking prediction quality/data statistics), triggering retraining with fresh data.
</details>
