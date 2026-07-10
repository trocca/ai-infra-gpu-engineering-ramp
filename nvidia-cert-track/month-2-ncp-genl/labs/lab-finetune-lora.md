# Lab 1 — LoRA Fine-Tune on a Single Commodity GPU

Fine-tune a small instruct model with HF PEFT + TRL, inspect the adapter, merge it, and compare generations before/after.

| | |
|---|---|
| GPU | 1 × L4 (24 GB) / A10G (24 GB) / RTX 4090 (24 GB). 16 GB also works at this model size. |
| Est. time | 60–90 min total (~15 min actual training) |
| Est. cost | L4 ≈ $0.40–0.70/hr, A10G ≈ $0.75–1.10/hr, 4090 (community clouds) ≈ $0.35–0.70/hr → **under $2 total** |
| Exam domains served | Fine-Tuning (13%), Model Optimization intuition (17%) |

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` — ungated, Apache-2.0, downloads without a token.
*Alternative:* `meta-llama/Llama-3.2-1B-Instruct` is **gated** on Hugging Face (accept the license, then `huggingface-cli login`). Everything below works with either; commands use Qwen so the lab runs with zero account friction. Another ungated fallback: `HuggingFaceTB/SmolLM2-1.7B-Instruct`.

**Dataset:** `yahma/alpaca-cleaned` (ungated, ~52k instruction/response pairs). We train on 2,000 examples — enough to see behavior change in ~15 min.

## 0. Prerequisites

- Cloud instance with 1 GPU (Lambda / RunPod / Vast / AWS g6.xlarge etc.), Ubuntu 22.04+, CUDA 12.x driver, ~40 GB disk.
- Python 3.10–3.12.

```bash
nvidia-smi                       # confirm the GPU and driver are visible
python3 -m venv ~/lab && source ~/lab/bin/activate
pip install --upgrade pip
pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.46" "trl>=0.12" "peft>=0.13" "datasets>=3.0" "accelerate>=1.0" bitsandbytes
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Expected: your torch version and e.g. `NVIDIA L4`.

## 1. Baseline generations (BEFORE)

Save as `generate.py` — we'll reuse it after training:

```python
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = sys.argv[1] if len(sys.argv) > 1 else None

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
if adapter:
    model = PeftModel.from_pretrained(model, adapter)

prompts = [
    "Give me three tips for onboarding new Kubernetes users.",
    "Explain the difference between a list and a tuple in Python.",
    "Write a haiku about GPUs.",
]
for p in prompts:
    msgs = [{"role": "user", "content": p}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(ids, max_new_tokens=200, do_sample=False)
    print("=" * 80, "\nPROMPT:", p, "\n", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

```bash
python generate.py | tee before.txt
```

**Observe:** first run downloads ~3 GB of weights. Baseline answers are decent but generic — save them; the comparison is the point of the lab.

## 2. Train the LoRA adapter

Save as `train_lora.py`:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")

ds = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")

def to_chat(ex):
    user = ex["instruction"] + (("\n\n" + ex["input"]) if ex["input"] else "")
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": ex["output"]}]}
ds = ds.map(to_chat, remove_columns=ds.column_names)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

cfg = SFTConfig(
    output_dir="qwen-lora-alpaca",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch 16
    learning_rate=2e-4,                 # note: ~10x higher than full fine-tuning
    lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, logging_steps=10,
    max_length=1024, packing=False,
    report_to="none",
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)
trainer.train()
trainer.save_model("qwen-lora-alpaca/final")
print("done")
```

```bash
python train_lora.py
```

**Expected output:** a line like `trainable params: 18,464,768 || all params: 1,562,000,000 || trainable%: 1.18` (numbers vary slightly), then loss logging every 10 steps for ~125 steps, loss falling from ~1.6–1.8 toward ~1.1–1.3. Runtime ~10–20 min on L4-class. VRAM ~8–12 GB (`watch -n2 nvidia-smi` in a second terminal).

> API drift note: TRL moves fast. If `SFTConfig(max_length=...)` errors on your version, the older name is `max_seq_length`. `pip show trl` and check the docs for your version.

**Observe:**
- Trainable ≈ 1% of parameters — say out loud *why* (frozen W, ΔW = (α/r)·BA, r=16).
- Optimizer state exists only for adapter params — that's why this fits in 24 GB while full FT would not.
- Optional QLoRA variant: load the base with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` and watch VRAM drop to ~5–6 GB.

## 3. Inspect the adapter

```bash
ls -lh qwen-lora-alpaca/final/
du -sh qwen-lora-alpaca/final/
```

**Expected:** `adapter_model.safetensors` around **70–75 MB** plus a small `adapter_config.json` — vs ~3 GB for the base model. Open `adapter_config.json` and confirm r, alpha, and target_modules match the script. This size ratio is the whole multi-tenant LoRA-serving story.

## 4. Generations AFTER + diff

```bash
python generate.py qwen-lora-alpaca/final | tee after.txt
diff before.txt after.txt | head -50
```

**Observe:** outputs shift toward Alpaca's terse instructional style (often numbered/compact answers, less boilerplate). The knowledge didn't change — the *behavior/format* did. That's the RAG-vs-FT discriminator made visible.

## 5. Merge the adapter

Save as `merge.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "qwen-lora-alpaca/final")
merged = model.merge_and_unload()          # W' = W + (alpha/r) * B @ A
merged.save_pretrained("qwen-merged")
AutoTokenizer.from_pretrained(BASE).save_pretrained("qwen-merged")
print("merged")
```

```bash
python merge.py && du -sh qwen-merged/
```

**Expected:** `qwen-merged/` is ~3.1 GB — same as the base; the adapter dissolved into the weights, so inference overhead is zero. Keep this folder if you're doing Lab 2 next (it can serve as your fine-tuned serving target); otherwise clean up.

## 6. Cleanup

```bash
rm -rf qwen-lora-alpaca qwen-merged ~/.cache/huggingface
```

Then **terminate the cloud instance** — that's the actual cost control.

## What you should now be able to explain (exam mapping)

- LoRA equation, r/alpha, target modules, ~1% trainable params (Fine-Tuning 13%)
- Why adapter ≈ 70 MB and what that enables in serving (Deployment 9%)
- LR 2e-4 vs full-FT 2e-5; grad accumulation as effective batch (Fine-Tuning)
- Merge math and why merged inference has zero overhead
- QLoRA as the same recipe with an NF4-quantized frozen base (Model Optimization 17%)
