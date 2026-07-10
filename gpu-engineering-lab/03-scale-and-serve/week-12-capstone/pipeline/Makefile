# Capstone pipeline: fine-tune -> merge -> quantize -> package -> (deploy) -> (loadtest)
# SKELETON — the stage wiring is complete; each stage's command is a TODO
# pointing at YOUR earlier weeks' code. No new ML code should be written
# this week; if a stage needs new code, that's a gap to fix in the earlier week.
#
# Contracts per stage: see steps.md. Every stage is idempotent: it skips
# itself if its output exists and is newer than its inputs (classic make).

ART      := artifacts
BASE     ?= $(ART)/base            # week-05 pretrained checkpoint (input)
ADAPTER  := $(ART)/lora-adapter
MERGED   := $(ART)/merged-fp16
QUANT    := $(ART)/model-quant
FINAL    := $(ART)/model-v1

.PHONY: pipeline deploy loadtest clean

pipeline: $(FINAL)/manifest.json
	@echo "=== pipeline complete: $(FINAL) ==="

# ---- stage 1: LoRA fine-tune (week 06) --------------------------------
$(ADAPTER)/adapter.safetensors: | $(BASE)
	@echo "=== [1/4] finetune ==="
	# TODO(you): invoke your week-06 LoRA trainer, e.g.
	# python -m weeks.week06.src.train_lora --base $(BASE) --out $(ADAPTER) \
	#     --config pipeline/configs/finetune.yaml
	# Must write: adapter.safetensors + manifest.json (see steps.md).
	@exit 1

# ---- stage 2: merge adapter into base weights --------------------------
$(MERGED)/model.safetensors: $(ADAPTER)/adapter.safetensors
	@echo "=== [2/4] merge ==="
	# TODO(you): W = W + (alpha/r) * B @ A, save merged fp16 weights.
	# Parity gate (steps.md): merged-model logits == base+adapter logits <= 1e-4.
	@exit 1

# ---- stage 3: quantize (week 07 method) --------------------------------
$(QUANT)/model.safetensors: $(MERGED)/model.safetensors
	@echo "=== [3/4] quantize ==="
	# TODO(you): your week-07 quantizer. Manifest must record method,
	# bits, group size, and the perplexity delta vs $(MERGED).
	@exit 1

# ---- stage 4: package the servable artifact ----------------------------
$(FINAL)/manifest.json: $(QUANT)/model.safetensors
	@echo "=== [4/4] package ==="
	# TODO(you): assemble $(FINAL)/ = weights + tokenizer + config +
	# merged manifest chain (all upstream manifests embedded), ready for
	# the week-08 server's --model-path.
	@exit 1

# ---- day 2, on the cloud node ------------------------------------------
deploy:
	@echo "see week-11: make everything, then point deployment.yaml at $(FINAL)"
	# TODO(you): document/script the artifact transfer (scp/registry/object store).

loadtest:
	# TODO(you): invoke the week-08 load-test harness against the deployed
	# service; write results to $(ART)/loadtest-final.json.
	@exit 1

clean:
	rm -rf $(ADAPTER) $(MERGED) $(QUANT) $(FINAL)
