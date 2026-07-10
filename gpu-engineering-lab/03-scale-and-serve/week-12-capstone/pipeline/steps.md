# Pipeline stage contracts (skeleton — tighten as you implement)

Every stage reads an input directory, writes an output directory, and
drops a `manifest.json`. A stage may assume its input's manifest is valid
(the previous stage's job) and must never modify its inputs.

## Common manifest fields (every stage)

```json
{
  "stage": "finetune | merge | quantize | package",
  "git_sha": "<repo state that produced this>",
  "created_utc": "...",
  "inputs": {"<name>": "<sha256 of each input file>"},
  "outputs": {"<name>": "<sha256>"},
  "config": {},
  "metrics": {}
}
```

The hash chain is the point: given `model-v1/manifest.json` you can walk
back to the exact git SHA and config of every stage. That is what
"reproducible artifact" means in this repo.

## Stage 1 — finetune (week 06)

- **In:** `artifacts/base/` (week-05 checkpoint), `pipeline/configs/finetune.yaml`
- **Out:** `artifacts/lora-adapter/` — `adapter.safetensors` (LoRA A/B only), manifest
- **Metrics:** final train/val loss, trainable-param count + % of total, wall time
- **Gate:** val loss improved vs base by a threshold you set (TODO: pick and justify)
- **Budget:** < 1 h on the RTX 5090 — size the task, not the ambition

## Stage 2 — merge

- **In:** base + adapter
- **Out:** `artifacts/merged-fp16/` — full weights, tokenizer copied through, manifest
- **Gate (blocking):** logits(merged, fixed prompt set) vs logits(base + live adapter)
  max abs diff <= 1e-4. TODO: write this as an assert inside the merge script —
  a merge that silently drifts is the classic pipeline bug.

## Stage 3 — quantize (week 07)

- **In:** merged-fp16
- **Out:** `artifacts/model-quant/` — quantized weights + quant config, manifest
- **Metrics:** bits/group-size, size on disk before/after, perplexity delta on the
  week-07 eval set (TODO: same eval set as week 07, or the comparison is meaningless)
- **Gate:** perplexity delta below the threshold you defended in week 07

## Stage 4 — package

- **In:** model-quant (+ tokenizer, + server config template)
- **Out:** `artifacts/model-v1/` — everything the week-08 server needs at
  `--model-path`, plus the embedded manifest chain
- **Gate:** a local smoke test — start the server against the artifact,
  one `/generate` request succeeds, coherent-ish output. TODO: script it
  (`pipeline/smoke_test.sh`), don't eyeball it.

## Deploy / loadtest (Day 2, cloud)

- Deploy on the week-11 stack; run the week-08 harness twice: once against
  `merged-fp16`, once against `model-v1` (quantized). Same load profile.
- **Out:** `artifacts/loadtest-final.json` — the numbers for the README table
  and REPORT.md headline.

## Open TODOs (resolve while implementing)

- [ ] Where does `artifacts/base/` come from — retrain from week 05 or a committed checkpoint? Document.
- [ ] Artifact transfer to the cloud node: scp, OCI registry, or object store? Pick, script it.
- [ ] Config files under `pipeline/configs/` — create as you go, keep them tiny and committed.
