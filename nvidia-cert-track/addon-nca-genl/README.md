# NCA-GENL — NVIDIA-Certified Associate: Generative AI LLMs (add-on track)

> **Status: optional easy-win badge.** This is the *Associate* exam — a different SKU
> from month-2's NCP-GENL (Professional). ~90% of its content is a subset of the
> NCP-GENL prep; this folder covers only the delta.
> Official page: https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/
> **Verify price / duration / question count / domain weights on the official page before booking — not filled in here on purpose.**

## Why bother

- A second badge for marginal extra effort (the delta is ~2 half-days of labs + 1 mock).
- Confidence run under real proctored-exam conditions **before** sitting the harder NCP-GENL.
- The Associate blueprint tests classical-ML fundamentals the Professional track skips —
  and which a Developer Advocate should be fluent in anyway.

## The delta (what month-2 does NOT cover)

| Associate topic | Where it's covered | Action |
|---|---|---|
| Classical data preprocessing: scaling, encoding, when/why | **nowhere in main track** | [`lab-preprocessing-fundamentals.md`](lab-preprocessing-fundamentals.md), notebook 1 |
| scikit-learn hands-on: MinMaxScaler, StandardScaler, OneHotEncoder | **nowhere in main track** | same lab, notebook 1 |
| Preprocessing tradeoffs & impact on model performance | **nowhere in main track** | same lab, notebook 1 (comparison section) |
| Text preprocessing / tokenization hands-on | month-2 week 5 day 2 | re-run as notebook 2, exam-lens |
| ML basics (supervised/unsupervised, train/val/test, over/underfitting, metrics) | month-1 week 1 (partially) | flashcard sweep + self-check |
| Transformer architecture, attention, embeddings | month-2 week 5 days 1–2 | already covered — no action |
| Prompt engineering, RAG basics | month-2 week 5 days 3–4 | already covered — no action |
| Fine-tuning / PEFT concepts | month-2 week 6 | already covered — no action |
| Trustworthy AI / safety | month-2 week 8 day 3 | already covered — no action |
| NVIDIA stack awareness (NeMo, TensorRT-LLM, Triton/NIM, RAPIDS/cuDF) | spread across months 1–2 | flashcard sweep |

## Schedule (fits inside the main calendar, no new weeks)

- **Onboarding weekend or any weekend of weeks 1–5**: run both notebooks in
  [`lab-preprocessing-fundamentals.md`](lab-preprocessing-fundamentals.md) (~4–5 h total).
- **Week 6 (Sep 14–18)**: by Friday you've completed arch + prompting + data prep +
  fine-tuning — everything Associate-level is now covered. Take a timed Associate-style
  mock over the weekend.
- **Target exam: end of week 6 / week 7 weekend (≈ Fri Sep 18 – Sun Sep 27)**, i.e. a
  low-stakes dress rehearsal ~2 weeks before NCP-GENL on Oct 2.
- **Gate (same rule as everything else)**: sit only at ≥ 80% on a timed, closed-book mock.

## Rules

- This track never steals time from the main plan on weekdays. Weekend-only.
- If a weekend is needed to repair a red Friday gate, the main track wins and this
  add-on slides. The Associate badge is a bonus, not a milestone.
- Log it in `../PROGRESS.md` under the restart note when done.
