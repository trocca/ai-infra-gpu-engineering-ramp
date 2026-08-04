# Lab — Preprocessing Fundamentals (NCA-GENL delta)

The one lab the main track doesn't cover. Two notebooks, ~4–5 h total, CPU-only.
Maps directly to the official exam-tip triad: **fundamentals** (purpose of each
technique), **practice** (scikit-learn hands-on), **tradeoffs** (impact on model
performance).

Environment: `pip install scikit-learn pandas matplotlib transformers datasets jupyter`

**Skeletons live in [`notebooks/`](notebooks/)** — setup/dataset cells run as-is (notebook 1 verified end-to-end offline; notebook 2's tokenizer cells download from HF Hub on first run), the teaching parts are `# TODO` cells you write yourself.

---

## Notebook 1 — Tabular preprocessing with scikit-learn (~2.5 h)

Dataset: any small mixed-type tabular set (e.g. sklearn's `fetch_openml('adult')`
or the built-in wine/breast-cancer sets plus a synthetic categorical column).

1. **Scalers, side by side.** Apply `MinMaxScaler` and `StandardScaler` to the same
   numeric columns; plot before/after distributions. Write one sentence per scaler:
   what it does, when to prefer it (MinMax → bounded inputs / distance-based or
   NN-adjacent models; Standard → ~Gaussian features, models assuming centered data),
   and one failure mode each (MinMax: outlier squashes the range; Standard: assumes
   scale is meaningful, sensitive to heavy tails).
2. **Encoders.** `OneHotEncoder` vs `OrdinalEncoder` on a categorical column.
   Show the dimensionality blowup on a high-cardinality column; note
   `handle_unknown='ignore'` and why it matters at inference time.
3. **The leakage demo (exam favorite).** Fit a scaler on the *full* dataset then
   split, vs fit on train only inside a `Pipeline` + `ColumnTransformer`. Compare
   cross-validation scores and explain why the first is subtly wrong even when the
   score barely moves: information from the test distribution leaked into the
   transform.
4. **Tradeoffs measurement.** Train a scale-sensitive model (KNN or logistic
   regression) and a scale-insensitive one (random forest) with and without
   scaling. Tabulate the four scores. The takeaway sentence to be able to say cold:
   *scaling changes distance- and gradient-based models, tree models don't care.*

## Notebook 2 — Text preprocessing for LLMs (~1.5–2 h)

Re-uses month-2 week-5 day-2 knowledge, exam-lens and hands-on.

1. Load a small text dataset with 🤗 `datasets`; tokenize with `AutoTokenizer`
   (one BPE model, e.g. GPT-2, and one WordPiece model, e.g. BERT).
2. Same sentence through both tokenizers: compare token counts, subword splits on
   rare words / numbers / non-English text. One paragraph on why vocab choice
   shifts effective context length and multilingual behavior.
3. Batch with `padding=True, truncation=True`; inspect `input_ids` and
   `attention_mask`; show what the mask actually masks. Compare fixed-length vs
   dynamic (longest-in-batch) padding and state the compute tradeoff.
4. Quick sweep of classic text cleaning (lowercasing, stopwords, stemming) and
   *when it's wrong for LLMs* (pretrained tokenizers expect raw-ish text; classic
   cleaning is for bag-of-words/TF-IDF pipelines) — a discriminator the exam
   likes.

## Exit criteria

- [ ] I can state purpose + one pro + one con for MinMaxScaler, StandardScaler, OneHotEncoder without notes
- [ ] I can explain data leakage via preprocessing and how Pipeline/ColumnTransformer prevents it
- [ ] I measured and can quote my own numbers: scaling's effect on KNN/logreg vs random forest
- [ ] I can explain BPE vs WordPiece output differences on a concrete sentence I ran
- [ ] I can explain padding + attention masks and the fixed-vs-dynamic padding tradeoff
- [ ] I can say when classical text cleaning helps and when it hurts an LLM pipeline
