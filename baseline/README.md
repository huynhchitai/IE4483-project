# TF-IDF + Logistic Regression baseline

Uses the saved shared split. Verified baseline results are available in
[results/baseline/baseline_report.md](../results/baseline/baseline_report.md) and
[the PDF report](../results/baseline/baseline_report.pdf).
Read [the method report](../reports/tfidf_logreg_method.pdf) first.

From the repository root:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python baseline/train.py --dry-run
# When ready to train:
.venv/bin/python baseline/train.py
```

Defaults: unigram + bigram TF-IDF, `min_df=2`, sublinear TF, smoothed IDF,
L2-normalized float64 sparse vectors; Logistic Regression with `C=4`,
`class_weight='balanced'`, `solver='lbfgs'`, `max_iter=2000`.
The word token pattern retains contractions such as `don't`, digits and
single-character words. No stop-word removal, stemming or lemmatization.
This differs from both the descriptive EDA tokenizer and sklearn's default
token pattern used in the earlier exploratory baseline.

`--dry-run` validates data fingerprints, source-index alignment, label values,
full source coverage and zero exact-text overlap; it does not fit or write files.
The pipeline fits vocabulary, IDF and classifier only on `train_splitted.json`.
Class weights come from that partition. Validation is transformed without refit.

Training exports to `artifacts/tfidf_logreg/` (ignored by Git):

- `model.joblib`: fitted vectorizer + classifier together.
- `metrics.json`: accuracy, macro-F1, per-class metrics, confusion matrix,
  majority-class reference, feature dimensions, nonzeros, numeric sparse storage,
  timing, process peak RSS, configuration, versions and split fingerprints.
- `validation_predictions.csv`: original row index, review, truth, prediction,
  positive-class probability for error analysis.

Use a new `--output-dir` per experiment; existing results are never overwritten.
Examples for a later controlled comparison:

```bash
.venv/bin/python baseline/train.py --ngram-max 1 --output-dir artifacts/unigram_balanced
.venv/bin/python baseline/train.py --class-weight none --output-dir artifacts/bigram_unweighted
```

Export a report from an existing run without refitting:

```bash
.venv/bin/python baseline/report_results.py
```

This verifies the saved model, predictions and split, then writes the PDF,
Markdown, exact metrics, verification fingerprints and summary CSV to `results/baseline/`.
It requires the chart/PDF dependencies included in root `requirements.txt`.

The initial defaults have now been checked in a 12-configuration validation
grid: n-grams {unigram, unigram+bigram}, class weights {none, balanced}, and
`C` {0.5, 1, 4}. The best tested validation macro-F1 is 0.8490 with the existing
defaults (bigram, balanced, C=4). This is a single-split selection, not a global
optimum or cross-validation estimate. See the
[questions (b)/(c) report](../data_analysis/question_b_c_baseline.md) and
[PDF](../results/baseline/question_b_c_baseline.pdf).

Reproduce tuning into fresh output directories:

```bash
.venv/bin/python baseline/tune.py --output-dir results/baseline/tuning_reproduction --artifact-dir artifacts/tfidf_logreg_tuning_reproduction
```

Generate the (b)/(c) Markdown, PDF and architecture figure from recorded evidence:

```bash
.venv/bin/python baseline/report_b_c.py
```

The tuning script limits numerical libraries to one thread; its fit times are
not directly comparable with the original run's resource measurements.
Select by validation macro-F1 while checking negative recall. Weighted model
probabilities need not be calibrated to the observed positive prevalence.
No test predictions or final full-data refit are performed by this script.
Process peak memory includes interpreter/import overhead; record equivalent
measurement boundaries when comparing against a GRU. `resource` requires Unix.

Run focused tests (toy data only; no real-data training):

```bash
.venv/bin/python -m unittest discover -s baseline/tests -v
```
