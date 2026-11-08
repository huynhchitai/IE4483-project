# TF-IDF + Logistic Regression baseline

Prepared for the saved shared split. No production training run has been made.
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

Initial hyperparameters are starting choices, not validation-selected winners.
Select by validation macro-F1 while checking negative recall. Weighted model
probabilities need not be calibrated to the observed positive prevalence.
No test predictions or final full-data refit are performed by this script.
Process peak memory includes interpreter/import overhead; record equivalent
measurement boundaries when comparing against a GRU. `resource` requires Unix.

Run focused tests (toy data only; no real-data training):

```bash
.venv/bin/python -m unittest discover -s baseline/tests -v
```
