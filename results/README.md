# Experiment results

- [`baseline/`](baseline/): verified TF-IDF + Logistic Regression validation
  metrics, resource measurements, reports and a confusion-matrix figure.
- [`baseline/question_b_c_baseline.pdf`](baseline/question_b_c_baseline.pdf):
  model architecture, training strategy, reproducibility and hyperparameter
  selection; [12-trial tuning evidence](baseline/tuning/tuning_results.csv).
- [`gru_classifier/`](gru_classifier/): BiGRU tuning runs, figures, ensemble checkpoints, test predictions and the
  [(b)/(c) report](gru_classifier/gru_report.md).

Model binaries and row-level predictions remain in `artifacts/` (ignored by Git).
Regenerate the baseline exports with:

```bash
.venv/bin/python baseline/report_results.py
```

The default report destination is `results/baseline/`. Keep model-specific
results in separate subdirectories and use the shared split for comparisons.
