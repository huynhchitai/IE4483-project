# Dataset analysis

Open `dataset_analysis.pdf` for the four-page descriptive report. It covers raw
table dimensions, file sizes, schema, class balance, review lengths, vocabulary,
exact duplicates and submission-template alignment. No model is trained.

Reproduce from the repository root:

```bash
.venv/bin/pip install -r analysis/requirements-report.txt
.venv/bin/python analysis/dataset_report.py
```

The script also exports `dataset_statistics.json` (including source-file SHA-256
fingerprints) and the two chart PNGs. Token counts use `[a-z']+` on lowercased
text, consistent with the existing EDA script; they are not model token counts.

Train/test overlap is **87 unique exact texts covering 101 test rows**. These
are different counting units; the PDF states both explicitly.

## Train-only split preparation

`train_split_analysis.pdf` analyses only the supplied training pool: per-class
lengths, duplicate groups, normalization sensitivity and deterministic 80/20
split simulations. It recommends a grouped stratified split but does not save
official split indices or train a model.

Regenerate with `.venv/bin/python analysis/train_split_report.py`. Exact results
and the input fingerprint are in `train_split_statistics.json`. The diagnostic
row split uses Python's random generator, not the sklearn split in the original
EDA, so its overlap count is specific to this simulation.
