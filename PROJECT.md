# IE4483 Mini Project (Option 1): Sentiment of Product Reviews

**Deadline:** Friday 13 Nov 2026, 11:59pm. Submit to NTULearn: **report + `submission.csv`**.
Group of 3, and the report must list members and each member's contribution.

Full brief: [docs/IE4483-Project1.pdf](docs/IE4483-Project1.pdf) · Task split: [docs/task-split.jpg](docs/task-split.jpg)

## Data (in `data/`, copied from the NTULearn course files)
- `train.json`: 7,401 reviews, fields `reviews` (raw text) and `sentiments` (0 = negative, 1 = positive).
  **Imbalanced: 6,319 positive (85%) / 1,082 negative (15%).** Always report F1 or macro-F1 as well as accuracy.
  An "always positive" classifier already scores about 85% accuracy.
- `test.json`: 1,851 reviews, `reviews` only.
- `submission_template.csv`: the course-provided `submission.csv`. Columns `reviews,predictions`, 1,851 rows,
  same order as `test.json`, all predictions 0. **Our `submission.csv` must keep this format and order.**
- `sampleSubmission.csv`: `id,label`, 500 rows. Does not match this dataset (probably from Option 2); ignore it.
- Example notebook from the course: `tfidf_features_example.ipynb` (repo root).
- Note: the brief calls them "csv files" and says `test.csv` in (d), but the data files are JSON.

## Report questions (marks)
| | Question | Marks | Owner |
|---|---|---|---|
| a | Feature format + preprocessing | 10% | P1 |
| b | Model: architecture figure, I/O dims, loss, training, code + run instructions, reproducibility (seed) | 20% | P2 |
| c | How hyperparameters (LR, etc.) were chosen and why | 10% | P2 |
| d | Predict on test set, submit `submission.csv` | 20% | P3 |
| e | 1–2 correct/incorrect test samples: strength and weakness | 10% | P3 |
| f | Feature format vs resource use and accuracy | 10% | P1 |
| g | Hotel reviews with **no labels**: performance + how to adapt | 10% | P1 |
| h | Hotel reviews with **noisy labels**: effect + 3 improvement methods | 10% | P3 |

## Task split
**Person 1: Data and baseline** (a, f, g · 30%)
- Load data; make a **shared train/validation split** (fixed seed) everyone uses.
- TF-IDF + logistic regression baseline; fit TF-IDF on **training reviews only**.
- Document preprocessing (a).
- Compare TF-IDF vs Person 2's features: validation accuracy, runtime, memory (f).
- Discuss adapting to unlabeled hotel reviews (g).

**Person 2: Neural model and tuning** (b, c · 30%)
- Embedding → GRU → linear classifier.
- Architecture figure, I/O dims, loss, training procedure, code, run instructions (b).
- Sweep learning rates and hidden sizes on the shared validation split; record seed, results, reason for final choice (c).

**Person 3: Evaluation and submission** (d, e, h · 40%)
- Write evaluation + CSV export code using Person 1's baseline first.
- After final model is chosen: generate `submission.csv`, check row order, row count, label mapping (0 neg, 1 pos) (d).
- Discuss 1–2 test reviews (one correct, one wrong) for strength and weakness (e).
- Noisy hotel ratings: likely effect + **three** ways to improve (h).

## Dependencies
P1 split → P2 tuning and P3 eval code. P1 baseline → P3 export pipeline. P2 final model → P3 `submission.csv` and (e). P2 features → P1 comparison (f).
