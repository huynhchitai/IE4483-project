# IE4483 Mini Project (Option 1): Sentiment of Product Reviews

Binary sentiment classifier (0 = negative, 1 = positive) for Amazon product reviews.
**Due Friday 13 Nov 2026, 11:59pm on NTULearn:** the report and `submission.csv`.

| Member | Role | Report sections |
|---|---|---|
| Vũ Ngọc Linh | Person 1: data, shared split, TF-IDF + LogReg baseline | (a), (f), (g) · 30% |
| Tai Huynh | Person 2: Embedding → GRU → Linear, hyperparameter tuning | (b), (c) · 30% |
| Richard Yeo | Person 3: evaluation, `submission.csv`, error analysis | (d), (e), (h) · 40% |

The full brief, marks and dependencies are in [PROJECT.md](PROJECT.md). The original PDFs are in [docs/](docs/).

## Quick start

### Shared train/validation split

Use `data/train_splitted.json` (5,921 rows: 866 negative, 5,055 positive)
and `data/valid_splitted.json` (1,480 rows: 216 negative, 1,264 positive)
for all baseline and GRU experiments. Both retain the original `reviews` and
`sentiments` fields. Fit vocabularies, TF-IDF and class weights on the training
partition only.

The split targets 80/20 per class with seed 42, keeping all identical review
strings together. It retains all 7,401 rows, preserves original row order within
each output, and has zero exact-text overlap between partitions. Grouping uses
the original text, including case and whitespace; near-duplicates may remain.
`data/split_manifest.json` records source/output fingerprints, indices and the
precise deterministic algorithm. Regenerate with `python analysis/split_data.py`
(standard library only). `analysis/eda.py` below runs the earlier exploratory
row-level split; its baseline scores are not scores on this shared split.

See [dataset analysis](reports/dataset_analysis.pdf) and
[train-only split preparation](reports/train_split_analysis.pdf) for descriptive
statistics and diagnostic simulations.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python analysis/eda.py      # reproduces every number in this README
```

```
data/train.json               7,401 labelled reviews  {"reviews", "sentiments"}
data/test.json                1,851 unlabelled reviews {"reviews"}
data/submission_template.csv  the course's submission.csv: reviews,predictions (all 0), test.json order
data/sampleSubmission.csv     id,label, 500 rows. Does NOT match this dataset; ignore it
tfidf_features_example.ipynb  course example notebook
analysis/eda.py               data analysis + quick baselines
```

---

## 1. Where the data comes from

The brief cites **Blitzer, Dredze & Pereira (ACL 2007)**, which introduced the
*Multi-Domain Sentiment Dataset* of Amazon reviews ([paper][acl07], [dataset page][mdsd]).

- **How labels were made.** Each review has a 1–5 star rating. Reviews with **> 3 stars are positive** and reviews with **< 3 stars are negative**. 3-star reviews were **discarded** as ambiguous ([paper][acl07], §3).
  So the labels are *derived from star ratings*, not hand-annotated, and the rating and the text can disagree. This matters for (h).
- **Which domain.** The 2007 paper used only four domains: books, DVDs, electronics and kitchen. Our reviews are about **clothing and shoes**: the most frequent content words include *comfortable, shoes, size, wear, fit, pair, bra*.
  This is most likely the *apparel* domain from the larger version 2.0 release ([v2.0 page][mdsd]). The official page doesn't list its domains, so this is our inference, not confirmed.
- **Balance.** The paper's labelled sets were artificially balanced at 1,000 positive / 1,000 negative per domain. **Ours is not:** 85% positive. That looks like the natural Amazon rating distribution, so we cannot compare our accuracy directly with the paper's.
- **The paper's reference numbers.** A linear classifier on unigram + bigram features, trained with 1,600 and tested on 400 balanced reviews, reached in-domain accuracy of
  books 80.4%, DVD 82.4%, electronics 84.4%, kitchen 87.7% ([paper][acl07], Fig. 1).

## 2. What the data looks like

All numbers come from `analysis/eda.py`.

| | Value |
|---|---|
| Train / test size | 7,401 / 1,851 |
| Class balance (train) | **6,319 positive (85.4%) / 1,082 negative (14.6%)** |
| Review length (words) | median 42, mean 56.5, p90 108, p95 142, p99 267, max 1,045, min 4 |
| Length by class | negative median **49** vs positive median **41** |
| Duplicate texts in train | 216 extra copies (151 texts, up to 5 copies each). No conflicting labels |
| Test reviews also in train | **101** (exact text match) |
| HTML, URLs, empty reviews | none |
| Contains a newline / `!!` / ALL-CAPS word | 16.5% / 5.1% / 9.0% |

**Words that separate the classes** (smoothed log-odds, words seen ≥ 20 times):

- **Negative:** *poorly, ripped, worst, waste, returned, horrible, poor, returning, terrible, refund, policy, disappointed, china, awful, return, manufacturer, smell, wool, polyester, broke, mistake*
- **Positive:** *lightweight, enjoy, highly, compliments, loves, fantastic, comfortable/"confortable", clarks, sneaker, nike, terrific, classy, pleased, roomy, functional, easy*

**Negation and contrast words** (share of reviews containing the word):

| Word | Negative reviews | Positive reviews |
|---|---|---|
| *not* | **52%** | 23% |
| *n't* | 32% | 23% |
| *but* | 40% | 31% |
| *however* | 7% | 3% |
| *return* | 9% | 1% |

**Quick baselines** (stratified 80/20 split, seed 42, `C=4`, `sublinear_tf`, `min_df=2`):

| Model | Accuracy | Macro-F1 | Negative recall | Time |
|---|---|---|---|---|
| Always predict positive | 0.853 | 0.460 | 0.000 | – |
| TF-IDF unigram + LogReg | 0.919 | 0.809 | 0.548 | 0.1 s |
| TF-IDF unigram + LogReg, `class_weight="balanced"` | 0.910 | 0.836 | 0.816 | 0.1 s |
| TF-IDF uni+bigram + LogReg | 0.918 | 0.793 | 0.479 | 0.3 s |
| **TF-IDF uni+bigram + LogReg, balanced** | **0.926** | **0.858** | 0.793 | 0.3 s |

The best TF-IDF model has 37,919 features and 446k non-zeros (sparse). These validation numbers are slightly optimistic because of the duplicates (insight 3).

## 3. Key insights

1. **Accuracy alone is misleading.** Predicting "positive" for everything already gives 85.3%. Report **macro-F1 and negative-class recall** alongside accuracy, and show a confusion matrix.
2. **Class imbalance hurts the minority class.** Without re-weighting, the bigram model catches only 48% of negative reviews. With `class_weight="balanced"` it catches 79%, and macro-F1 rises from 0.79 to 0.86.
3. **Duplicates leak into validation.** In a naive random 80/20 split, **73 of 1,481 validation reviews** also appear word-for-word in the training part, which inflates validation scores. 101 test reviews are also in train, so predictions on those are partly memorisation.
4. **Negation and contrast carry the signal.** *not* appears in over half of the negative reviews, and *but* / *however* are more common there too. Negative reviews are often mixed: "love the look **but** it ripped after one wash". Order-insensitive bag-of-words misses this; a sequence model like the GRU should do better here.
5. **Negative sentiment is domain-specific.** It is about returns and refunds (*returned, refund, policy*), materials (*polyester, wool, smell*), fit (*small*: 9% of negative vs 5% of positive reviews), and origin (*china*).
   Positive sentiment is about comfort and compliments. These words will not transfer to hotel reviews, which is the core of (g).
6. **Minimal cleaning is needed.** There is no HTML or URLs. Do keep stop-words like *not / no / n't*, which standard stop-word lists remove. There are typos (*confortable*), so subword or character features would help.
7. **The labels are noisy by construction.** They come from star ratings, and 3-star reviews were dropped. Some "positive" reviews read negatively and vice versa, which is real material for (e) and (h).

## 4. Recommendations

### For everyone
- **Use one shared, de-duplicated split.** Split by unique review text (or drop duplicates first) with stratification and `random_state=42`. Save the indices to `data/split.json` so all three of us use exactly the same validation set.
- **Select models by validation macro-F1**, not accuracy.
- **The baseline to beat** is TF-IDF uni+bigram + balanced LogReg: accuracy 0.926 / macro-F1 0.858.

### Person 1: data and baseline (a, f, g)
- **(a)** Lowercase and tokenise, keeping negations and apostrophes. Fit TF-IDF on the training part only (unigram + bigram, `sublinear_tf`, `min_df=2`). Use `class_weight="balanced"`.
- **(f)** Compare on the same split:
  - TF-IDF: sparse, ~38k dims, fits in under a second, but no word order.
  - GRU embeddings: dense, word order, minutes of training, more memory.
  - Optionally GloVe or BERT features.

  Report accuracy, macro-F1, train/inference time and memory in one table.
- **(g)** Hotel reviews without labels is *domain adaptation*. Draw on the paper:
  - For books vs kitchen, 85–90% of each domain's top-1,000 informative features were not in the other domain's top 1,000 ([acl07], §4).
  - **SCL / SCL-MI** uses unlabelled target reviews to align domain-specific words through shared "pivot" words such as *excellent* and *awful*.
  - The **A-distance**, computed from unlabelled data, predicts how much accuracy is lost when moving to a domain (§6).
  - Labelling just **50 target reviews** to correct the alignment brought the average relative error reduction to 46% (§5).
  - Modern options: pseudo-labelling or self-training on hotel reviews, and pretrained language models (BERT) that already know general sentiment.

### Person 2: GRU model and tuning (b, c)
- **Sequence length.** The median is 42 words and p95 is about 145, so use `max_len` 150–200 with truncation and `pack_padded_sequence`. Mention the long tail (max 1,045) in the report.
- **Vocabulary.** Build it from the training part only, with `min_freq=2`, plus `<pad>` and `<unk>`. Do **not** remove negation words.
- **Imbalance.** Use a weighted loss: `BCEWithLogitsLoss(pos_weight=…)` or `CrossEntropyLoss(weight=…)`, with weights from the class ratio of about 5.8 : 1. Alternatively, tune the decision threshold on validation.
- **Architecture.** Embedding (100–300 dims, optionally initialised with GloVe [5]) → **bidirectional GRU** → final hidden state or mean-pooling → dropout → Linear.
  7.4k training reviews is small for a model trained from scratch. If the GRU does not beat TF-IDF, that is a legitimate result to discuss in (f).
- **(c) Tuning grid.**
  - Learning rate {1e-3, 3e-3, 1e-2} × hidden size {64, 128, 256}, with Adam.
  - Early stopping on validation macro-F1.
  - Fixed seed, ideally averaged over 3 seeds.
  - Log every run to a CSV so the table and plots for (c) come for free.

### Person 3: evaluation and submission (d, e, h)
- **(d)** Write `submission.csv` with columns **`reviews,predictions`** and **1,851 rows in `test.json` order**, the same as `data/submission_template.csv`. Assert row count, order and the label set {0, 1} before submitting.
- **(e)** Pick examples that are **not** among the 101 test reviews also found in train, otherwise "correct" may just be memorisation.
  - Good candidates: negation ("not bad at all"), contrast ("love the colour but it fell apart"), sarcasm, and very long reviews that get truncated.
- **(h)** The data already has rating-based label noise (insight 7). Ways to improve:
  - Robust losses or label smoothing.
  - Detect and filter noisy labels: confident learning, or small-loss selection as in co-teaching.
  - Use the noisy star rating as a soft or ordinal target instead of a hard 0/1.

## References

1. J. Blitzer, M. Dredze, F. Pereira. *Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification.* ACL 2007. [PDF][acl07]
2. Multi-Domain Sentiment Dataset, v2.0 ([page][mdsd]) and v1 ([page][mdsd1]). Johns Hopkins University.
3. T. Mikolov et al. *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781, 2013.
4. J. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
5. J. Pennington, R. Socher, C. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014. https://nlp.stanford.edu/projects/glove/

[acl07]: https://www.cs.jhu.edu/~mdredze/publications/sentiment_acl07.pdf
[mdsd]: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
[mdsd1]: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
