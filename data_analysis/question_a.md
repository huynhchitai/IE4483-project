# (a) Feature format and data preprocessing

> **Scope:** This section describes the TF-IDF representation used by the
> prepared Logistic Regression baseline in [`baseline/train.py`](../baseline/train.py).
> It documents the implemented preprocessing configuration, rather than claiming
> completed model training. If the final submission also includes the planned
> GRU, its actual sequence-tokenization and embedding procedure must be added
> after that implementation is available.

## Choice of feature format

We choose **word-level TF-IDF features containing both unigrams and bigrams**
as the input representation for our Logistic Regression sentiment classifier.
A unigram represents an individual token, such as `comfortable`, whereas a
bigram represents two adjacent tokens, such as `not comfortable`. Combining
them allows the baseline to represent individual sentiment cues and short
phrases involving negation. However, this representation does not preserve
complete sentence order or long-range context.

Each review is converted into a fixed-dimensional, sparse numerical vector.
Each dimension corresponds to one retained unigram or bigram, and its value
is the normalized TF-IDF weight of that feature in the review. TF-IDF weights
are based on within-review frequency and training-corpus document frequency;
they are not sentiment scores. The classifier subsequently learns how these
features relate to the sentiment labels.

This format provides a practical baseline for our relatively small labelled
dataset, requires no pretrained embeddings, and can be supplied directly to a
linear classifier. Sparse storage avoids explicitly storing the many zero
entries for features absent from an individual review. We will evaluate its
predictive performance rather than assume that it is superior to other formats.

## Preprocessing procedure

### 1. Load and validate the raw records

Although the assignment refers to CSV files, the supplied review datasets are
JSON arrays. The labelled source, `data/train.json`, contains **7,401 records**
with two fields: `reviews` (raw text) and `sentiments` (integer target). The
label mapping is **0 = negative** and **1 = positive**. The label is kept
separate from the text input and is never included as a TF-IDF feature.

We check that review strings are non-empty and labels have valid values.
The supplied labelled dataset contains no missing or whitespace-only reviews,
so no text imputation or removal of empty records is necessary.

### 2. Use the shared split before learning text features

We use the saved approximately 80/20 split with seed 42. The split is stratified
by sentiment while assigning identical review strings to the same partition.
Duplicate rows are retained, but cannot occur on opposite sides of the split.
The resulting partitions are:

| Partition | Reviews | Negative (0) | Positive (1) |
|---|---:|---:|---:|
| `data/train_splitted.json` | 5,921 | 866 | 5,055 |
| `data/valid_splitted.json` | 1,480 | 216 | 1,264 |

The loading code checks source/output file fingerprints and source-row indices
against `data/split_manifest.json`, complete coverage of the original records,
and zero exact-text overlap. Grouping uses the original case and whitespace;
near-duplicate reviews are not guaranteed to be separated from validation.

### 3. Normalize case and tokenize

We lowercase the review text so that capitalization variants such as `Good`
and `good` share a feature. We then extract word tokens using:

```python
token_pattern = r"(?u)\b\w+(?:['’]\w+)*\b"
```

This pattern retains word characters, including digits and single-character
tokens, and preserves internal straight or curly apostrophes in contractions
such as `don't`. Whitespace and other punctuation delimit tokens; punctuation
such as `!` and `?` is not represented as a separate feature. Bigrams are formed
from the resulting token sequence, so they can span punctuation boundaries.
The tokenizer does not normalize straight and curly apostrophes to each other.

We do **not** remove stop words, because words such as `not`, `no`, and `but`
can contribute to sentiment and contrast. We also do not apply stemming,
lemmatization, spelling correction, accent stripping, or additional HTML/URL
cleaning in this baseline. The original JSON text remains unchanged; these
transformations occur inside the vectorizer.

### 4. Construct the training vocabulary

We generate unigrams and bigrams using `ngram_range=(1, 2)`. With `min_df=2`,
a feature is retained only if it appears in at least **two training reviews**.
This removes singleton features, which may include idiosyncratic spellings and
rare expressions. We do not impose an explicit maximum vocabulary size.

Both vocabulary construction and document-frequency estimation use only the
5,921 training reviews. Validation and official test reviews do not contribute
to either step. Terms absent from the retained vocabulary are ignored during
transformation; no `<unk>` feature is added.

### 5. Calculate and normalize TF-IDF weights

Let $c(t,d)$ be the occurrence count of feature $t$ in review $d$, $N$ the
number of training reviews, and $\mathrm{df}(t)$ the number of training reviews
containing $t$. We use sublinear term frequency and smoothed inverse document
frequency:

$$
\mathrm{tf}(t,d) =
\begin{cases}
1 + \ln c(t,d), & c(t,d) > 0, \\
0, & c(t,d) = 0,
\end{cases}
$$

$$
\mathrm{idf}(t) = \ln\left(\frac{1+N}{1+\mathrm{df}(t)}\right)+1,
\qquad
u_{d,t}=\mathrm{tf}(t,d)\,\mathrm{idf}(t).
$$

For each nonzero review vector, we apply L2 normalization:

$$
\mathbf{x}_d = \frac{\mathbf{u}_d}{\lVert\mathbf{u}_d\rVert_2}.
$$

Sublinear TF reduces the effect of repeating the same feature, while IDF
reduces the relative weight of features common across training reviews.
Normalization controls vector magnitude across reviews of different lengths.
A review with no retained features remains an all-zero vector.

### 6. Supply consistent numerical inputs to the classifier

The vectorizer produces a **sparse matrix of float64 values**. If the fitted
vocabulary contains $V$ features, the training input has shape
$(5921,V)$ and the validation input has shape $(1480,V)$. Their target arrays
have shapes $(5921,)$ and $(1480,)$ respectively. The value of $V$ is determined
when fitting the training vocabulary; the vocabulary count from descriptive
analysis of the full labelled pool is not the fitted model dimension.

The vectorizer and classifier are combined in a scikit-learn `Pipeline`.
Calling `fit` on training data learns the vocabulary, IDF weights and
classifier. Validation data is processed with the same fitted vectorizer
without refitting. When the official test set is eventually predicted, it
must use the vectorizer paired with the chosen fitted classifier as well.
TF-IDF requires no sequence padding or truncation.

The training data is imbalanced, but we do not resample records during feature
construction. The prepared baseline addresses imbalance through
`class_weight='balanced'` in the classifier, which is a training choice rather
than a text-preprocessing operation.

## Implementation settings

```python
TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+(?:['’]\w+)*\b",
    stop_words=None,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    smooth_idf=True,
    norm="l2",
    dtype=np.float64,
)
```

These are the prepared default settings, not the outcome of a completed
hyperparameter search. Final feature counts and experimental performance
should be reported after training.

**Implementation and reproducibility sources:**

- [`baseline/train.py`](../baseline/train.py): tokenization, vectorization and pipeline.
- [`analysis/split_data.py`](../analysis/split_data.py): shared split generation.
- [`data/split_manifest.json`](../data/split_manifest.json): split indices and fingerprints.
- [`reports/dataset_statistics.json`](../reports/dataset_statistics.json): source-data quality checks.
