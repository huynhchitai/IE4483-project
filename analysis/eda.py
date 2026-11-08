"""Exploratory analysis of the IE4483 product-review sentiment data.

Run from the repo root:  .venv/bin/python analysis/eda.py
Prints the numbers quoted in README.md.
"""
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

SEED = 42
DATA = Path(__file__).resolve().parent.parent / "data"

train = pd.read_json(DATA / "train.json")
test = pd.read_json(DATA / "test.json")


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def section(title):
    print(f"\n=== {title} ===")


section("Size and class balance")
print("train", train.shape, "test", test.shape)
vc = train["sentiments"].value_counts()
print(vc.to_dict(), "positive share = %.1f%%" % (100 * vc[1] / len(train)))

section("Length in words")
for name, df in [("train", train), ("test", test)]:
    n = df["reviews"].map(lambda s: len(words(s)))
    print(name, "mean %.1f median %d p90 %d p95 %d p99 %d max %d min %d"
          % (n.mean(), n.median(), n.quantile(.9), n.quantile(.95), n.quantile(.99), n.max(), n.min()))
train["n_words"] = train["reviews"].map(lambda s: len(words(s)))
print("by class:", train.groupby("sentiments")["n_words"].agg(["mean", "median"]).round(1).to_dict())

section("Duplicates and leakage")
print("duplicate texts within train:", train["reviews"].duplicated().sum())
print("duplicate texts within test:", test["reviews"].duplicated().sum())
dup_conflict = train.groupby("reviews")["sentiments"].nunique()
print("texts in train with conflicting labels:", int((dup_conflict > 1).sum()))
overlap = set(train["reviews"]) & set(test["reviews"])
print("exact train/test text overlap:", len(overlap))
print("empty / whitespace-only reviews:", int((train["reviews"].str.strip() == "").sum()),
      int((test["reviews"].str.strip() == "").sum()))

section("Noise markers (share of train reviews)")
markers = {
    "html entity (&quot; etc.)": r"&[a-z]+;|&#\d+;",
    "html tag": r"<[^>]+>",
    "url": r"http|www\.",
    "newline": r"\n",
    "emoticon": r"[:;]-?[)(dpDP]",
    "!!": r"!!",
    "ALL-CAPS word (>=4 letters)": r"\b[A-Z]{4,}\b",
}
for k, pat in markers.items():
    print(f"  {k}: {100 * train['reviews'].str.contains(pat, regex=True).mean():.1f}%")

section("Negation and contrast words (share of reviews containing, by class)")
for w in ["not", "n't", "no", "but", "however", "return", "returned", "small", "large", "size", "fit"]:
    pat = r"n't" if w == "n't" else rf"\b{w}\b"
    has = train["reviews"].str.lower().str.contains(pat, regex=True)
    print(f"  {w!r}: neg {100 * has[train.sentiments == 0].mean():.0f}%  pos {100 * has[train.sentiments == 1].mean():.0f}%")

section("Domain check: most common content words")
stop = set("""the a an and or but to of in on for it is was this that i my me with as at be are
have has had so they them you your its it's i'm i've were from just very would will can if all
one get got not no n't""".split())
cnt = Counter(w for s in train["reviews"] for w in words(s) if w not in stop and len(w) > 2)
print([w for w, _ in cnt.most_common(40)])

section("Most class-indicative words (smoothed log-odds, min count 20)")
pos_c = Counter(w for s in train.loc[train.sentiments == 1, "reviews"] for w in set(words(s)))
neg_c = Counter(w for s in train.loc[train.sentiments == 0, "reviews"] for w in set(words(s)))
n_pos, n_neg = (train.sentiments == 1).sum(), (train.sentiments == 0).sum()
rows = []
for w in set(pos_c) | set(neg_c):
    if pos_c[w] + neg_c[w] < 20:
        continue
    lo = np.log((pos_c[w] + 1) / (n_pos + 2)) - np.log((neg_c[w] + 1) / (n_neg + 2))
    rows.append((lo, w))
rows.sort()
print("negative:", [w for _, w in rows[:25]])
print("positive:", [w for _, w in rows[-25:][::-1]])

section("Quick baselines on a stratified 80/20 split (seed 42)")
X_tr, X_va, y_tr, y_va = train_test_split(
    train["reviews"], train["sentiments"], test_size=0.2, stratify=train["sentiments"], random_state=SEED)


def report(name, y_pred, secs=None):
    extra = f"  fit+predict {secs:.1f}s" if secs is not None else ""
    print(f"  {name:45s} acc {accuracy_score(y_va, y_pred):.3f}  macro-F1 {f1_score(y_va, y_pred, average='macro'):.3f}"
          f"  neg-recall {recall_score(y_va, y_pred, pos_label=0):.3f}{extra}")


report("majority class (always positive)", np.ones_like(y_va))
for ngram in [(1, 1), (1, 2)]:
    for cw in [None, "balanced"]:
        t0 = time.time()
        vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=2)
        Xtr = vec.fit_transform(X_tr)
        clf = LogisticRegression(max_iter=2000, class_weight=cw, C=4.0)
        clf.fit(Xtr, y_tr)
        pred = clf.predict(vec.transform(X_va))
        report(f"TF-IDF {ngram} + LogReg, class_weight={cw}", pred, time.time() - t0)
        if ngram == (1, 2) and cw == "balanced":
            print(f"    vocab size {len(vec.vocabulary_)}, matrix nnz {Xtr.nnz}")
