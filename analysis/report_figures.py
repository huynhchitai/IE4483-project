"""Figures + numbers for data_analysis/data_analysis.md.  Run: .venv/bin/python analysis/report_figures.py"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.data import get_split, load_test, load_train  # noqa: E402

FIG = ROOT / "data_analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42

# reference palette (validated): blue / orange / aqua; text inks; surface
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEG, POS = ORANGE, BLUE
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})
N = {}  # numbers for the report


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def gru_tokenize(s):  # same as gru_classifier/text.py
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[!?]", s.lower())


def words(s):  # descriptive tokenizer used in the PDFs
    return re.findall(r"[a-z']+", s.lower())


train, test = load_train(), load_test()
tr_df, va_df = get_split()
train["n"] = train.reviews.map(lambda s: len(words(s)))
test["n"] = test.reviews.map(lambda s: len(words(s)))
train["n_gru"] = train.reviews.map(lambda s: len(gru_tokenize(s)))

# ---------------------------------------------------------------- 1 labels
c = train.sentiments.value_counts().sort_index()
N["labels"] = {"neg": int(c[0]), "pos": int(c[1]), "ratio": round(c[1] / c[0], 2), "majority_acc": round(c[1] / c.sum(), 4)}
fig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.barh(["Negative (0)", "Positive (1)"], [c[0], c[1]], color=[NEG, POS], height=.55)
for y, v in enumerate([c[0], c[1]]):
    ax.text(v + 80, y, f"{v:,}  ({100 * v / c.sum():.1f}%)", va="center", color=INK)
ax.set_xlim(0, 8000)
ax.set_xlabel("training reviews")
ax.grid(axis="y", visible=False)
ax.set_title("Label distribution (train) — 5.84 : 1")
save(fig, "01_labels")

# ---------------------------------------------------------------- 2 lengths
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
bins = np.arange(0, 330, 10)
for d, col, lab in [(train, BLUE, "train"), (test, AQUA, "test")]:
    w = np.full(len(d), 100 / len(d))
    axes[0].hist(d.n.clip(upper=320), bins=bins, weights=w, histtype="step", lw=2, color=col, label=lab)
axes[0].set(xlabel="tokens per review (clipped at 320)", ylabel="% of reviews", title="Review length")
axes[0].legend()
cov = {}
for L in (100, 150, 200, 300):
    cov[L] = round(100 * (train.n_gru <= L).mean(), 1)
x = np.sort(train.n_gru.values)
axes[1].plot(x, 100 * np.arange(1, len(x) + 1) / len(x), color=BLUE)
for L in (150, 200):
    axes[1].axvline(L, color=GRAY, lw=1, ls="--")
    axes[1].text(L + 8, 60 if L == 150 else 45, f"max_len {L}\ncovers {cov[L]}%", color=INK2, fontsize=9)
axes[1].set(xlim=(0, 600), xlabel="tokens (GRU tokenizer)", ylabel="cumulative % of train reviews",
            title="Coverage of max_len")
save(fig, "02_length")
N["length"] = {"train": train.n.describe(percentiles=[.5, .9, .95, .99]).round(1).to_dict(),
               "test": test.n.describe(percentiles=[.5, .9, .95, .99]).round(1).to_dict(), "coverage_gru": cov}

# ---------------------------------------------------------------- 3 length by class
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for lab, col, name in [(0, NEG, "negative"), (1, POS, "positive")]:
    x = np.sort(train.loc[train.sentiments == lab, "n"].values)
    ax.plot(x, 100 * np.arange(1, len(x) + 1) / len(x), color=col, label=f"{name} (median {int(np.median(x))})")
ax.set(xlim=(0, 350), xlabel="tokens per review", ylabel="cumulative % within class",
       title="Negative reviews are longer")
ax.legend(loc="lower right")
save(fig, "03_length_by_class")

# ---------------------------------------------------------------- 4 duplicates
g = train.groupby("reviews").size()
sizes = g.value_counts().sort_index()
N["dup"] = {"unique": int(len(g)), "extra_rows": int(len(train) - len(g)), "groups_multi": int((g > 1).sum()),
            "test_in_train": int(test.reviews.isin(set(train.reviews)).sum())}
fig, ax = plt.subplots(figsize=(6.4, 3))
ax.bar(sizes.index.astype(str), sizes.values, color=BLUE, width=.6)
ax.set_yscale("log")
for i, v in enumerate(sizes.values):
    ax.text(i, v * 1.15, f"{v:,}", ha="center", color=INK, fontsize=9)
ax.set(xlabel="copies of the same review text", ylabel="groups (log)", title="Exact duplicate groups (train)")
ax.grid(axis="x", visible=False)
save(fig, "04_duplicates")

# ---------------------------------------------------------------- 5 split leakage
Xtr, Xva = train_test_split(train.reviews, test_size=.2, stratify=train.sentiments, random_state=SEED)
naive_leak = int(Xva.isin(set(Xtr)).sum())
grouped_leak = int(va_df.reviews.isin(set(tr_df.reviews)).sum())
N["split"] = {"naive_leak": naive_leak, "grouped_leak": grouped_leak, "train": len(tr_df), "val": len(va_df),
              "val_neg": int((va_df.sentiments == 0).sum()), "val_pos": int(va_df.sentiments.sum())}
fig, axes = plt.subplots(1, 2, figsize=(11, 2.8))
axes[0].barh(["row-level\n(train_test_split)", "grouped by text\n(split_manifest.json)"], [naive_leak, grouped_leak],
             color=[ORANGE, BLUE], height=.5)
for y, v in enumerate([naive_leak, grouped_leak]):
    axes[0].text(v + 1, y, str(v), va="center")
axes[0].set(xlabel="validation rows whose exact text is in train", title="Leakage, 80/20 seed 42")
axes[0].grid(axis="y", visible=False)
parts = [("train", tr_df), ("val", va_df)]
for i, (name, d) in enumerate(parts):
    p = 100 * d.sentiments.mean()
    axes[1].barh(name, 100 - p, color=NEG, height=.5)
    axes[1].barh(name, p, left=100 - p, color=POS, height=.5)
    axes[1].text(102, i, f"{len(d):,} rows · {p:.1f}% pos", va="center", fontsize=9)
axes[1].set(xlim=(0, 140), xticks=[0, 25, 50, 75, 100], xlabel="% of rows  (orange = negative, blue = positive)", title="Shared split keeps the ratio")
axes[1].grid(axis="y", visible=False)
save(fig, "05_split")

# ---------------------------------------------------------------- 6 Naive Bayes log-odds words
pos_c = Counter(w for s in train.loc[train.sentiments == 1, "reviews"] for w in set(words(s)))
neg_c = Counter(w for s in train.loc[train.sentiments == 0, "reviews"] for w in set(words(s)))
n_pos, n_neg = c[1], c[0]
lo = pd.Series({w: np.log((pos_c[w] + 1) / (n_pos + 2)) - np.log((neg_c[w] + 1) / (n_neg + 2))
                for w in set(pos_c) | set(neg_c) if pos_c[w] + neg_c[w] >= 20}).sort_values()
top = pd.concat([lo.head(15), lo.tail(15)])
fig, ax = plt.subplots(figsize=(6.4, 7))
ax.barh(top.index, top.values, color=[NEG if v < 0 else POS for v in top.values], height=.7)
ax.axvline(0, color=INK2, lw=1)
ax.set(xlabel="log P(word | pos) − log P(word | neg)   (Laplace smoothing)",
       title="Most class-indicative words (Naive Bayes log-likelihood ratio)")
ax.grid(axis="y", visible=False)
save(fig, "06_logodds_words")
N["logodds"] = {"neg": list(lo.head(15).index), "pos": list(lo.tail(15).index[::-1])}

# ---------------------------------------------------------------- 7 negation / contrast
cue = ["not", "n't", "no", "but", "however", "return", "refund", "small", "size", "fit", "love", "great"]
low = train.reviews.str.lower()
rows = []
for w in cue:
    pat = r"n't" if w == "n't" else rf"\b{w}\b"
    has = low.str.contains(pat, regex=True)
    rows.append((w, 100 * has[train.sentiments == 0].mean(), 100 * has[train.sentiments == 1].mean()))
cdf = pd.DataFrame(rows, columns=["word", "neg", "pos"])
N["cues"] = cdf.round(1).to_dict("records")
fig, ax = plt.subplots(figsize=(10, 3.2))
xi = np.arange(len(cdf))
ax.bar(xi - .19, cdf.neg, width=.36, color=NEG, label="negative reviews")
ax.bar(xi + .19, cdf.pos, width=.36, color=POS, label="positive reviews")
ax.set_xticks(xi, cdf.word)
ax.set(ylabel="% of reviews containing", title="Negation, contrast and topic cues")
ax.legend()
ax.grid(axis="x", visible=False)
save(fig, "07_cues")

# ---------------------------------------------------------------- 8 vocabulary (Zipf + min_freq)
tok_counts = Counter(t for s in tr_df.reviews for t in gru_tokenize(s))
freq = np.array(sorted(tok_counts.values(), reverse=True))
mf = [1, 2, 3, 5, 10]
vsize = [int((freq >= m).sum()) + 2 for m in mf]
val_tokens = [t for s in va_df.reviews for t in gru_tokenize(s)]
oov = [round(100 * np.mean([tok_counts.get(t, 0) < m for t in val_tokens]), 2) for m in mf]
N["vocab"] = {"min_freq": mf, "vocab_size": vsize, "val_unk_pct": oov, "singletons": int((freq == 1).sum())}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
axes[0].loglog(np.arange(1, len(freq) + 1), freq, color=BLUE)
axes[0].set(xlabel="word rank", ylabel="count", title="Zipf's law (train split)")
axes[1].plot(mf, vsize, marker="o", ms=8, color=BLUE)
for m, v, o in zip(mf, vsize, oov):
    axes[1].annotate(f"{v:,} words\n{o}% val <unk>", (m, v), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK2)
axes[1].set(xlabel="min_freq", ylabel="vocabulary size", title="Vocabulary vs <unk> rate", xticks=mf, ylim=(1500, 15500))
save(fig, "08_vocab")

# ---------------------------------------------------------------- 9-10 unsupervised: SVD + K-means
vec = TfidfVectorizer(sublinear_tf=True, min_df=3, stop_words="english")
X = vec.fit_transform(train.reviews)
svd = TruncatedSVD(100, random_state=SEED)
Z = normalize(svd.fit_transform(X))
N["svd_var_100"] = round(float(svd.explained_variance_ratio_.sum()), 3)
rng = np.random.default_rng(SEED)
idx = rng.choice(len(train), 3000, replace=False)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for lab, col, name in [(1, POS, "positive"), (0, NEG, "negative")]:
    m = train.sentiments.values[idx] == lab
    ax.scatter(Z[idx][m, 0], Z[idx][m, 1], s=8, alpha=.5, color=col, label=name, edgecolors="none")
ax.set(xlabel="SVD component 1", ylabel="SVD component 2", title="TF-IDF → 2-D (LSA / PCA-style projection)")
ax.legend(markerscale=2)
save(fig, "09_svd_scatter")

ks = list(range(2, 11))
nmi, purity_neg = [], []
for k in ks:
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    nmi.append(normalized_mutual_info_score(train.sentiments, lab))
    ct = pd.crosstab(lab, train.sentiments)
    purity_neg.append((ct[0] / ct.sum(axis=1)).max())
km6 = KMeans(6, n_init=10, random_state=SEED).fit(Z)
centers = svd.inverse_transform(km6.cluster_centers_)
terms = np.array(vec.get_feature_names_out())
clusters = []
for i in range(6):
    m = km6.labels_ == i
    clusters.append({"cluster": i, "size": int(m.sum()), "neg_pct": round(100 * (train.sentiments.values[m] == 0).mean(), 1),
                     "top_terms": ", ".join(terms[np.argsort(centers[i])[::-1][:8]])})
N["kmeans"] = {"k": ks, "nmi": [round(v, 4) for v in nmi], "max_neg_purity": [round(v, 3) for v in purity_neg], "k6": clusters}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(ks, nmi, marker="o", ms=8, color=BLUE)
axes[0].set(xlabel="k", ylabel="NMI(cluster, sentiment)", title="K-means clusters vs sentiment", ylim=(0, max(.1, max(nmi) * 1.4)))
cl = pd.DataFrame(clusters).sort_values("neg_pct")
axes[1].barh([f"C{r.cluster} ({r.size})" for r in cl.itertuples()], cl.neg_pct, color=ORANGE, height=.55)
axes[1].axvline(100 * c[0] / c.sum(), color=INK2, lw=1, ls="--")
axes[1].text(100 * c[0] / c.sum() + .3, 5.35, "overall 14.6%", fontsize=8, color=INK2)
axes[1].set(xlabel="% negative in cluster", title="k = 6: clusters ≈ product topics")
axes[1].grid(axis="y", visible=False)
save(fig, "10_kmeans")

(FIG / "numbers.json").write_text(json.dumps(N, indent=1, default=str))
print(json.dumps(N, indent=1, default=str)[:6000])
