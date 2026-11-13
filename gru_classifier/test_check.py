"""Checks behind report section (d). Run: .venv/bin/python -m gru_classifier.test_check"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import ROOT, get_split, load_test
from common.metrics import evaluate
from gru_classifier.optimize import val_proba
from gru_classifier.plots import _save
from gru_classifier.train import OUT

sys.path.insert(0, str(ROOT / "baseline"))
from train import build_model  # noqa: E402

if __name__ == "__main__":
    fm = json.loads((OUT / "final_model.json").read_text())
    thr = fm["threshold"]
    train_df, val_df = get_split()
    y_val, p_val = val_proba(fm["run_ids"])
    pred_val = (p_val >= thr).astype(int)
    test = pd.read_csv(OUT / "test_predictions.csv")
    p_test, pred_test = test.prob_positive.to_numpy(), test.predictions.to_numpy()

    base = build_model().fit(train_df.reviews, train_df.sentiments)
    base_val, base_test = base.predict(val_df.reviews), base.predict(load_test().reviews)

    def sure(p):
        return float(((p < 0.1) | (p > 0.9)).mean())

    out = {
        "threshold": thr,
        "validation": evaluate(y_val, pred_val),
        "predicted_positive_rate": {"validation": float(pred_val.mean()), "test": float(pred_test.mean())},
        "share_confident_below_0.1_or_above_0.9": {"validation": sure(p_val), "test": sure(p_test)},
        "agreement_with_baseline": {"validation": float((pred_val == base_val).mean()),
                                    "test": float((pred_test == base_test).mean()),
                                    "test_disagreements": int((pred_test != base_test).sum())},
        "validation_accuracy_where_models": {"agree": float((pred_val == y_val)[pred_val == base_val].mean()),
                                             "disagree": float((pred_val == y_val)[pred_val != base_val].mean())},
    }
    (OUT / "test_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    m = out["validation"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, a = plt.subplots(figsize=(4.4, 3.8))
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["negative", "positive"], yticklabels=["negative", "positive"],
          xlabel="Predicted", ylabel="True", title="Validation (1,480 labelled reviews)")
    fig.tight_layout()
    _save(fig, "validation_confusion.png")

    fig, b = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(0, 1, 21)
    w = [np.full(len(p), 100 / len(p)) for p in (p_val, p_test)]
    b.hist([p_val, p_test], bins=bins, weights=w, color=["#2a78d6", "#eb6834"], label=["validation", "test"])
    b.axvline(thr, color="#52514e", ls="--", lw=1)
    b.text(thr + 0.01, b.get_ylim()[1] * 0.9, f"threshold {thr}", fontsize=9, color="#52514e")
    b.set(xlabel="Predicted P(positive)", ylabel="% of reviews")
    b.legend()
    fig.tight_layout()
    _save(fig, "test_vs_validation_proba.png")
