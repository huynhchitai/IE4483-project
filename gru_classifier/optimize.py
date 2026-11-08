"""Post-training: seed ensemble + decision threshold (tuned on val)."""
import numpy as np
import pandas as pd

from common.data import get_split
from common.metrics import evaluate
from gru_classifier.train import predict_texts


def val_proba(run_ids):
    _, val_df = get_split()
    return val_df["sentiments"].to_numpy(), np.mean([predict_texts(r, val_df["reviews"]) for r in run_ids], axis=0)


def threshold_curve(y, p, grid=np.arange(0.05, 0.96, 0.01)):
    return pd.DataFrame([{"threshold": round(t, 2), **evaluate(y, (p >= t).astype(int))} for t in grid])


def compare(run_ids):
    """Single models vs ensemble, at 0.5 and at the val-tuned threshold."""
    rows, probs = [], {}
    y = None
    for r in run_ids:
        y, probs[r] = val_proba([r])
    probs["ensemble"] = np.mean(list(probs.values()), axis=0)
    for name, p in probs.items():
        curve = threshold_curve(y, p)
        best = curve.loc[curve.macro_f1.idxmax()]
        m = evaluate(y, (p >= .5).astype(int))
        rows.append({"model": name if name == "ensemble" else "seed " + name.rsplit("_s", 1)[1],
                     "macro_f1@0.5": m["macro_f1"], "acc@0.5": m["accuracy"], "neg_recall@0.5": m["neg_recall"],
                     "best_threshold": best.threshold, "macro_f1@best": best.macro_f1,
                     "acc@best": best.accuracy, "neg_recall@best": best.neg_recall})
    return pd.DataFrame(rows).round(4), threshold_curve(y, probs["ensemble"])
