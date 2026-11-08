import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results"
MANIFEST = DATA_DIR / "split_manifest.json"   # official shared split (analysis/split_data.py)


def load_train() -> pd.DataFrame:
    return pd.read_json(DATA_DIR / "train.json")


def load_test() -> pd.DataFrame:
    return pd.read_json(DATA_DIR / "test.json")


def get_split():
    """(train_df, val_df) from data/split_manifest.json; index = row in train.json."""
    m = json.loads(MANIFEST.read_text())
    df = load_train()
    return df.loc[m["train_indices"]], df.loc[m["validation_indices"]]


def describe_split() -> pd.DataFrame:
    tr, va = get_split()
    return pd.DataFrame([{"part": n, "rows": len(d), "negative": int((d.sentiments == 0).sum()),
                          "positive": int(d.sentiments.sum()), "positive %": round(100 * d.sentiments.mean(), 1),
                          "texts also in train": int(d.reviews.isin(set(tr.reviews)).sum()) if n == "val" else 0}
                         for n, d in [("train", tr), ("val", va)]])
