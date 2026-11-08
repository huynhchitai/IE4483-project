"""Print GRU results.  Run: .venv/bin/python -m gru_classifier.summary"""
import json

import pandas as pd

from gru_classifier.train import OUT, load_results

pd.set_option("display.width", 200, "display.max_columns", 20)
df = load_results()
cols = ["run_id", "val_macro_f1", "val_accuracy", "val_neg_recall", "train_macro_f1", "best_epoch", "train_seconds"]
print(f"{len(df)} finished runs\n")
print(df[cols].sort_values("val_macro_f1", ascending=False).round(4).head(15).to_string(index=False))
final = OUT / "final_model.json"
if final.exists():
    print("\nfinal model:", json.dumps(json.loads(final.read_text()), indent=1))
