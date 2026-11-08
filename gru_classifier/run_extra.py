"""Extra runs (L2 sweep, 25-epoch final). Run: .venv/bin/python -m gru_classifier.run_extra"""
from dataclasses import replace

from gru_classifier.experiments import reg_sweep, seed_runs
from gru_classifier.train import Config

BASE = Config(lr=3e-3, hidden_size=128, dropout=0.5, weight_decay=1e-4)

if __name__ == "__main__":
    print(reg_sweep(BASE, workers=4)[["weight_decay", "train_macro_f1", "val_macro_f1", "best_epoch"]].round(4).to_string(index=False))
    r = seed_runs(replace(BASE, max_epochs=25), workers=4)
    print(r[["seed", "val_macro_f1", "best_epoch"]].round(4).to_string(index=False), "\nmean", round(r.val_macro_f1.mean(), 4))
