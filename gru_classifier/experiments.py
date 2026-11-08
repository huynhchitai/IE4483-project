"""Extra runs for the report: complexity sweep, data-size learning curve, pooling."""
from dataclasses import replace

import pandas as pd

from gru_classifier.train import Config, run_many

SIZES = [(4, 2), (8, 4), (16, 8), (32, 16), (64, 32), (128, 64), (128, 128), (256, 256)]  # (embed, hidden)


def complexity_sweep(base: Config, sizes=SIZES, workers=1) -> pd.DataFrame:
    """No dropout / L2 / early stopping (patience 15) so capacity alone drives fit."""
    return run_many([replace(base, embed_dim=e, hidden_size=h, dropout=0.0, weight_decay=0.0,
                             word_dropout=0.0, patience=15) for e, h in sizes], workers)


def data_size_curve(base: Config, fracs=(0.1, 0.25, 0.5, 0.75, 1.0), workers=1) -> pd.DataFrame:
    return run_many([replace(base, train_frac=f) for f in fracs], workers)


def seed_runs(base: Config, seeds=(42, 7, 123), workers=1) -> pd.DataFrame:
    return run_many([replace(base, seed=s) for s in seeds], workers)


def reg_sweep(base: Config, wds=(0.0, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1), workers=1) -> pd.DataFrame:
    """L2 strength: weak -> overfit (large gap), strong -> underfit (both errors high)."""
    return run_many([replace(base, weight_decay=w) for w in wds], workers)
