import matplotlib.pyplot as plt
import pandas as pd

from gru_classifier.train import OUT, RUNS

FIG = OUT / "figures"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")


def plot_heatmap(summary, row, col, name, value="val_f1"):
    pivot = summary.pivot(index=row, columns=col, values=value).sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > pivot.values.mean() else "black")
    fig.colorbar(im, ax=ax, label="val macro-F1")
    _save(fig, f"{name}.png")
    return fig


def _label(rid, keys):
    return " ".join(p for p in rid.split("_") if p.startswith(keys)) or rid


def plot_curves(run_ids, name, keys=("lr", "h", "do", "wd", "wdrop")):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for rid in run_ids:
        h = pd.read_csv(RUNS / rid / "history.csv")
        lab = _label(rid, keys)
        axes[0].plot(h.epoch, h.train_loss, marker="o", ms=3, label=lab)
        axes[1].plot(h.epoch, h.val_loss, marker="o", ms=3, label=lab)
        axes[2].plot(h.epoch, h.macro_f1, marker="o", ms=3, label=lab)
    for ax, t in zip(axes, ["train loss", "val loss", "val macro-F1"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    axes[2].legend(fontsize=7)
    _save(fig, f"{name}.png")
    return fig


def plot_gap(run_ids, name, keys=("do", "wd", "wdrop")):
    """Train vs val macro-F1 per epoch: gap = variance (overfitting)."""
    fig, axes = plt.subplots(1, len(run_ids), figsize=(3.6 * len(run_ids), 3.2), sharey=True, squeeze=False)
    for ax, rid in zip(axes[0], run_ids):
        h = pd.read_csv(RUNS / rid / "history.csv")
        ax.plot(h.epoch, h.train_macro_f1, marker="o", ms=3, label="train")
        ax.plot(h.epoch, h.macro_f1, marker="o", ms=3, label="val")
        ax.fill_between(h.epoch, h.macro_f1, h.train_macro_f1, alpha=.15)
        ax.set_title(_label(rid, keys), fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("macro-F1")
    axes[0][0].legend()
    _save(fig, f"{name}.png")
    return fig


UNDER, GOOD, OVER = "#eb6834", "#2a78d6", "#1baf7a"


def plot_fit_regimes(runs: dict, name="fit_regimes"):
    """runs = {"Underfit (...)": run_id, "Good fit (...)": run_id, "Overfit (...)": run_id}"""
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), sharey=True)
    for ax, (title, rid) in zip(axes, runs.items()):
        h = pd.read_csv(RUNS / rid / "history.csv")
        ax.plot(h.epoch, h.train_loss, marker="o", ms=4, color="#52514e", label="train loss")
        ax.plot(h.epoch, h.val_loss, marker="o", ms=4, color=GOOD, label="val loss")
        b = h.val_loss.idxmin()
        ax.axvline(h.epoch[b], color="#a3a29c", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("weighted cross-entropy")
    axes[0].legend()
    _save(fig, f"{name}.png")
    return fig


def plot_complexity(df, name="complexity_curve"):
    """df rows: params, last_train_macro_f1, last_macro_f1, val_macro_f1 (best epoch)."""
    d = df.sort_values("params")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(d.params, 1 - d.last_train_macro_f1, marker="o", ms=7, color="#52514e", label="train error (1 − macro-F1)")
    ax.plot(d.params, 1 - d.last_macro_f1, marker="o", ms=7, color=GOOD, label="val error, last epoch")
    ax.plot(d.params, 1 - d.val_macro_f1, marker="o", ms=7, ls="--", color=OVER, label="val error, early-stopped")
    ax.set_xscale("log")
    for r in d.itertuples():
        ax.annotate(f"h{r.hidden_size}", (r.params, 1 - r.last_train_macro_f1), textcoords="offset points", xytext=(0, -13),
                    fontsize=7, ha="center", color="#52514e")
    ax.set(xlabel="trainable parameters (log)", ylabel="error = 1 − macro-F1",
           title="Capacity sweep without regularization (15 epochs)")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=.3)
    _save(fig, f"{name}.png")
    return fig


def plot_learning_curve(df, name="learning_curve_data_size"):
    """df rows: n_train, train_macro_f1, val_macro_f1 (best epoch)."""
    d = df.sort_values("n_train")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(d.n_train, d.train_macro_f1, marker="o", ms=7, color="#52514e", label="train macro-F1")
    ax.plot(d.n_train, d.val_macro_f1, marker="o", ms=7, color=GOOD, label="val macro-F1")
    ax.fill_between(d.n_train, d.val_macro_f1, d.train_macro_f1, color=OVER, alpha=.12, label="gap (variance)")
    ax.set(xlabel="training reviews", ylabel="macro-F1", title="Learning curve: more data shrinks the gap")
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    _save(fig, f"{name}.png")
    return fig


def plot_threshold(curve, name="threshold_tuning"):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(curve.threshold, curve.macro_f1, color=GOOD, label="macro-F1")
    ax.plot(curve.threshold, curve.neg_recall, color=UNDER, label="negative recall")
    ax.plot(curve.threshold, curve.pos_recall, color="#52514e", label="positive recall")
    b = curve.loc[curve.macro_f1.idxmax()]
    ax.axvline(.5, color="#a3a29c", ls=":", lw=1)
    ax.axvline(b.threshold, color="#a3a29c", ls="--", lw=1)
    ax.text(b.threshold + .01, .05, f"best {b.threshold:.2f}\nmacro-F1 {b.macro_f1:.3f}", fontsize=9)
    ax.set(xlabel="decision threshold on P(positive)", ylim=(0, 1.02), title="Threshold tuning (ensemble, val)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=.3)
    _save(fig, f"{name}.png")
    return fig


def plot_reg_curve(df, name="regularization_curve"):
    """df rows: weight_decay, train_macro_f1, val_macro_f1 (best epoch)."""
    d = df.sort_values("weight_decay").reset_index(drop=True)
    x = range(len(d))
    tr, va = 1 - d.train_macro_f1, 1 - d.val_macro_f1
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, tr, marker="o", ms=7, color="#52514e", label="train error")
    ax.plot(x, va, marker="o", ms=7, color=GOOD, label="val error")
    ax.fill_between(x, tr, va, color=OVER, alpha=.12, label="gap = variance")
    b = int(va.idxmin())
    ax.axvline(b, color="#a3a29c", ls="--", lw=1)
    ax.text(0, va.max(), "overfitting\n(low bias, high variance)", fontsize=9, color="#52514e", va="top")
    ax.text(len(d) - 1.25, va.max(), "underfitting\n(high bias)", fontsize=9, color="#52514e", va="top", ha="right")
    ax.text(b + .08, tr.max() * .5 + va.min() * .5, f"best L2 = {d.weight_decay[b]:g}", fontsize=9)
    ax.set_xticks(list(x), [f"{w:g}" for w in d.weight_decay])
    ax.set(xlabel="L2 weight decay (Adam)", ylabel="error = 1 − macro-F1 (best epoch)",
           title="Regularization strength: overfit ↔ underfit")
    ax.legend(loc="center left", fontsize=8)
    ax.grid(alpha=.3)
    _save(fig, f"{name}.png")
    return fig
