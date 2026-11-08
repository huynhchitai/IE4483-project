"""Architecture figure for report (b). Run: .venv/bin/python -m gru_classifier.architecture"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from gru_classifier.plots import FIG

INK, INK2, BLUE, ORANGE, AQUA, LIGHT = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#f4f3f0"

STEPS = [  # (title, detail, shape, colour)
    ("Raw review", '"Not comfortable at all, returned them."', "string", LIGHT),
    ("Tokenize + vocabulary", "lowercase · keep not / n't / ! ? · 7,025-word vocab · truncate 200", "(B, T)  T ≤ 200", LIGHT),
    ("Embedding", "7,025 × 128, padding_idx 0  ·  dropout 0.5", "(B, T, 128)", "#dbe8f9"),
    ("Bidirectional GRU", "hidden 128 per direction · packed sequences (padding ignored)", "h_fwd, h_bwd: (B, 128)", "#dbe8f9"),
    ("Concat final states", "[h_fwd ; h_bwd]  ·  dropout 0.5", "(B, 256)", "#dbe8f9"),
    ("Linear", "256 → 2", "logits (B, 2)", "#dbe8f9"),
    ("Softmax → P(positive)", "weighted cross-entropy (neg 3.42, pos 0.59) in training", "(B,)", LIGHT),
    ("Decision", "ensemble of 3 seeds · positive if P ≥ 0.42", "0 = negative, 1 = positive", LIGHT),
]

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8.6, 10.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(STEPS) * 1.25 + .2)
    ax.axis("off")
    for i, (title, detail, shape, col) in enumerate(STEPS):
        y = (len(STEPS) - 1 - i) * 1.25 + .35
        ax.add_patch(FancyBboxPatch((.3, y), 6.4, .85, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=col, ec=BLUE if col != LIGHT else "#a3a29c", lw=1.4))
        ax.text(.55, y + .58, title, fontsize=11.5, fontweight="bold", color=INK, va="center")
        ax.text(.55, y + .24, detail, fontsize=8.6, color=INK2, va="center")
        ax.text(6.95, y + .42, shape, fontsize=9.5, color=BLUE, va="center", family="monospace")
        if i < len(STEPS) - 1:
            ax.annotate("", xy=(3.5, y - .38), xytext=(3.5, y - .02), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
    ax.text(.3, len(STEPS) * 1.25 + .05, "BiGRU sentiment classifier  —  1,097,858 parameters  (B = batch 64)",
            fontsize=12, fontweight="bold", color=INK)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", FIG / "architecture.png")
