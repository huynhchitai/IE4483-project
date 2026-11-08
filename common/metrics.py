import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "neg_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "pos_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
