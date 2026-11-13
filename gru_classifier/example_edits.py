"""Edited versions of the two reviews in report section (e). Run: .venv/bin/python -m gru_classifier.example_edits"""
import json
import re

import numpy as np

from common.data import load_test
from gru_classifier.train import OUT, predict_texts

if __name__ == "__main__":
    fm = json.loads((OUT / "final_model.json").read_text())
    test = load_test().reviews
    r1, r2 = test[142], test[134]
    s1, s2 = ([s for s in re.split(r"(?<=[.!?])\s+", r) if s] for r in (r1, r2))
    variants = {
        "review 1: original": r1,
        "review 1: sentence 1 removed": " ".join(s1[1:]),
        "review 1: word 'pleased' removed": r1.replace("pleased ", ""),
        "review 1: sentence 1 only": s1[0],
        "review 1: sentence 1 moved to the end": " ".join(s1[1:] + s1[:1]),
        "review 2: original": r2,
        "review 2: sentence 1 only (praise)": s2[0],
        "review 2: sentences 1-2 (praise + 'But' complaint)": " ".join(s2[:2]),
        "review 2: sentences 1 and 3 (praise + 'I returned them')": " ".join([s2[0], s2[2]]),
        "review 2: 'returned' replaced by 'kept'": r2.replace("returned", "kept"),
    }
    p = np.mean([predict_texts(r, list(variants.values())) for r in fm["run_ids"]], axis=0)
    out = {k: round(float(v), 3) for k, v in zip(variants, p)}
    (OUT / "example_edits.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
