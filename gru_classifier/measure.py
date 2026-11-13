"""Resource measurements for (f). Run: .venv/bin/python -m gru_classifier.measure"""
import json
import platform
import resource
import time

import torch

from common.data import get_split
from common.metrics import evaluate
from gru_classifier.train import OUT, RUNS, load_model, make_loader, predict_proba

if __name__ == "__main__":
    torch.set_num_threads(8)
    fm = json.loads((OUT / "final_model.json").read_text())
    _, val_df = get_split()
    model, vocab, cfg = load_model(fm["run_ids"][0])
    loader = make_loader(val_df["reviews"], val_df["sentiments"], vocab, cfg, shuffle=False)
    predict_proba(model, loader, torch.device("cpu"))  # warm-up
    t = time.perf_counter()
    p = predict_proba(model, loader, torch.device("cpu"))
    single = time.perf_counter() - t
    hist = [json.loads((RUNS / r / "done.json").read_text()) for r in fm["run_ids"]]
    out = {
        "hardware": f"{platform.processor() or platform.machine()} CPU, torch {torch.__version__}, 8 threads",
        "params": sum(x.numel() for x in model.parameters()),
        "embedding_matrix_mib": round(model.embedding.weight.numel() * 4 / 2**20, 2),
        "checkpoint_mib": round((RUNS / fm["run_ids"][0] / "best.pt").stat().st_size / 2**20, 2),
        "val_predict_seconds_single": round(single, 3),
        "val_predict_ms_per_review": round(1000 * single / len(val_df), 3),
        "train_seconds_per_run_mean": round(sum(h["train_seconds"] for h in hist) / len(hist), 1),
        "epochs_run": [h["epochs_run"] for h in hist],
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),  # macOS: bytes
        "val_single_seed42": {k: round(v, 4) for k, v in evaluate(val_df.sentiments, (p >= .5).astype(int)).items()},
    }
    (OUT / "resources.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
