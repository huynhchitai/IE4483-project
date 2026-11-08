"""Train / checkpoint-resume / grid search / predict.
results/gru_classifier/runs/<run_id>/: last.pt (each epoch), best.pt, history.csv, done.json
"""
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from common.data import OUTPUT_DIR, get_split, load_test
from common.metrics import evaluate
from gru_classifier.model import GRUClassifier, count_parameters
from gru_classifier.text import PAD_ID, UNK_ID, Vocab

OUT = OUTPUT_DIR / "gru_classifier"
RUNS = OUT / "runs"


@dataclass(frozen=True)
class Config:
    lr: float = 1e-3
    hidden_size: int = 128
    embed_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.3
    max_len: int = 200
    min_freq: int = 2
    batch_size: int = 64
    max_epochs: int = 15
    patience: int = 3
    weight_decay: float = 0.0   # L2
    word_dropout: float = 0.0   # augmentation: random tokens -> <unk>
    pooling: str = "last"       # "last" | "meanmax"
    train_frac: float = 1.0     # learning curve over data size
    class_weighted: bool = True
    grad_clip: float = 1.0
    seed: int = 42

    @property
    def run_id(self) -> str:
        d = "bi" if self.bidirectional else "uni"
        rid = (f"lr{self.lr:g}_h{self.hidden_size}_e{self.embed_dim}_l{self.num_layers}{d}"
               f"_do{self.dropout:g}_len{self.max_len}_cw{int(self.class_weighted)}_ep{self.max_epochs}")
        if self.weight_decay:
            rid += f"_wd{self.weight_decay:g}"
        if self.word_dropout:
            rid += f"_wdrop{self.word_dropout:g}"
        if self.pooling != "last":
            rid += f"_{self.pooling}"
        if self.train_frac < 1:
            rid += f"_tf{self.train_frac:g}"
        if self.patience != 3:
            rid += f"_pat{self.patience}"
        return rid + f"_s{self.seed}"


def pick_device(device="auto"):
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # MPS not faster here


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(texts, labels, vocab, cfg, shuffle, epoch_seed=0):
    encoded = [vocab.encode(t, cfg.max_len) for t in texts]
    labels = list(labels) if labels is not None else [0] * len(encoded)

    def collate(batch):
        seqs, ys = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs])
        ids = torch.full((len(seqs), int(lengths.max())), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        return ids, lengths, torch.tensor(ys, dtype=torch.long)

    g = torch.Generator().manual_seed(epoch_seed)
    return DataLoader(list(zip(encoded, labels)), batch_size=cfg.batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=g if shuffle else None)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    out = [torch.softmax(model(ids.to(device), lengths), 1)[:, 1].cpu() for ids, lengths, _ in loader]
    return torch.cat(out).numpy()


def class_weights(y):
    counts = np.bincount(np.asarray(y), minlength=2)
    return torch.tensor(len(y) / (2 * counts), dtype=torch.float)


def weighted_nll(proba, y, weight):
    log_p = torch.log(torch.tensor(np.stack([1 - proba, proba], 1), dtype=torch.float).clamp_min(1e-9))
    return float(nn.functional.nll_loss(log_p, torch.tensor(y), weight=weight))


def train_run(cfg: Config, device="auto", verbose=True) -> dict:
    run_dir = RUNS / cfg.run_id
    done = run_dir / "done.json"
    if done.exists():
        if verbose:
            print(f"[skip] {cfg.run_id}")
        return json.loads(done.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)

    train_df, val_df = get_split()
    if cfg.train_frac < 1:
        train_df, _ = train_test_split(train_df, train_size=cfg.train_frac, stratify=train_df["sentiments"], random_state=42)
    vocab = Vocab.build(train_df["reviews"], cfg.min_freq)
    y_tr, y_val = train_df["sentiments"].to_numpy(), val_df["sentiments"].to_numpy()
    train_eval_loader = make_loader(train_df["reviews"], y_tr, vocab, cfg, shuffle=False)
    val_loader = make_loader(val_df["reviews"], y_val, vocab, cfg, shuffle=False)

    set_seed(cfg.seed)
    model = GRUClassifier(len(vocab), cfg.embed_dim, cfg.hidden_size, cfg.num_layers,
                          cfg.bidirectional, cfg.dropout, pooling=cfg.pooling).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weight = class_weights(y_tr) if cfg.class_weighted else None
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(dev) if weight is not None else None)

    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "bad_epochs": 0, "history": []}
    last = run_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng"])
        state = ckpt["state"]
        if verbose:
            print(f"[resume] {cfg.run_id} from epoch {state['epoch']}")
    elif verbose:
        print(f"[start] {cfg.run_id} | vocab {len(vocab)} | params {count_parameters(model):,} | {dev}")

    while state["epoch"] < cfg.max_epochs and state["bad_epochs"] < cfg.patience:
        epoch = state["epoch"] + 1
        t0 = time.time()
        model.train()
        loader = make_loader(train_df["reviews"], y_tr, vocab, cfg, shuffle=True, epoch_seed=cfg.seed * 1000 + epoch)
        for ids, lengths, y in loader:
            if cfg.word_dropout:
                drop = (torch.rand(ids.shape) < cfg.word_dropout) & (ids != PAD_ID)
                ids = ids.masked_fill(drop, UNK_ID)
            opt.zero_grad()
            loss = loss_fn(model(ids.to(dev), lengths), y.to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        p_tr, p_val = predict_proba(model, train_eval_loader, dev), predict_proba(model, val_loader, dev)
        m_tr, m = evaluate(y_tr, (p_tr >= .5).astype(int)), evaluate(y_val, (p_val >= .5).astype(int))
        row = {"epoch": epoch, "train_loss": weighted_nll(p_tr, y_tr, weight), "val_loss": weighted_nll(p_val, y_val, weight),
               "train_macro_f1": m_tr["macro_f1"], "train_accuracy": m_tr["accuracy"],
               **{k: m[k] for k in ("accuracy", "macro_f1", "neg_recall", "pos_recall")},
               "seconds": time.time() - t0}
        state["history"].append(row)
        state["epoch"] = epoch

        if m["macro_f1"] > state["best_f1"]:
            state.update(best_f1=m["macro_f1"], best_epoch=epoch, bad_epochs=0)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "itos": vocab.itos,
                        "epoch": epoch, "val_metrics": m, "train_metrics": m_tr}, run_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "torch_rng": torch.get_rng_state(), "state": state}, last)
        pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
        if verbose:
            print(f"  ep {epoch:2d} | loss tr {row['train_loss']:.3f} val {row['val_loss']:.3f} | "
                  f"F1 tr {m_tr['macro_f1']:.3f} val {m['macro_f1']:.3f} | acc {m['accuracy']:.3f} | "
                  f"{row['seconds']:.1f}s" + (" *" if state["best_epoch"] == epoch else ""))

    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {"run_id": cfg.run_id, **asdict(cfg), "best_epoch": state["best_epoch"], "epochs_run": state["epoch"],
               **{f"val_{k}": v for k, v in best["val_metrics"].items()},
               "train_macro_f1": best.get("train_metrics", {}).get("macro_f1"),
               **{f"last_{k}": state["history"][-1].get(k) for k in ("train_loss", "val_loss", "train_macro_f1", "macro_f1")},
               "train_seconds": sum(r["seconds"] for r in state["history"]),
               "params": count_parameters(model), "vocab_size": len(vocab), "n_train": len(train_df)}
    done.write_text(json.dumps(summary, indent=2))
    last.unlink(missing_ok=True)
    load_results()
    return summary


def load_results() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*/done.json"))]
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "results.csv", index=False)
    return df


def _init_worker(threads):
    torch.set_num_threads(threads)


def _run_quiet(cfg):
    return train_run(cfg, verbose=False)


def run_many(cfgs, workers=1, verbose=True) -> pd.DataFrame:
    """Train configs (skipping finished ones), optionally `workers` processes in parallel."""
    todo = [c for c in cfgs if not (RUNS / c.run_id / "done.json").exists()]
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        threads = max(1, torch.get_num_threads() // workers)
        with ProcessPoolExecutor(min(workers, len(todo)), mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=(threads,)) as ex:
            futs = [ex.submit(_run_quiet, c) for c in todo]
            for f in as_completed(futs):
                r = f.result()
                if verbose:
                    print(f"[done] {r['run_id']} | val F1 {r['val_macro_f1']:.4f} | {r['train_seconds']:.0f}s", flush=True)
    else:
        for c in todo:
            train_run(c, verbose=verbose)
    out = pd.DataFrame([json.loads((RUNS / c.run_id / "done.json").read_text()) for c in cfgs])
    load_results()
    return out


def grid_search(params: dict, seeds=(42,), base=Config(), workers=1, verbose=True) -> pd.DataFrame:
    """params e.g. {"lr": [...], "hidden_size": [...]}. Finished runs are skipped."""
    combos = [dict(zip(params, v)) for v in product(*params.values())]
    return run_many([replace(base, **c, seed=s) for c in combos for s in seeds], workers, verbose)


def summarize(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Mean/std over seeds per setting, best first."""
    g = df.groupby(list(keys)).agg(
        val_f1=("val_macro_f1", "mean"), val_f1_std=("val_macro_f1", "std"),
        train_f1=("train_macro_f1", "mean"), val_acc=("val_accuracy", "mean"),
        neg_recall=("val_neg_recall", "mean"), best_epoch=("best_epoch", "mean"),
        seconds=("train_seconds", "mean"), n_seeds=("seed", "count"))
    g.insert(3, "gap", g.train_f1 - g.val_f1)
    return g.sort_values("val_f1", ascending=False).round(4).reset_index()


def load_model(run_id, device="cpu"):
    ckpt = torch.load(RUNS / run_id / "best.pt", map_location=device, weights_only=False)
    cfg, vocab = Config(**ckpt["config"]), Vocab(ckpt["itos"])
    model = GRUClassifier(len(vocab), cfg.embed_dim, cfg.hidden_size, cfg.num_layers,
                          cfg.bidirectional, cfg.dropout, pooling=cfg.pooling).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), vocab, cfg


def predict_texts(run_id, texts, device="cpu"):
    model, vocab, cfg = load_model(run_id, device)
    return predict_proba(model, make_loader(list(texts), None, vocab, cfg, shuffle=False), torch.device(device))


def predict_test(run_ids, threshold=0.5, device="cpu") -> pd.DataFrame:
    """Average P(pos) over run_ids (seed ensemble) -> results/gru_classifier/test_predictions.csv, test.json order."""
    run_ids = [run_ids] if isinstance(run_ids, str) else list(run_ids)
    test = load_test()
    proba = np.mean([predict_texts(r, test["reviews"], device) for r in run_ids], axis=0)
    out = pd.DataFrame({"reviews": test["reviews"], "predictions": (proba >= threshold).astype(int),
                        "prob_positive": proba.round(5)})
    out.to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "final_model.json").write_text(json.dumps({"run_ids": run_ids, "threshold": threshold}, indent=1))
    return out
