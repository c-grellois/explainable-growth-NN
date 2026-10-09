"""Sanity summary of feature caches: shapes, class count, per-class counts,
protocol metadata, and joint nearest-class-mean accuracy on L2-normalized
features (a feature-quality number, NOT a continual-learning result).

  python cache_summary.py out/ptm_*.npz
"""
import sys
import glob

import numpy as np


def ncm(Xtr, ytr, Xte, yte):
    def n(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xn, Xt = n(Xtr.astype(np.float32)), n(Xte.astype(np.float32))
    cls = np.unique(ytr)
    M = n(np.stack([Xn[ytr == c].mean(0) for c in cls]))
    return float((cls[(Xt @ M.T).argmax(1)] == yte).mean())


paths = [p for a in sys.argv[1:] for p in sorted(glob.glob(a))]
if not paths:
    raise SystemExit("usage: cache_summary.py <npz or glob> ...")
for p in paths:
    d = np.load(p, allow_pickle=False)
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    n_cls = int(d["n_classes"]) if "n_classes" in d.files else int(ytr.max()) + 1
    inc = int(d["inc"]) if "inc" in d.files else "(none: 10 assumed)"
    ds = str(d["dataset"]) if "dataset" in d.files else "?"
    bb = str(d["backbone"]) if "backbone" in d.files else "?"
    per = np.bincount(ytr, minlength=n_cls)
    nan = bool(np.isnan(Xtr.astype(np.float32)).any() or np.isnan(Xte.astype(np.float32)).any())
    print(f"{p}\n  dataset {ds} | backbone {bb}\n"
          f"  train {Xtr.shape} test {Xte.shape} | classes {n_cls} | inc {inc} "
          f"-> {int(np.ceil(n_cls / inc)) if isinstance(inc, int) else '?'} tasks\n"
          f"  train/class min {per.min()} median {int(np.median(per))} max {per.max()} | "
          f"test classes present {len(np.unique(yte))} | NaN {nan}\n"
          f"  joint NCM (feature quality only) {ncm(Xtr, ytr, Xte, yte):.4f}")
