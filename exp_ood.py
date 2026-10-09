"""Head-to-head measurement of proximity geometries as OOD detectors.

For each dataset and each 2-class task, fit each geometry on the task's train
data only, threshold at the own-val q-quantile (0.97, matching the organ's
q_cov_in), and measure the FOREIGN ACCEPT RATE on all other classes' val data --
the exact quantity that drives frozen error in class-incremental learning --
plus threshold-free AUROC.

Run:  python exp_ood.py [--datasets mnist kmnist fashion] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import load_real, prepare, split_tasks
from cgrow.proximity import make_prox

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)

QS = [0.90, 0.95, 0.97, 0.99]
GEOMS = ["kmeans", "pca", "ae", "combo", "disc"]
KW = {"kmeans": dict(k_per_class=25), "pca": dict(n_components=24), "ae": dict(),
      "combo": dict(), "disc": dict()}


def auroc(pos, neg):
    """AUROC for 'foreign has higher score': rank-based, no sklearn needed."""
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64)
    n_p, n_n = len(pos), len(neg)
    return float((r[:n_p].sum() - n_p * (n_p - 1) / 2) / (n_p * n_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "kmnist", "fashion"])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    out = {}
    for ds in args.datasets:
        X, y, C = load_real(ds)
        rows = {g: {"far": {q: [] for q in QS}, "auroc": [], "fit_s": []}
                for g in GEOMS}
        for seed in range(args.seeds):
            (Xtr, ytr), (Xva, yva), _ = prepare(X, y, seed=seed)
            for cls in split_tasks(y, 2):
                own_tr = np.isin(ytr, cls)
                own_va = np.isin(yva, cls)
                Xf = Xva[~own_va]
                if len(Xf) > 4000:
                    Xf = Xf[:4000]
                for g in GEOMS:
                    torch.manual_seed(seed)
                    t0 = time.time()
                    prox = make_prox(g, seed=seed, **KW[g]).fit(Xtr[own_tr], ytr[own_tr])
                    fit_s = time.time() - t0
                    s_own = prox.score(Xva[own_va])
                    s_for = prox.score(Xf)
                    for q in QS:
                        thr = np.quantile(s_own, q)
                        rows[g]["far"][q].append(float((s_for <= thr).mean()))
                    rows[g]["auroc"].append(auroc(s_for, s_own))
                    rows[g]["fit_s"].append(fit_s)
        out[ds] = {}
        print(f"\n[{ds}]  foreign-accept rate at own-coverage q (lower is better)")
        print(f"  {'geom':7s}  " + "  ".join(f"q={q:.2f}" for q in QS)
              + "   AUROC   fit")
        for g in GEOMS:
            far, roc = rows[g]["far"], rows[g]["auroc"]
            out[ds][g] = {"far": {str(q): {"mean": float(np.mean(far[q])),
                                           "sd": float(np.std(far[q]))} for q in QS},
                          "auroc_mean": float(np.mean(roc)),
                          "fit_s": float(np.mean(rows[g]["fit_s"]))}
            print(f"  {g:7s}  " + "  ".join(f"{np.mean(far[q]):.3f} " for q in QS)
                  + f"  {np.mean(roc):.3f}   {np.mean(rows[g]['fit_s']):.1f}s")

    with open(os.path.join(RES, "ood_geometry.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(RES, 'ood_geometry.json')}")


if __name__ == "__main__":
    main()
