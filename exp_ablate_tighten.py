"""Ablation: which tightening mechanism carries class-IL performance?

Four cells on split-MNIST, identical geometry everywhere (PCA proximity,
q_cov_in 0.99, veto theta 0.985): no tightening / scalar only / veto only /
scalar + veto. The prediction from the error decomposition: scalar tightening
is net-harmful once vetoes exist, because its self-loss releases own-class
samples that downstream organs then answer WRONGLY (correct -> wrong), while
vetoes cut false accepts at far lower self-loss (wrong -> abstain at worst).

Run:  python exp_ablate_tighten.py [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import load_real, prepare, split_tasks
from exp_continual import run_ours, CFG

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CELLS = {
    "none":        dict(self_budget=0.0,  veto=False),
    "scalar_only": dict(self_budget=0.08, veto=False),
    "veto_only":   dict(self_budget=0.0,  veto=True),
    "both":        dict(self_budget=0.08, veto=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    X, y, C = load_real("mnist")
    out = {}
    for cell, over in CELLS.items():
        cfg = {**CFG["mnist"], **over}
        rows = []
        for seed in range(args.seeds):
            splits = prepare(X, y, seed=seed)
            tasks = split_tasks(y, 2)
            torch.manual_seed(seed)
            _, r = run_ours(tasks, splits, X.shape[1], cfg, seed)
            rows.append({"strict": r["final"]["strict"],
                         "coverage": r["final"]["coverage"],
                         "precision": r["final"]["precision"],
                         "growth_flips": r["growth_flips"]})
        out[cell] = rows
        s = [r["strict"] for r in rows]
        print(f"{cell:12s}: strict {np.mean(s):.3f} +- {np.std(s):.3f}   "
              f"cov {np.mean([r['coverage'] for r in rows]):.3f}   "
              f"prec {np.mean([r['precision'] for r in rows]):.3f}   "
              f"growth flips {sum(r['growth_flips'] for r in rows)}")

    with open(os.path.join(RES, "ablate_tighten.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {os.path.join(RES, 'ablate_tighten.json')}")


if __name__ == "__main__":
    main()
