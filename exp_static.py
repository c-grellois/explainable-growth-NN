"""B1 -- static tasks: growth-by-abstention vs a single organ vs a plain MLP.

The organism development loop on a fixed task: found, then repeatedly
(tighten each organ away from its measured val mistakes under a small budget,
then grow a fresh organ on the mass the organism declines). Growth is gated:
a candidate joins only if it answers net-positively on the declined val mass.

Comparisons: a single prototypical organ (the contracts note's §6.5 method) and
a plain MLP, plus certified input robustness (certAcc@eps) which the plain MLP
does not have at all.

Run:  python exp_static.py [--datasets digits letter ...] [--seeds N]
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
from cgrow.data import load_real, prepare
from cgrow.organ import Organ
from cgrow.organism import Organism, ABSTAIN
from cgrow.baselines import train_plain, accuracy

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)

CFG = {
    "digits": dict(hidden=(64,), feat=16, cap=2.0, epochs=60, p_target=0.96,
                   b_hidden=(128,), b_epochs=40, max_train=None, eps=0.1),
    "letter": dict(hidden=(128, 64), feat=32, cap=3.0, epochs=60, p_target=0.85,
                   b_hidden=(128,), b_epochs=40, max_train=None, eps=0.014),
    "letter10": dict(hidden=(96,), feat=24, cap=2.5, epochs=40, p_target=0.9,
                     b_hidden=(128,), b_epochs=40, max_train=None, eps=0.014),
    "mnist": dict(hidden=(256,), feat=32, cap=2.5, epochs=20, p_target=0.96,
                  b_hidden=(256,), b_epochs=15, max_train=None, eps=0.1),
    "kmnist": dict(hidden=(256,), feat=32, cap=2.5, epochs=20, p_target=0.9,
                   b_hidden=(256,), b_epochs=15, max_train=None, eps=0.1),
    "fashion": dict(hidden=(256,), feat=32, cap=2.5, epochs=20, p_target=0.88,
                    b_hidden=(256,), b_epochs=15, max_train=None, eps=0.1),
}
MAX_ORGANS = 8


def grow_static(splits, in_dim, C, cfg, seed):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    org = Organism()
    stages = []
    rejects = 0
    for k in range(MAX_ORGANS):
        if org.organs:
            # tighten each organ away from its measured val mistakes
            r = org.predict(Xva)
            wrong = (r["pred"] != ABSTAIN) & (r["pred"] != yva)
            if wrong.any():
                for i, o in enumerate(org.organs):
                    sel = wrong & (r["organ"] == i)
                    if sel.sum() >= 5:
                        d1, m, _, d_in, _, _ = o.evaluate(Xva[sel])
                        o.solve_tightening(d1, m, d_in, self_budget=0.02)
            need = org.declined_mask(Xtr)
        else:
            need = np.ones(len(Xtr), bool)
        Xn, yn = Xtr[need], ytr[need]
        classes = [c for c in np.unique(yn) if (yn == c).sum() >= 12]
        if len(classes) < 1 or need.sum() < 50:
            break
        cand = Organ(in_dim, cfg["feat"], cfg["hidden"], cap=cfg["cap"],
                     class_ids=classes, seed=seed * 100 + k)
        cand.train_proto(Xn, yn, epochs=cfg["epochs"])
        # declining precision ladder: take the best slice still available
        ladder = [p for p in (0.97, 0.94, 0.9, 0.85, 0.8, 0.72, 0.65)
                  if p >= cfg["p_target"] - 0.25]
        if org.try_grow(cand, Xva, yva, p_target=(cfg["p_target"] if k == 0 else ladder),
                        gate=(k > 0), label=f"o{k}"):
            rejects = 0
        else:
            rejects += 1
            if rejects >= 2:
                break
            continue
        stages.append(org.scores(Xte, yte))
    return org, stages


def cert_acc(org, Xte, yte, eps):
    r = org.predict(Xte)
    return float(((r["pred"] == yte) & (r["radius"] >= eps)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["digits", "letter"])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    all_rec = {}
    for ds in args.datasets:
        cfg = CFG[ds]
        X, y, C = load_real(ds)
        records = []
        for seed in range(args.seeds):
            t0 = time.time()
            splits = prepare(X, y, seed=seed, max_train=cfg["max_train"])
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            torch.manual_seed(seed)

            org, stages = grow_static(splits, X.shape[1], C, cfg, seed)
            final = org.scores(Xte, yte)
            ca = cert_acc(org, Xte, yte, cfg["eps"])

            # single organ, same per-organ architecture, full data
            single = Organ(X.shape[1], cfg["feat"], cfg["hidden"], cap=cfg["cap"],
                           class_ids=list(range(C)), seed=seed)
            single.train_proto(Xtr, ytr, epochs=cfg["epochs"] * 2)
            single.solve_thresholds(Xva, yva, p_target=cfg["p_target"], require=False)
            single.freeze(Xva, yva)
            so = Organism(); so.organs = [single]
            s_final = so.scores(Xte, yte)
            s_ca = cert_acc(so, Xte, yte, cfg["eps"])

            plain = train_plain(Xtr, ytr, C, hidden=cfg["b_hidden"],
                                epochs=cfg["b_epochs"], seed=seed)
            p_acc = accuracy(plain, Xte, yte)[0]

            rec = {"seed": seed, "n_organs": len(org.organs),
                   "stages": stages, "final": final, "certacc": ca,
                   "single": s_final, "single_certacc": s_ca,
                   "plain_acc": p_acc,
                   "params": {"organism": org.n_params(),
                              "single": single.n_params(),
                              "plain": plain.n_params()},
                   "log": org.log}
            records.append(rec)
            print(f"[{ds} seed {seed}] ({time.time()-t0:.0f}s) organs={len(org.organs)}")
            print(f"  organism: strict {final['strict']:.3f}  forced {final['forced']:.3f}"
                  f"  cov {final['coverage']:.3f}  prec {final['precision']:.3f}"
                  f"  certAcc@{cfg['eps']} {ca:.3f}  params {org.n_params()}")
            print(f"  single  : strict {s_final['strict']:.3f}  forced {s_final['forced']:.3f}"
                  f"  cov {s_final['coverage']:.3f}  certAcc {s_ca:.3f}  params {single.n_params()}")
            print(f"  plain   : acc {p_acc:.3f}  certAcc 0 (no certificate)  params {plain.n_params()}")
        all_rec[ds] = records

    out = os.path.join(RES, "static.json")
    existing = {}
    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
    existing.update(all_rec)
    with open(out, "w") as f:
        json.dump(existing, f, indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
