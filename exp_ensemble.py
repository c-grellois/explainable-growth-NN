"""Redundant growth: is the merged-twin gain an ENSEMBLE, or abstention-filling?

The twin control (exp_redundant.py) threw up a side observation: concatenating
two independently grown organisms scored 0.909 against parents of 0.829 and
0.873. That looks like a free ensemble. It cannot be one. In a first-accept
cascade the PREFIX wins every contest, so organs appended after organism A can
only answer where A ABSTAINED -- B never overrides A, and A's answers survive
bitwise (checked here, not assumed). Any gain is therefore coverage-filling,
and the real question becomes whether genuine ensembling over the same organ
pool beats the cascade's birth-order prior.

Measured, per k = 1..K independently grown organisms over the same class split:
  first-accept    the concatenated cascade (theorem: 0 flips vs the prefix)
  vote-forced     majority vote of the k organisms' forced predictions
  vote-accepting  majority among organisms that accept (abstain if none)
  arbiter         the trained arbiter over the merged pool
  market-maker    pool-priced auction over the merged pool
plus the channel decomposition -- coverage, precision on covered, and accuracy
on the NEWLY answered set (what the added organs actually contribute) -- and a
capacity control (one organism widened k-fold).

Run:  python exp_ensemble.py [--benchmarks digits] [--kmax 4] [--seeds 5]
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
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import sample_pool
from exp_continual import run_ours, CFG as MLP_CFG
from exp_pfc import evidence, train_arbiter, arbiter_predict
from exp_market import fit_prices_pool, bids, market_predict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def merge(orgs):
    m = Organism()
    m.organs = [o for g in orgs for o in g.organs]
    return m


def vote(preds, abstain_aware):
    """preds: (k, n) predictions. Majority; ties go to the earliest organism."""
    k, n = preds.shape
    out = np.full(n, ABSTAIN, dtype=np.int64)
    for i in range(n):
        col = preds[:, i]
        if abstain_aware:
            col = col[col != ABSTAIN]
            if len(col) == 0:
                continue
        vals, counts = np.unique(col, return_counts=True)
        cands = set(vals[counts == counts.max()].tolist())
        for v in preds[:, i]:            # earliest organism breaks the tie
            if v in cands:
                out[i] = v
                break
    return out


def routing_variants(merged, splits, seed):
    """Arbiter and pool-priced market-maker over the merged organ pool."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    rng = np.random.default_rng(seed * 61 + 7)
    X_syn, y_syn = sample_pool(merged, 2000, rng)
    perm = rng.permutation(len(X_syn))
    ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]

    F_tr, P_tr = evidence(merged, X_syn[tr])
    net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)
    F_ho, P_ho = evidence(merged, X_syn[ho])
    best = (-1.0, None)
    for th in np.linspace(0.3, 0.9, 13):
        s = float((arbiter_predict(net, F_ho, P_ho, th) == y_syn[ho]).mean())
        if s > best[0]:
            best = (s, float(th))
    F_te, P_te = evidence(merged, Xte)
    arb = float((arbiter_predict(net, F_te, P_te, best[1]) == yte).mean())

    curves = fit_prices_pool(merged, X_syn[tr], y_syn[tr])
    B_ho, Pb_ho = bids(merged, curves, X_syn[ho])
    bestr = (-1.0, None)
    for r in np.linspace(0.2, 0.9, 15):
        a, _ = market_predict(B_ho, Pb_ho, r)
        s = float((a == y_syn[ho]).mean())
        if s > bestr[0]:
            bestr = (s, float(r))
    B_te, Pb_te = bids(merged, curves, Xte)
    mk = float((market_predict(B_te, Pb_te, bestr[1])[0] == yte).mean())
    return arb, mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["digits"])
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-name", default="ensemble.json")
    args = ap.parse_args()

    out = {}
    for bench in args.benchmarks:
        cfg = MLP_CFG[bench]
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            X, y, _ = load_real(bench)
            splits = prepare(X, y, seed=seed)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            tasks = split_tasks(y, 2)

            orgs = []
            for j in range(args.kmax):
                s = seed + 100 * j
                torch.manual_seed(s)
                g, _ = run_ours(tasks, splits, X.shape[1], cfg, s)
                orgs.append(g)

            per_k, prev_pred = [], None
            for k in range(1, args.kmax + 1):
                m = merge(orgs[:k])
                r = m.predict(Xte)
                sc = m.scores(Xte, yte)
                ans = r["pred"] != ABSTAIN
                # the retention theorem: prefix answers must be untouched
                flips = (0 if prev_pred is None else
                         int(((prev_pred != ABSTAIN)
                              & (r["pred"] != prev_pred)).sum()))
                # what the ADDED organs contribute: accuracy on newly answered
                if prev_pred is None:
                    new_n, new_acc = 0, None
                else:
                    new = (prev_pred == ABSTAIN) & ans
                    new_n = int(new.sum())
                    new_acc = (float((r["pred"][new] == yte[new]).mean())
                               if new_n else None)
                preds_f = np.stack([g.predict(Xte)["forced"]
                                    for g in orgs[:k]])
                preds_a = np.stack([g.predict(Xte)["pred"] for g in orgs[:k]])
                vf = float((vote(preds_f, False) == yte).mean())
                va = float((vote(preds_a, True) == yte).mean())
                arb, mk = ((None, None) if k == 1
                           else routing_variants(m, splits, seed))
                per_k.append({
                    "k": k, "strict": sc["strict"], "forced": sc["forced"],
                    "coverage": sc["coverage"], "precision": sc["precision"],
                    "organs": len(m.organs), "params": m.n_params(),
                    "flips_vs_prefix": flips,
                    "newly_answered": new_n, "acc_on_newly_answered": new_acc,
                    "vote_forced": vf, "vote_accepting": va,
                    "arbiter": arb, "market_maker": mk})
                prev_pred = r["pred"]

            # capacity control: one organism widened k-fold
            wide = dict(cfg)
            wide["hidden"] = tuple(h * args.kmax for h in cfg["hidden"])
            wide["feat"] = cfg["feat"] * args.kmax
            torch.manual_seed(seed)
            gw, _ = run_ours(tasks, splits, X.shape[1], wide, seed)
            sw = gw.scores(Xte, yte)

            rec = {"seed": seed, "per_k": per_k,
                   "wide": {"strict": sw["strict"], "coverage": sw["coverage"],
                            "precision": sw["precision"],
                            "params": gw.n_params()}}
            recs.append(rec)
            head = " | ".join(
                f"k{p['k']} {p['strict']:.3f}(cov {p['coverage']:.2f})"
                for p in per_k)
            print(f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) {head}",
                  flush=True)
            last = per_k[-1]
            print("    flips vs prefix: "
                  + ",".join(str(p["flips_vs_prefix"]) for p in per_k)
                  + f" | vote-forced {last['vote_forced']:.3f}"
                  f" vote-acc {last['vote_accepting']:.3f}"
                  f" arbiter {last['arbiter']:.3f}"
                  f" maker {last['market_maker']:.3f}"
                  f" | wide {sw['strict']:.3f}"
                  f" ({gw.n_params()} vs {last['params']} params)", flush=True)
            out[bench] = recs
            with open(os.path.join(RES, args.out_name), "w") as f:
                json.dump(out, f, indent=1, default=float)
    print(f"wrote results/{args.out_name}")


if __name__ == "__main__":
    main()
