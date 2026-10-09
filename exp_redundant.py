"""The control apoptosis was missing: CAN the v4 certificate ever kill?

Session-15 measured 0 deaths under per-class death certificates and concluded
the grown anatomy is minimal -- no organ is redundant at per-class resolution.
That conclusion is only sound if the certificate is capable of firing at all.
This builds an anatomy that is redundant BY CONSTRUCTION and asks it to.

  Twin organisms: grow organism A on the class split with seed s, and organism
  B on the SAME split with seed s+100 (different init, different data order,
  same class coverage). Concatenate their organs into one cascade. Every class
  is now covered by two independently grown organs, so roughly half the organs
  are genuinely dispensable -- and a correct certificate should say so.

Preregistered before running (same constants as exp_apoptosis v4):
  PASS  the certificate retires ~half the organs (>= 25% of them) with strict
        accuracy within 0.02 of the twin cascade -> "0 deaths" on the grown
        anatomy is a property of the ANATOMY, and session-15 stands.
  FAIL  it retires nothing (< 10%) -> the per-class gate is simply too strict
        to ever fire, and session-15's positive claim collapses into a
        statement about the gate. Reported either way.

Run:  python exp_redundant.py [--benchmarks digits kmnist] [--escrow 60]
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
from cgrow.organism import Organism
from exp_continual import run_ours, CFG as MLP_CFG
from exp_apoptosis import apoptose, COV_KEEP, ACC_SLACK, CLASS_SLACK

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MIN_DEATH_FRAC_PASS = 0.25     # preregistered
MAX_DEATH_FRAC_FAIL = 0.10     # preregistered


def twins(bench, seed, cfg):
    """Two organisms over the same classes, grown independently."""
    X, y, _ = load_real(bench)
    splits = prepare(X, y, seed=seed)
    tasks = split_tasks(y, 2)
    orgs = []
    for k, s in enumerate((seed, seed + 100)):
        torch.manual_seed(s)
        org, _ = run_ours(tasks, splits, X.shape[1], cfg, s)
        orgs.append(org)
    merged = Organism()
    merged.organs = list(orgs[0].organs) + list(orgs[1].organs)
    return orgs, merged, splits, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["digits", "kmnist"])
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--escrow", type=int, default=60,
                    help="REAL samples per class, gates only, never training")
    args = ap.parse_args()

    out = {}
    for bench in args.benchmarks:
        cfg = MLP_CFG[bench]
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            (A, B), merged, splits, tasks = twins(bench, seed, cfg)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            last = np.isin(yva, tasks[-1])

            s_A = A.scores(Xte, yte)["strict"]
            s_B = B.scores(Xte, yte)["strict"]
            s_M = merged.scores(Xte, yte)["strict"]

            rng_e = np.random.default_rng(seed * 71 + 3)
            Xe, ye = [], []
            for c in np.unique(ytr):
                m = np.flatnonzero(ytr == c)
                take = rng_e.choice(m, size=min(args.escrow, len(m)),
                                    replace=False)
                Xe.append(Xtr[take]); ye.append(ytr[take])
            escrow = (np.concatenate(Xe), np.concatenate(ye))

            n0 = len(merged.organs)
            after, genome, log, refusals = apoptose(
                merged, Xva[last], yva[last], Xte, yte, seed, escrow=escrow)
            s_after = after.scores(Xte, yte)["strict"]
            frac = len(log) / n0

            verdict = ("PASS" if frac >= MIN_DEATH_FRAC_PASS
                       and s_after >= s_M - ACC_SLACK
                       else "FAIL" if frac < MAX_DEATH_FRAC_FAIL else "PARTIAL")
            fails = {}
            for r in refusals:
                for f in r["fails"]:
                    fails[f] = fails.get(f, 0) + 1
            rec = {"seed": seed, "strict_A": s_A, "strict_B": s_B,
                   "strict_merged": s_M, "strict_after": s_after,
                   "organs_before": n0, "deaths": len(log),
                   "death_frac": frac, "verdict": verdict,
                   "params_before": merged.n_params() + sum(
                       l["params_freed"] for l in log),
                   "params_after": after.n_params(),
                   "total_flips": sum(l["real_test_flips"] for l in log),
                   "refusal_causes": fails,
                   "worst_class_drop_refused": max(
                       (r.get("worst_class_drop", 0.0) for r in refusals),
                       default=0.0),
                   "log": log}
            recs.append(rec)
            print(f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) twins "
                  f"{s_A:.3f}/{s_B:.3f} merged {s_M:.3f} | deaths "
                  f"{len(log)}/{n0} ({frac:.0%}) strict -> {s_after:.3f} "
                  f"flips {rec['total_flips']} | {verdict}", flush=True)
            if fails:
                print(f"    refusals {len(refusals)}: {fails} worst-class "
                      f"{rec['worst_class_drop_refused']:.3f}", flush=True)
            out[bench] = recs
            with open(os.path.join(RES, "redundant.json"), "w") as f:
                json.dump(out, f, indent=1,
                          default=lambda o: "<prox>" if hasattr(o, "score")
                          else str(o))
    print("wrote results/redundant.json")


if __name__ == "__main__":
    main()
