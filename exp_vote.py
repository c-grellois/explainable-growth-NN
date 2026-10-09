"""Certified voting: the one free ensembling effect, put to work.

B-ensemble found exactly one genuine ensembling gain over first-accept that is
contract-covered by construction: majority vote among the organs that ACCEPT
the input (+0.010 on the twin pool). It needs no arbiter, no trained routing
state, no real data -- only the contracts' own accept bits -- so it is a
routing upgrade the certified organism gets for free. Nobody has measured it
on the real lifecycle organisms.

For every CPU benchmark (7 MLP + 3 sheet) on evaluation seeds 2-4, build the
certified organism exactly as exp_lifecycle does (growth, then adolescence if
its gate accepts) and compare two contract-covered routers on the same test
set:

  first-accept   the cascade (birth order decides among accepting organs)
  vote           majority among accepting organs; ties -> earliest (cascade)

Also reports the contest rate (inputs where >= 2 organs accept -- the only
place the two can differ) and the direction of every changed answer.

Run:  python exp_vote.py [--benchmarks digits ...] [--first-seed 2 --seeds 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import consolidate_protos
from exp_lifecycle import build, MLP_SETS, SHEET_SETS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def vote_accepting(org: Organism, X: np.ndarray):
    """Majority vote among accepting organs; ties broken by cascade order."""
    A, P = [], []
    for o in org.organs:
        d1, m, p, d_in, vet, _ = o.evaluate(X)
        A.append(o.accept(d1, m, d_in, vet))
        P.append(p)
    A, P = np.stack(A, axis=1), np.stack(P)          # (n,K), (K,n)
    n, K = A.shape
    out = np.full(n, ABSTAIN, dtype=np.int64)
    n_acc = A.sum(axis=1)
    for i in np.flatnonzero(n_acc > 0):
        ks = np.flatnonzero(A[i])
        labels = P[ks, i]
        vals, counts = np.unique(labels, return_counts=True)
        cands = set(vals[counts == counts.max()].tolist())
        for k in ks:                                  # earliest accepter wins
            if P[k, i] in cands:
                out[i] = P[k, i]
                break
    return out, n_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=MLP_SETS + SHEET_SETS)
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-name", default="vote.json")
    args = ap.parse_args()

    out = {}
    for bench in args.benchmarks:
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            org, splits, tasks, cfg, Xva_last, yva_last = build(bench, seed)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            org2, rep = consolidate_protos(org, Xtr.shape[1], cfg, seed,
                                           Xva_last=Xva_last,
                                           yva_last=yva_last)
            rec = {"seed": seed, "adol_accepted": rep["accepted"]}
            for tag, o in (("raw", org), ("certified", org2)):
                fa = o.predict(Xte)["pred"]
                vt, n_acc = vote_accepting(o, Xte)
                ch = (fa != vt)
                rec[tag] = {
                    "first_accept": float((fa == yte).mean()),
                    "vote": float((vt == yte).mean()),
                    "coverage": float((fa != ABSTAIN).mean()),
                    "contest_rate": float((n_acc >= 2).mean()),
                    "changed": int(ch.sum()),
                    "changed_to_right": int((ch & (vt == yte)).sum()),
                    "changed_to_wrong": int((ch & (fa == yte)).sum()),
                    "organs": len(o.organs)}
            recs.append(rec)
            c = rec["certified"]
            print(f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) certified: "
                  f"first-accept {c['first_accept']:.4f} -> vote "
                  f"{c['vote']:.4f} ({c['vote']-c['first_accept']:+.4f}) | "
                  f"contest {c['contest_rate']:.3f} changed {c['changed']} "
                  f"({c['changed_to_right']} right / {c['changed_to_wrong']} "
                  f"wrong) | raw {rec['raw']['first_accept']:.4f} -> "
                  f"{rec['raw']['vote']:.4f}", flush=True)
            out[bench] = recs
            with open(os.path.join(RES, args.out_name), "w") as f:
                json.dump(out, f, indent=1, default=float)
    print(f"wrote results/{args.out_name}")


if __name__ == "__main__":
    main()
