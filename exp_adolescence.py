"""B-adolescence -- the verified metamorphosis, benchmarked before/after.

Protocol (MLP ladder): grow the organism through all tasks exactly as in B2,
then run adolescence: sample labeled data from every organ's contract, train
one consolidated organ jointly on the pooled synthetic data (NO raw data),
verify on a synthetic holdout + the last task's real val, and swap it in ahead
of the old cascade (which stays as fallback, so coverage cannot drop).

Measured on real test data:
  strict/forced/coverage/precision before vs after; the metamorphosis audit
  (answer flips and their direction -- adolescence is the one sanctioned
  exception to the retention invariant, so every change is counted); the
  parameter ledger (before / after with fallback / consolidated organ alone,
  plus C-alone accuracy -- what full synaptic retirement would cost).

Protocol (sheet): same, with the consolidated organ's units re-placed by the
wiring economy and local connections re-learned from scratch under the hinge
cost -- the massive recombination of local wiring -- plus the wiring ledger
(live edges, weighted mean edge length) before vs after.

Run:  python exp_adolescence.py [--datasets digits mnist ...] [--sheets sweeps shapes] [--seeds 2]
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
from cgrow.adolescence import (consolidate_mlp, consolidate_sheet,
                               metamorphosis_audit)
from exp_continual import run_ours, CFG
import exp_sheet as ES

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)


def c_alone(new_org: Organism, Xte, yte):
    solo = Organism()
    solo.organs = [new_org.organs[0]]
    return solo.scores(Xte, yte)


def run_mlp(ds: str, seed: int):
    cfg = CFG[ds]
    X, y, C = load_real(ds)
    splits = prepare(X, y, seed=seed)
    tasks = split_tasks(y, 2)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    torch.manual_seed(seed)
    org, ours = run_ours(tasks, splits, X.shape[1], cfg, seed)

    last_mask = np.isin(yva, tasks[-1])
    new_org, rep = consolidate_mlp(org, X.shape[1], cfg, seed,
                                   Xva_last=Xva[last_mask],
                                   yva_last=yva[last_mask])
    before = org.scores(Xte, yte)
    rec = {"seed": seed, "before": before, "adolescence": rep}
    if rep["accepted"]:
        after = new_org.scores(Xte, yte)
        audit = metamorphosis_audit(org, new_org, Xte, yte)
        solo = c_alone(new_org, Xte, yte)
        rec.update({"after": after, "audit": audit, "c_alone": solo,
                    "params": {"before": org.n_params(),
                               "after_with_fallback": new_org.n_params(),
                               "c_alone": new_org.organs[0].n_params()}})
    return rec


def sheet_wiring(org: Organism):
    import torch as t
    w_all, d_all = [], []
    for o in org.organs:
        w, d = o.edges()
        w_all.append(w); d_all.append(d)
    w, d = t.cat(w_all), t.cat(d_all)
    return {"live_edges": int(sum(int(o.edge_mask_self.sum())
                                  + (int(o.edge_mask_prev.sum())
                                     if o.W_prev is not None else 0)
                                  for o in org.organs)),
            "mean_edge_len": float((w * d).sum() / w.sum())}


def run_sheet(kind: str, seed: int):
    cfg = ES.CFG[kind]
    splits_all = ES.make_data(kind, seed)
    splits, in_dim = splits_all[:3], splits_all[3]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    torch.manual_seed(seed)
    org, ours = ES.run_ours(splits, in_dim, cfg, seed, listen=True)

    last_mask = np.isin(yva, ES.TASKS[-1])
    new_org, rep = consolidate_sheet(org, in_dim, ES.T, cfg, seed,
                                     Xva_last=Xva[last_mask],
                                     yva_last=yva[last_mask])
    before = org.scores(Xte, yte)
    rec = {"seed": seed, "before": before, "adolescence": rep,
           "wiring_before": sheet_wiring(org)}
    if rep["accepted"]:
        after = new_org.scores(Xte, yte)
        audit = metamorphosis_audit(org, new_org, Xte, yte)
        solo_org = Organism(); solo_org.organs = [new_org.organs[0]]
        rec.update({"after": after, "audit": audit,
                    "c_alone": solo_org.scores(Xte, yte),
                    "wiring_c": sheet_wiring(solo_org),
                    "params": {"before": org.n_params(),
                               "c_alone": new_org.organs[0].n_params()}})
    return rec


def show(rec, tag):
    b = rec["before"]
    line = (f"[{tag} seed {rec['seed']}] before: strict {b['strict']:.3f} "
            f"cov {b['coverage']:.3f} prec {b['precision']:.3f}")
    if not rec["adolescence"]["accepted"]:
        print(line + f"  |  adolescence REJECTED ({rec['adolescence'].get('why')})")
        return
    a, au = rec["after"], rec["audit"]
    print(line)
    print(f"    after : strict {a['strict']:.3f} cov {a['coverage']:.3f} "
          f"prec {a['precision']:.3f}  |  C answers {au['share_answered_by_C']:.2f} "
          f"of test  flips {au['flips']} ({au['flips_to_right']} right, "
          f"{au['flips_to_wrong']} wrong)  +ans {au['gained_answers']} -ans {au['lost_answers']}")
    ca = rec["c_alone"]
    print(f"    C alone: strict {ca['strict']:.3f}  params {rec['params'].get('c_alone')}"
          f" (before {rec['params']['before']})")
    if "wiring_before" in rec and "wiring_c" in rec:
        wb, wc = rec["wiring_before"], rec["wiring_c"]
        print(f"    wiring : edges {wb['live_edges']} -> {wc['live_edges']}   "
              f"mean len {wb['mean_edge_len']:.3f} -> {wc['mean_edge_len']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["digits", "mnist"])
    ap.add_argument("--sheets", nargs="+", default=["sweeps", "shapes"])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    out = {"mlp": {}, "sheet": {}}
    for ds in args.datasets:
        recs = []
        for seed in range(args.seeds):
            t0 = time.time()
            rec = run_mlp(ds, seed)
            recs.append(rec)
            show(rec, f"{ds}")
            print(f"    ({time.time()-t0:.0f}s)")
        out["mlp"][ds] = recs
    for kind in args.sheets:
        recs = []
        for seed in range(args.seeds):
            t0 = time.time()
            rec = run_sheet(kind, seed)
            recs.append(rec)
            show(rec, f"sheet-{kind}")
            print(f"    ({time.time()-t0:.0f}s)")
        out["sheet"][kind] = recs

    p = os.path.join(RES, "adolescence.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
