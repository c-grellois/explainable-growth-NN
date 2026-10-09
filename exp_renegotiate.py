"""B-renegotiate -- can the CONTRACTS themselves change at adolescence?

A contract has two halves. The guarantee side (anchors, margins, thresholds)
already changes at adolescence: the consolidated organ's are brand new. This
experiment asks about the assumption/territory side, under two policies:

  union   inherit the old territories verbatim (they were calibrated on real
          data at each organ's birth)
  refit   renegotiate: fit fresh territories on the contract-sampled pool --
          a contract re-derived from its own generated data

and runs a REFINEMENT AUDIT for whichever organism results: sample fresh data
from the OLD contracts (the regions where promises exist) and measure, against
the pre-adolescence organism,
  coverage_kept   fraction of old-territory samples still answered
  agreement       answer agreement where both answer
  accuracy        label accuracy on the sampled region (both organisms)
A contract change is SAFE iff coverage is kept and agreement stays high --
refinement in the assume/guarantee sense, checked by sampling rather than
assumed. Anything beyond that is a versioned interface change, not an update.

Run:  python exp_renegotiate.py [--datasets digits mnist] [--seeds 2]
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
from cgrow.adolescence import (consolidate_mlp, consolidate_sheet,
                               metamorphosis_audit, sample_pool)
from exp_continual import run_ours, CFG
import exp_sheet as ES

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def refinement_audit(old: Organism, new: Organism, rng):
    """Sample the OLD contracts' territories; check the new organism against
    the promises made there."""
    X_old, y_old = sample_pool(old, 2000, rng)
    ro, rn = old.predict(X_old), new.predict(X_old)
    o_ans, n_ans = ro["pred"] != ABSTAIN, rn["pred"] != ABSTAIN
    both = o_ans & n_ans
    return {
        "old_coverage": float(o_ans.mean()),
        "new_coverage": float(n_ans.mean()),
        "coverage_kept": float(n_ans[o_ans].mean()) if o_ans.any() else 1.0,
        "agreement": float((ro["pred"][both] == rn["pred"][both]).mean())
            if both.any() else 1.0,
        "old_acc": float((ro["pred"][o_ans] == y_old[o_ans]).mean()) if o_ans.any() else 0.0,
        "new_acc": float((rn["pred"][n_ans] == y_old[n_ans]).mean()) if n_ans.any() else 0.0,
    }


def run_sheet_row(kind: str, seed: int):
    cfg = ES.CFG[kind]
    splits_all = ES.make_data(kind, seed)
    splits, in_dim = splits_all[:3], splits_all[3]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    torch.manual_seed(seed)
    org, _ = ES.run_ours(splits, in_dim, cfg, seed, listen=True)
    last = np.isin(yva, ES.TASKS[-1])
    row = {"seed": seed, "before": org.scores(Xte, yte)}
    for policy in ["union", "refit"]:
        new_org, rep = consolidate_sheet(org, in_dim, ES.T, cfg, seed,
                                         Xva_last=Xva[last], yva_last=yva[last],
                                         prox_policy=policy)
        cell = {"accepted": rep["accepted"]}
        if rep["accepted"]:
            cell["after"] = new_org.scores(Xte, yte)
            cell["audit"] = metamorphosis_audit(org, new_org, Xte, yte)
            cell["refinement"] = refinement_audit(
                org, new_org, np.random.default_rng(seed * 7 + 3))
        else:
            cell["why"] = rep.get("why")
        row[policy] = cell
    return row


def show_row(row, tag):
    b = row["before"]["strict"]
    print(f"[{tag} seed {row['seed']}] before strict {b:.3f}")
    for policy in ["union", "refit"]:
        c = row[policy]
        if not c["accepted"]:
            print(f"  {policy:6s}: REJECTED ({c.get('why')})")
            continue
        a, r = c["after"], c["refinement"]
        print(f"  {policy:6s}: strict {a['strict']:.3f}  cov {a['coverage']:.3f}"
              f"  share_C {c['audit']['share_answered_by_C']:.2f}"
              f"  | refinement: cov_kept {r['coverage_kept']:.3f}"
              f"  agreement {r['agreement']:.3f}"
              f"  acc {r['old_acc']:.3f}->{r['new_acc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["digits", "mnist", "kmnist", "fashion"])
    ap.add_argument("--sheets", nargs="+", default=["sweeps", "shapes"])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    out = {}
    for ds in args.datasets:
        cfg = CFG[ds]
        X, y, C = load_real(ds)
        recs = []
        for seed in range(args.seeds):
            t0 = time.time()
            splits = prepare(X, y, seed=seed)
            tasks = split_tasks(y, 2)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            torch.manual_seed(seed)
            org, _ = run_ours(tasks, splits, X.shape[1], cfg, seed)
            last = np.isin(yva, tasks[-1])
            row = {"seed": seed, "before": org.scores(Xte, yte)}
            for policy in ["union", "refit"]:
                new_org, rep = consolidate_mlp(
                    org, X.shape[1], cfg, seed, Xva_last=Xva[last],
                    yva_last=yva[last], prox_policy=policy)
                cell = {"accepted": rep["accepted"]}
                if rep["accepted"]:
                    cell["after"] = new_org.scores(Xte, yte)
                    cell["audit"] = metamorphosis_audit(org, new_org, Xte, yte)
                    cell["refinement"] = refinement_audit(
                        org, new_org, np.random.default_rng(seed * 7 + 3))
                else:
                    cell["why"] = rep.get("why")
                row[policy] = cell
            recs.append(row)
            show_row(row, ds)
        out[ds] = recs

    for kind in args.sheets:
        recs = []
        for seed in range(args.seeds):
            row = run_sheet_row(kind, seed)
            recs.append(row)
            show_row(row, f"sheet-{kind}")
        out[f"sheet_{kind}"] = recs

    p = os.path.join(RES, "renegotiate.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
