"""Per-organ refinement audit of adolescence on the spatial sheet.

The aggregate audit says the metamorphosis keeps the old contracts' promises;
this asks whether it keeps them UNIFORMLY: for each old organ, sample its own
territory (its contract as generator) and measure, before vs after adolescence,
  coverage_kept  fraction of that territory still answered
  agreement      answer agreement with the pre-adolescence organism
  acc            label accuracy on the territory, before -> after
under both contract policies (union = inherited territories, refit =
renegotiated from generated data).

Run:  python exp_sheet_audit.py [--streams sweeps shapes] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import consolidate_sheet
import exp_sheet as ES

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def per_organ_audit(old: Organism, new: Organism, rng):
    rows = []
    for k, o in enumerate(old.organs):
        X, y = o.prox.sample(1500, rng)
        ro, rn = old.predict(X), new.predict(X)
        o_ans, n_ans = ro["pred"] != ABSTAIN, rn["pred"] != ABSTAIN
        both = o_ans & n_ans
        rows.append({
            "organ": k, "classes": [int(c) for c in o.class_ids],
            "coverage_kept": float(n_ans[o_ans].mean()) if o_ans.any() else 1.0,
            "agreement": float((ro["pred"][both] == rn["pred"][both]).mean())
                if both.any() else 1.0,
            "acc_before": float((ro["pred"][o_ans] == y[o_ans]).mean())
                if o_ans.any() else 0.0,
            "acc_after": float((rn["pred"][n_ans] == y[n_ans]).mean())
                if n_ans.any() else 0.0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=["sweeps", "shapes"])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    out = {}
    for kind in args.streams:
        cfg = ES.CFG[kind]
        recs = []
        for seed in range(args.seeds):
            splits_all = ES.make_data(kind, seed)
            splits, in_dim = splits_all[:3], splits_all[3]
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            torch.manual_seed(seed)
            org, _ = ES.run_ours(splits, in_dim, cfg, seed, listen=True)
            last = np.isin(yva, ES.TASKS[-1])
            row = {"seed": seed}
            for policy in ["union", "refit"]:
                new_org, rep = consolidate_sheet(
                    org, in_dim, ES.T, cfg, seed, Xva_last=Xva[last],
                    yva_last=yva[last], prox_policy=policy)
                if rep["accepted"]:
                    row[policy] = per_organ_audit(
                        org, new_org, np.random.default_rng(seed * 11 + 2))
                else:
                    row[policy] = {"rejected": rep.get("why")}
            recs.append(row)
            print(f"[sheet-{kind} seed {seed}]")
            for policy in ["union", "refit"]:
                cells = row[policy]
                if isinstance(cells, dict):
                    print(f"  {policy}: REJECTED")
                    continue
                for c in cells:
                    print(f"  {policy} organ {c['organ']} (classes {c['classes']}): "
                          f"cov_kept {c['coverage_kept']:.3f}  "
                          f"agree {c['agreement']:.3f}  "
                          f"acc {c['acc_before']:.3f}->{c['acc_after']:.3f}")
        out[kind] = recs

    p = os.path.join(RES, "sheet_audit.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
