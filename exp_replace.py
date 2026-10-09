"""B-replace -- the replaceable organ: lesion one organ of a grown organism and
re-derive it from its published contract alone, with NO access to training data.

An organ's spec is finite and checkable: input centroids + rho_in, feature anchors
+ rho + tau, class ids, Lipschitz cap. The spec is also a data generator (the
contracts note's §3.7.2 move): sample x in the accepted input region (gaussian
around the centroids, clipped to rho_in), label it with its centroid's class, and
train a fresh trunk -- different width, different seed -- against only that.

Measured: agreement of the transplanted organism with the original on the test
set, split into (a) inputs the lesioned organ answered, (b) everything else
(which must agree EXACTLY -- other organs are untouched; routing can only be
disturbed on the lesioned organ's accepted set).

Control that must fail: swap the class labels in the generated data -- agreement
on the organ's region must invert toward zero, not fall to chance.

Run:  python exp_replace.py [--dataset digits] [--seeds N]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import load_real, prepare, split_tasks
from cgrow.organ import Organ, DTYPE
from cgrow.organism import Organism, ABSTAIN

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)


def spec_of(o: Organ) -> dict:
    """The published contract: everything a replacement implementer receives.
    The proximity object IS the spec's data generator (means, axes, eigenvalues,
    calibration scalars -- no raw samples)."""
    return {"prox": o.prox, "rho_in": o.rho_in, "class_ids": o.class_ids.copy(),
            "rho": o.rho, "tau": o.tau, "cap": o.trunk.cap}


def rederive(spec: dict, in_dim: int, feat: int, hidden, seed: int,
             n_samples: int = 4000, swap: bool = False, epochs: int = 60):
    """Train a fresh organ from the spec alone: sample labeled data from the
    published proximity contract, train a new trunk against it, re-solve the
    feature thresholds on more sampled data, and inherit the proximity contract
    verbatim. No real data is ever touched."""
    rng = np.random.default_rng(seed)
    X, yl = spec["prox"].sample(n_samples, rng)
    if swap:                              # invert the spec's meaning
        cls = list(spec["class_ids"])
        flip = {c: cls[len(cls) - 1 - i] for i, c in enumerate(cls)}
        yl = np.array([flip[c] for c in yl])
    new = Organ(in_dim, feat, hidden, cap=spec["cap"],
                class_ids=list(spec["class_ids"]), seed=seed + 500)
    new.train_proto(X.astype(np.float64), yl, epochs=epochs)
    new.solve_thresholds(X, yl, p_target=0.95, require=False)
    new.freeze(X, yl)
    new.prox = spec["prox"]               # the proximity contract, verbatim
    new.rho_in = spec["rho_in"]
    return new


def grow_for_dataset(ds: str, seed: int):
    """A small class-incremental organism (one organ per 2-class task)."""
    X, y, C = load_real(ds)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare(X, y, seed=seed)
    tasks = split_tasks(y, 2)
    org = Organism()
    for t, cls in enumerate(tasks):
        trm, vam = np.isin(ytr, cls), np.isin(yva, cls)
        cand = Organ(X.shape[1], 16, (64,), cap=2.0, class_ids=cls,
                     seed=seed * 100 + t)
        cand.train_proto(Xtr[trm], ytr[trm], epochs=50)
        org.try_grow(cand, Xva[vam], yva[vam], p_target=0.95, gate=False,
                     label=f"t{t}")
    return org, (Xte, yte), X.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--organ", type=int, default=1, help="which organ to lesion")
    args = ap.parse_args()

    records = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        org, (Xte, yte), in_dim = grow_for_dataset(args.dataset, seed)
        base = org.predict(Xte)
        lesion = args.organ
        region = base["organ"] == lesion          # inputs the organ answered
        spec = spec_of(org.organs[lesion])

        row = {"seed": seed, "region_size": int(region.sum())}
        for arch_name, hidden, feat in [("same", (64,), 16), ("wider", (128,), 32),
                                        ("deeper", (64, 32), 16)]:
            new = rederive(spec, in_dim, feat, hidden, seed=seed)
            org2 = copy.deepcopy(org)
            org2.organs[lesion] = new
            r2 = org2.predict(Xte)
            agree_region = float((r2["pred"][region] == base["pred"][region]).mean()) \
                if region.any() else 1.0
            agree_else = float((r2["pred"][~region] == base["pred"][~region]).mean())
            acc_region = float((r2["pred"][region] == yte[region]).mean()) if region.any() else 1.0
            row[arch_name] = {"agree_region": agree_region, "agree_else": agree_else,
                              "acc_region": acc_region}
        # control: swapped spec must invert
        new_sw = rederive(spec, in_dim, 16, (64,), seed=seed, swap=True)
        org3 = copy.deepcopy(org)
        org3.organs[lesion] = new_sw
        r3 = org3.predict(Xte)
        ans = (r3["pred"][region] != ABSTAIN)
        agree_sw = float((r3["pred"][region][ans] == base["pred"][region][ans]).mean()) \
            if ans.any() else float("nan")
        row["swapped_agree_region"] = agree_sw
        records.append(row)
        print(f"[{args.dataset} seed {seed}] organ {lesion} region={row['region_size']}")
        for a in ["same", "wider", "deeper"]:
            r = row[a]
            print(f"  {a:6s}: agree(region) {r['agree_region']:.3f}  "
                  f"agree(elsewhere, must be 1.0) {r['agree_else']:.3f}  "
                  f"acc(region) {r['acc_region']:.3f}")
        print(f"  swapped control: agreement on answered region {agree_sw:.3f} (must collapse)")

    out = os.path.join(RES, f"replace_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump(records, f, indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
