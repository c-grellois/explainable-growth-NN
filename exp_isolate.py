"""B-isolate -- which features carry the approach, which impede it.

Already measured and REMOVED in earlier sessions (evidence in claims.json):
scalar tightening once vetoes exist (-8 pts when added back), trunk-mode
consolidation at high class counts (collapses), mid-life consolidation timing
(inferior to end-of-life), territory renegotiation from generated data
(vacuous), decentralized price routing (anti-calibrated).

This battery isolates what was never separated, through the FULL final
pipeline (growth -> adolescence(protos, gated) -> arbiter(gated)) on the two
flagship regimes, eval seeds 2-4:

  no_veto   are vetoes still needed once the arbiter exists, or does central
            routing subsume learned inhibition?
  no_cap    does the spectral cap on organ trunks help or impede?
  no_adol   does adolescence add anything beyond what the arbiter recovers?

Removal rule (preregistered): a feature leaves the final configuration iff
removing it costs no more than 0.005 mean strict on BOTH benchmarks; it is
confirmed load-bearing iff removal costs more than 0.01 on either. The 'full'
reference is the existing lifecycle.json (identical pipeline and seeds).

Run:  python exp_isolate.py [--cells no_veto no_cap no_adol]
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
from cgrow.adolescence import consolidate_protos, sample_pool, _strict_per_class
from exp_continual import run_ours, CFG as MLP_CFG
from exp_cifar import load_feats, CFG as CIFAR_CFG
from exp_pfc import evidence, train_arbiter, arbiter_predict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CELLS = {
    "no_veto": {"veto": False},
    "no_cap": {"cap": 100.0},
    "no_adol": {"skip_adol": True},
}
BENCH = ["mnist", "cifar_vitb"]


def pipeline(bench: str, seed: int, over: dict):
    skip_adol = over.pop("skip_adol", False)
    if bench == "cifar_vitb":
        cfg = {**CIFAR_CFG, **over}
        splits = load_feats(seed, feats_path=os.path.join(
            RES, "cifar100_vitb.npz"))
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
    else:
        cfg = {**MLP_CFG[bench], **over}
        X, y, _ = load_real(bench)
        splits = prepare(X, y, seed=seed)
        tasks = split_tasks(y, 2)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    torch.manual_seed(seed)
    org, _ = run_ours(tasks, splits, Xtr.shape[1], cfg, seed)
    growth = org.scores(Xte, yte)["strict"]

    last = np.isin(yva, tasks[-1])
    if skip_adol:
        org2, adol_ok = org, False
    else:
        org2, rep = consolidate_protos(org, Xtr.shape[1], cfg, seed,
                                       Xva_last=Xva[last], yva_last=yva[last])
        adol_ok = rep["accepted"]
    adol = org2.scores(Xte, yte)["strict"]

    rng = np.random.default_rng(seed * 41 + 9)
    X_syn, y_syn = sample_pool(org, 2500, rng)
    perm = rng.permutation(len(X_syn))
    ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
    F_tr, P_tr = evidence(org2, X_syn[tr])
    net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)
    F_ho, P_ho = evidence(org2, X_syn[ho])
    best = (0.0, None)
    for th in np.linspace(0.3, 0.9, 13):
        s = float((arbiter_predict(net, F_ho, P_ho, th) == y_syn[ho]).mean())
        if s > best[0]:
            best = (s, float(th))
    a_ho = arbiter_predict(net, F_ho, P_ho, best[1])
    classes = np.asarray(sorted(set(int(c) for c in y_syn)))
    pc_new = _strict_per_class(a_ho, a_ho != ABSTAIN, y_syn[ho], classes)
    r_ref = org2.predict(X_syn[ho])
    pc_ref = _strict_per_class(r_ref["pred"], r_ref["pred"] != ABSTAIN,
                               y_syn[ho], classes)
    F_v, P_v = evidence(org2, Xva[last])
    val_new = float((arbiter_predict(net, F_v, P_v, best[1]) == yva[last]).mean())
    val_ref = org2.scores(Xva[last], yva[last])["strict"]
    accepted = (pc_new.mean() >= pc_ref.mean() + 0.01
                and pc_new.min() >= pc_ref.min() - 0.05
                and val_new >= val_ref)
    if accepted:
        F_te, P_te = evidence(org2, Xte)
        final = float((arbiter_predict(net, F_te, P_te, best[1]) == yte).mean())
    else:
        final = adol
    return {"growth": growth, "adol": adol, "adol_ok": adol_ok,
            "pfc_ok": accepted, "final": final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    path = os.path.join(RES, "isolate.json")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            out = json.load(f)
    for cell in args.cells:
        for bench in BENCH:
            key = f"{bench}:{cell}"
            recs = out.get(key, [])
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                if any(r["seed"] == seed for r in recs):
                    continue
                t0 = time.time()
                r = pipeline(bench, seed, dict(CELLS[cell]))
                r["seed"] = seed
                recs.append(r)
                print(f"[{key} seed {seed}] ({time.time()-t0:.0f}s) growth "
                      f"{r['growth']:.3f} -> adol "
                      f"{'ok' if r['adol_ok'] else 'skip/rej'} {r['adol']:.3f}"
                      f" -> final {r['final']:.3f} "
                      f"(pfc {'ok' if r['pfc_ok'] else 'rej'})", flush=True)
                out[key] = recs
                with open(path, "w") as f:
                    json.dump(out, f, indent=1, default=float)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
