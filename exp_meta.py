"""B-meta -- second-order contracts: an SLA on the organism's own drift.

The intermediate guarantee (audited prototypes) logs drift after the fact and
forecasts it from contract samples (measured: correlation 0.65-0.98, real
1.5-2x the forecast on the promised region). A SECOND-ORDER contract turns the
forecast into a published, checkable bound: before each prototype-entry event
the system announces

    "on the promised region, at most Delta_t of answers will change",
    Delta_t = forecast_t * factor_t

with factor_t calibrated ONLINE from the system's own past events: the
conformal quantile of past (real/forecast) ratios (alpha = 0.25), floored at a
preregistered prior of 2.0 (the dev-era measurement) while history is short.

Measured on real test flips, per event: bound VIOLATIONS (want ~<= alpha) and
TIGHTNESS (bound/real; huge = vacuous promise). Baseline: the fixed prior 2.0
alone. If the online SLA holds with non-vacuous margins, the organism can sell
promises about its own future change -- governance all the way up.

Run:  python exp_meta.py [--tags r18 vitb] [--first-seed 2 --seeds 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.organ import conformal_quantile
from exp_cifar import load_feats, FEATS
from exp_drift import AuditedPrototypes

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PRIOR, ALPHA = 2.0, 0.25          # preregistered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["r18", "vitb"])
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    out = {}
    for tag in args.tags:
        feats = os.path.join(RES, f"cifar100_{tag}.npz")
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            splits = load_feats(seed, feats_path=feats)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            order = np.random.default_rng(seed + 400).permutation(100)
            tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                     for i in range(10)]
            rng = np.random.default_rng(seed * 13 + 1)
            model = AuditedPrototypes(seed)
            events, ratios = [], []
            pred_te = None
            for t, cls in enumerate(tasks):
                m = np.isin(ytr, cls)
                if t > 0:
                    X_old = model.sample_old(rng)
                    before_syn = model.predict(X_old)
                model.add_task(Xtr[m], ytr[m], cls)
                new_pred = model.predict(Xte)
                if t > 0:
                    fc = float((before_syn != model.predict(X_old)).mean())
                    seen = np.isin(yte, [c for cl in tasks[:t] for c in cl])
                    real = float((pred_te != new_pred)[seen].mean())
                    # the published bound, from history available BEFORE t
                    factor = (max(PRIOR, conformal_quantile(
                        np.array(ratios), ALPHA)) if len(ratios) >= 2
                        else PRIOR)
                    bound = factor * max(fc, 1e-4)
                    events.append({"t": t, "forecast": fc, "real": real,
                                   "factor": factor, "bound": bound,
                                   "violated": bool(real > bound),
                                   "prior_violated": bool(real > PRIOR *
                                                          max(fc, 1e-4))})
                    ratios.append(real / max(fc, 1e-4))
                pred_te = new_pred
            v = sum(e["violated"] for e in events)
            pv = sum(e["prior_violated"] for e in events)
            tight = float(np.median([e["bound"] / max(e["real"], 1e-4)
                                     for e in events]))
            recs.append({"seed": seed, "events": events,
                         "violations": v, "prior_violations": pv,
                         "median_tightness": tight})
            print(f"[meta {tag} seed {seed}] events {len(events)}  "
                  f"SLA violations {v} (prior-only {pv})  "
                  f"median bound/real {tight:.2f}", flush=True)
        out[tag] = recs

    with open(os.path.join(RES, "meta_sla.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote results/meta_sla.json")


if __name__ == "__main__":
    main()
