"""B-midlife -- does adolescence compose with further development?

CIFAR-100 on frozen ViT features, 10 tasks. Grow tasks 1-5 as usual, run the
verified metamorphosis (prototype mode, relative gate), then CONTINUE growing
tasks 6-10 on the metamorphosed organism -- new organs behind the consolidated
one, tightening and vetoes as usual, freezing resumed. Compared against the two
known endpoints: no adolescence at all (0.640/0.680) and adolescence at the end
(0.765/0.785).

Also measured: growth flips in phase 2 (must be exactly 0 -- the invariant must
survive the metamorphosis), and the consolidated organ's untightenable false
accepts on later classes (its union territory cannot take vetoes -- a known
structural gap, priced here).

Run:  python exp_midlife.py [--seeds 2]
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
from cgrow.organ import Organ
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import consolidate_protos
from exp_cifar import load_feats, CFG

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FEATS = os.path.join(RES, "cifar100_vitb.npz")


def grow_task(org, cls, splits, cfg, seed, t):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    trm, vam = np.isin(ytr, cls), np.isin(yva, cls)
    Xt_tr, yt_tr = Xtr[trm], ytr[trm]
    fa = None
    if org.organs:
        fa_before = float((org.predict(Xt_tr)["pred"] != ABSTAIN).mean())
        org.tighten_against(Xt_tr, self_budget=0.0,
                            veto_theta_q=cfg["veto_theta_q"])
        fa = (fa_before,
              float((org.predict(Xt_tr)["pred"] != ABSTAIN).mean()))
    p_before = org.predict(Xte)["pred"] if org.organs else None
    need = org.declined_mask(Xt_tr) if org.organs else np.ones(len(Xt_tr), bool)
    cand = Organ(Xtr.shape[1], cfg["feat"], cfg["hidden"], cap=cfg["cap"],
                 class_ids=cls, seed=seed * 100 + t,
                 q_cov_in=cfg["q_cov_in"], prox=cfg["prox"])
    cand.train_proto(Xt_tr[need], yt_tr[need], epochs=cfg["epochs"])
    grew = org.try_grow(cand, Xva[vam], yva[vam], p_target=cfg["p_target"],
                        gate=(t > 0), label=f"task{t}")
    if grew and t > 0 and cfg.get("prox") in ("pca", "mix"):
        rng = np.random.default_rng(seed * 10 + t)
        Xneg = np.concatenate([o.prox.sample(2000, rng)[0]
                               for o in org.organs[:-1]])
        org.organs[-1].add_veto(Xneg, theta_q=cfg["veto_theta_q"],
                                seed=seed * 100 + 50 + t,
                                X_self=Xva[vam], max_self_loss=0.03)
    flips = 0
    if p_before is not None:
        p_after = org.predict(Xte)["pred"]
        was = p_before != ABSTAIN
        flips = int((was & (p_after != p_before)).sum())
    return flips, fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    records = []
    for seed in range(args.seeds):
        t0 = time.time()
        splits = load_feats(seed, feats_path=FEATS)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
        torch.manual_seed(seed)
        org = Organism()
        flips1 = 0
        for t in range(5):
            f, _ = grow_task(org, tasks[t], splits, CFG, seed, t)
            flips1 += f

        mid_mask = np.isin(yte, [c for cl in tasks[:5] for c in cl])
        mid_before = org.scores(Xte[mid_mask], yte[mid_mask])
        last = np.isin(yva, tasks[4])
        org2, rep = consolidate_protos(org, Xtr.shape[1], CFG, seed,
                                       Xva_last=Xva[last], yva_last=yva[last])
        mid_after = org2.scores(Xte[mid_mask], yte[mid_mask])

        flips2 = 0
        fa_log = []
        for t in range(5, 10):
            f, fa = grow_task(org2, tasks[t], splits, CFG, seed, t)
            flips2 += f
            fa_log.append(fa)
        final = org2.scores(Xte, yte)
        r = org2.predict(Xte)
        c_wrong_late = float(((r["organ"] == 0)
                              & np.isin(yte, [c for cl in tasks[5:] for c in cl])
                              ).mean()) if rep["accepted"] else 0.0
        rec = {"seed": seed, "adol_accepted": rep["accepted"],
               "mid_before": mid_before, "mid_after": mid_after,
               "final": final, "flips_phase1": flips1,
               "flips_phase2_post_metamorphosis": flips2,
               "share_C_on_late_classes": c_wrong_late,
               "fa_late": fa_log}
        records.append(rec)
        print(f"[midlife seed {seed}] ({time.time()-t0:.0f}s) adol "
              f"{'ACCEPTED' if rep['accepted'] else 'REJECTED: ' + str(rep.get('why'))}")
        print(f"  mid (tasks 1-5): {mid_before['strict']:.3f} -> {mid_after['strict']:.3f}")
        print(f"  final (all 10) : strict {final['strict']:.3f}  cov {final['coverage']:.3f}"
              f"  prec {final['precision']:.3f}")
        print(f"  growth flips: phase1 {flips1}  phase2 (post-metamorphosis, must be 0) {flips2}")
        print(f"  C answering late-class samples (frozen error channel): {c_wrong_late:.3f}")

    p = os.path.join(RES, "midlife.json")
    with open(p, "w") as f:
        json.dump(records, f, indent=1, default=float)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
