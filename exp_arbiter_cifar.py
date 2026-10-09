"""Does the territory-respecting arbiter hold on CIFAR-ViT?

The lifecycle's headline number (growth 0.667 -> adolescence 0.776 -> arbiter
0.835) is a FREE-arbiter number: exp_pfc.evidence() hands the arbiter each
organ's forced prediction and arbiter_predict() answers with the argmax organ
whether or not that organ accepted the input. On digits, 45% of the arbiter's
advantage came from answers outside every certified territory, and a
territory-respecting variant (choice restricted to accepting organs, abstain
when none accept) recovered 72% of the gain with 0% uncertified answers.

This runs exactly exp_pfc's phase-2/phase-3 protocol on the cached CIFAR-ViT
organisms -- same pool (3,500 contract samples drawn from the pre-adolescence
organs), same rng, same theta grid -- and adds:

  * the free vs territory-respecting comparison on the real test set
  * the exact additive channel decomposition (fill / reroute / drop)
  * whether each variant passes the SAME phase-4 acceptance gate, which decides
    if it could actually serve as the lifecycle's phase 3

Structural reason to expect a better result here than on twins: after
adolescence the organism carries a consolidated generalist C whose territory is
a union of the old PCA territories, so "restricted to organs that accept" is far
less restrictive than it is for a pool of narrow specialists.

Run:  python exp_arbiter_cifar.py [--first-seed 2] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import (consolidate_protos, sample_pool,
                               _strict_per_class)
from exp_cifar import load_feats, CFG
from exp_pfc import evidence, train_arbiter
from exp_arbiter_source import who_and_pred, decompose

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FEATS = os.path.join(RES, "cifar100_vitb.npz")


def gate_check(ans_ho, y_ho, org2, X_ho, classes, Xva_last, yva_last,
               net, thetas, gated):
    """exp_pfc's phase-4 acceptance gate, applied to whichever variant."""
    pc_new = _strict_per_class(ans_ho, ans_ho != ABSTAIN, y_ho,
                               np.asarray(classes))
    r_ref = org2.predict(X_ho)
    pc_ref = _strict_per_class(r_ref["pred"], r_ref["pred"] != ABSTAIN, y_ho,
                               np.asarray(classes))
    F_v, P_v = evidence(org2, Xva_last)
    a_v, _ = who_and_pred(net, F_v, P_v,
                          thetas["gated"] if gated else thetas["free"],
                          gated=gated)
    val_new = float((a_v == yva_last).mean())
    val_ref = org2.scores(Xva_last, yva_last)["strict"]
    ok = (pc_new.mean() >= pc_ref.mean() - 0.02
          and pc_new.min() >= pc_ref.min() - 0.05
          and val_new >= val_ref - 0.05)
    return {"accepted": bool(ok), "new_mean": float(pc_new.mean()),
            "ref_mean": float(pc_ref.mean()), "new_floor": float(pc_new.min()),
            "ref_floor": float(pc_ref.min()), "val_new": val_new,
            "val_ref": val_ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-name", default="arbiter_cifar.json")
    args = ap.parse_args()

    recs = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.time()
        splits = load_feats(seed, feats_path=FEATS)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
        with open(os.path.join(RES, f"adol_org_vitb_{seed}.pkl"), "rb") as fh:
            org = pickle.load(fh)
        s_grow = org.scores(Xte, yte)["strict"]

        # phase 2: adolescence, exactly as exp_pfc does it
        last = np.isin(yva, tasks[-1])
        org2, rep = consolidate_protos(org, Xtr.shape[1], CFG, seed,
                                       Xva_last=Xva[last], yva_last=yva[last])
        s_adol = org2.scores(Xte, yte)["strict"]

        # phase 3: the arbiter, on contract samples only (exp_pfc's protocol)
        rng = np.random.default_rng(seed * 41 + 9)
        X_syn, y_syn = sample_pool(org, 3500, rng)
        perm = rng.permutation(len(X_syn))
        ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
        F_tr, P_tr = evidence(org2, X_syn[tr])
        net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)

        # one theta per variant, both solved on the synthetic holdout
        F_ho, P_ho = evidence(org2, X_syn[ho])
        thetas = {}
        for gated in (False, True):
            best = (-1.0, None)
            for th in np.linspace(0.3, 0.9, 13):
                a, _ = who_and_pred(net, F_ho, P_ho, th, gated=gated)
                sc = float((a == y_syn[ho]).mean())
                if sc > best[0]:
                    best = (sc, float(th))
            thetas["gated" if gated else "free"] = best[1]

        d = decompose(org2, net, thetas, Xte, yte)
        classes = sorted(set(int(c) for c in y_syn))
        for gated in (False, True):
            a_ho, _ = who_and_pred(net, F_ho, P_ho,
                                   thetas["gated" if gated else "free"],
                                   gated=gated)
            g = gate_check(a_ho, y_syn[ho], org2, X_syn[ho], classes,
                           Xva[last], yva[last], net, thetas, gated)
            d["gate_gated" if gated else "gate_free"] = g

        rec = {"seed": seed, "growth": s_grow, "adolescence": s_adol,
               "adol_accepted": rep["accepted"], **d}
        recs.append(rec)
        ch = d["channels"]
        gt = d["gated"]
        print(f"[cifar_vitb seed {seed}] ({time.time()-t0:.0f}s) growth "
              f"{s_grow:.3f} -> adol {s_adol:.3f} (cascade "
              f"{d['cascade_strict']:.3f}) -> FREE arbiter "
              f"{d['arbiter_strict']:.3f} | GATED {gt['strict']:.3f} "
              f"(cov {gt['coverage']:.3f})", flush=True)
        kept = (gt["delta_vs_cascade"] / d["delta"] if d["delta"] else float("nan"))
        print(f"    free: fill {ch['fill']['contribution']:+.3f} reroute "
              f"{ch['reroute']['contribution']:+.3f} drop "
              f"{ch['drop']['contribution']:+.3f} | outside-territory "
              f"{d['picked_organ_did_not_accept']:.3f} worth "
              f"{d['gain_from_outside_territory']:+.3f}", flush=True)
        print(f"    gated keeps {kept:.0%} of the gain, outside-territory "
              f"{gt['answers_outside_territory']:.4f}, reroutes "
              f"{gt['reroute_wrong_to_right']}:{gt['reroute_right_to_wrong']}"
              f" | gate free={d['gate_free']['accepted']} "
              f"gated={d['gate_gated']['accepted']}", flush=True)
        with open(os.path.join(RES, args.out_name), "w") as f:
            json.dump(recs, f, indent=1, default=float)
    print(f"wrote results/{args.out_name}")


if __name__ == "__main__":
    main()
