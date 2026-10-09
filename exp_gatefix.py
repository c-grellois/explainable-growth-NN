"""Does the repaired gate catch what the old one missed?

Re-runs the three lifecycle changes that the phase-4 gate accepted, on the
CIFAR-ViT organisms, under the OLD gate (per-class on contract samples plus
last-task real validation) and the REPAIRED gate (adds a data-free territory
check and per-class floors on a small REAL escrow):

  free arbiter    accepted by the old gate 3/3; buys 94% of its gain by
                  answering outside every certified territory
  gated arbiter   territory-respecting by construction; should stay accepted
  negotiate/mkt   the market arm, a true null (-0.0001)
  negotiate/orcl  the oracle arm, quietly -0.0045 on real test, accepted 3/3

Escrow: 60 real samples per class drawn from the HELD-OUT VALIDATION split
(gates only, never training). Overall verdict is a one-sided McNemar test on
discordant pairs (z_crit 1.645); a class is flagged only if its own paired test
survives a Bonferroni correction, plus an unconditional hard floor at 0.25 for
catastrophic single-class collapse. A candidate is ACCEPTED only if every sub-check passes,
and `certified` is reported separately so "beneficial but uncertified" becomes
sayable -- the old gate had no way to express it.

Run:  python exp_gatefix.py [--first-seed 2 --seeds 3] [--escrow 60]
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
from cgrow.adolescence import consolidate_protos, sample_pool
from cgrow.gates import (territory_check, escrow_check, contract_check,
                         certify, escrow_sample, detectable_drop)
from exp_cifar import load_feats, CFG as CIFAR_CFG
from exp_pfc import evidence, train_arbiter
from exp_arbiter_source import who_and_pred
from exp_negotiate import negotiate

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FEATS = os.path.join(RES, "cifar100_vitb.npz")


def organism_answers(org: Organism, X):
    """Predictions plus 'was the answer inside an accepting territory'."""
    r = org.predict(X)
    ans = r["pred"]
    # an organism only ever answers via an organ that accepted, by construction
    return ans, ans != ABSTAIN, np.ones(len(X), dtype=bool)


def arbiter_answers(org, net, theta, X, gated):
    F, P = evidence(org, X)
    ans, who = who_and_pred(net, F, P, theta, gated=gated)
    acc = F[:, 3::4] > 0.5
    inside = acc[np.arange(len(X)), who]
    return ans, ans != ABSTAIN, inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--escrow", type=int, default=None,
                    help="samples/class; default = whole validation split")
    ap.add_argument("--out-name", default="gatefix.json")
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
            org0 = pickle.load(fh)
        last = np.isin(yva, tasks[-1])
        org, _ = consolidate_protos(org0, Xtr.shape[1], CIFAR_CFG, seed,
                                    Xva_last=Xva[last], yva_last=yva[last])

        # escrow from the HELD-OUT validation split (train flatters candidates)
        Xe, ye = escrow_sample(Xva, yva, args.escrow, seed)
        classes = np.asarray(sorted(set(int(c) for c in yte)))

        rng = np.random.default_rng(seed * 41 + 9)
        X_syn, y_syn = sample_pool(org, 3500, rng)
        perm = rng.permutation(len(X_syn))
        ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
        F_tr, P_tr = evidence(org, X_syn[tr])
        net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)
        F_ho, P_ho = evidence(org, X_syn[ho])
        thetas = {}
        for g in (False, True):
            best = (-1.0, None)
            for t in np.linspace(0.3, 0.9, 13):
                a, _ = who_and_pred(net, F_ho, P_ho, t, gated=g)
                sc = float((a == y_syn[ho]).mean())
                if sc > best[0]:
                    best = (sc, float(t))
            thetas["gated" if g else "free"] = best[1]

        ref_ho = org.predict(X_syn[ho])["pred"]
        ref_es, _, _ = organism_answers(org, Xe)
        ref_val = org.scores(Xva[last], yva[last])["strict"]
        ref_test = org.scores(Xte, yte)["strict"]

        cands = {}
        for g in (False, True):
            nm = "arbiter_gated" if g else "arbiter_free"
            th = thetas["gated" if g else "free"]
            a_ho, _, _ = arbiter_answers(org, net, th, X_syn[ho], g)
            a_es, _, _ = arbiter_answers(org, net, th, Xe, g)
            a_te, ans_te, ins_te = arbiter_answers(org, net, th, Xte, g)
            a_v, _, _ = arbiter_answers(org, net, th, Xva[last], g)
            cands[nm] = dict(ho=a_ho, es=a_es, te=a_te, answered=ans_te,
                             inside=ins_te,
                             val=float((a_v == yva[last]).mean()))
        for arm in ("market", "oracle"):
            new, _ = negotiate(org, X_syn[tr], y_syn[tr], arm, seed, CIFAR_CFG,
                               self_loss=0.10, margin=0.10, include_c=True)
            n_ho, _, _ = organism_answers(new, X_syn[ho])
            n_es, _, _ = organism_answers(new, Xe)
            n_te, ans_te, ins_te = organism_answers(new, Xte)
            cands[f"negotiate_{arm}"] = dict(
                ho=n_ho, es=n_es, te=n_te, answered=ans_te, inside=ins_te,
                val=new.scores(Xva[last], yva[last])["strict"])

        rec = {"seed": seed, "ref_test": ref_test,
               "escrow_per_class": args.escrow,
               "escrow_n": int(len(ye)),
               "escrow_resolution": detectable_drop(len(ye)),
               "candidates": {}}
        for nm, c in cands.items():
            old = contract_check(c["ho"], ref_ho, y_syn[ho],
                                 np.unique(y_syn[ho]), c["val"], ref_val)
            terr = territory_check(c["answered"], c["inside"])
            esc = escrow_check(c["es"], ref_es, ye, classes)
            v_old = certify([old])
            v_new = certify([old, terr, esc])
            test = float((c["te"] == yte).mean())
            rec["candidates"][nm] = {
                "test_strict": test, "test_delta": test - ref_test,
                "old_gate": v_old["accepted"], "new_gate": v_new["accepted"],
                "failed": v_new["failed"], "certified": v_new["certified"],
                "uncertified_fraction": v_new["uncertified_fraction"],
                "checks": v_new["checks"]}
        recs.append(rec)

        print(f"[cifar_vitb seed {seed}] ({time.time()-t0:.0f}s) reference "
              f"{ref_test:.4f} | escrow {args.escrow}/class "
              f"n={rec['escrow_n']}, resolution {rec['escrow_resolution']:.4f})", flush=True)
        for nm, d in rec["candidates"].items():
            print(f"    {nm:<16} test {d['test_strict']:.4f} "
                  f"({d['test_delta']:+.4f}) | OLD gate "
                  f"{'ACCEPT' if d['old_gate'] else 'refuse'} -> NEW gate "
                  f"{'ACCEPT' if d['new_gate'] else 'REFUSE'}"
                  f"{' (' + ','.join(d['failed']) + ')' if d['failed'] else ''}"
                  f" | uncertified {d['uncertified_fraction']:.3f}", flush=True)
        with open(os.path.join(RES, args.out_name), "w") as f:
            json.dump(recs, f, indent=1, default=float)
    print(f"wrote results/{args.out_name}")


if __name__ == "__main__":
    main()
