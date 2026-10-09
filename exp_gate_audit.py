"""Re-audit every lifecycle gate decision under the repaired gate.

The paper reports one measured gate false-accept: on wall-robot the arbiter
passes both gate signals and then regresses ~12 points on test, systematically,
on all three evaluation seeds. We called it "the honest boundary of data-free
gating" because detecting it seemed to need held-out old data.

The repaired gate (cgrow/gates.py) has two new checks, and they are not equally
priced, which is the point of this audit:

  territory  DATA-FREE and exact. If this alone refuses wall-robot, then the
             failure never needed real data at all -- it needed a structural
             check we had not written, and the "boundary of data-free gating"
             claim in the paper is too pessimistic.
  escrow     needs a little REAL held-out validation. If only this catches it,
             the paper's diagnosis was right and the fix has a data price.

Run over every MLP lifecycle benchmark and both gates, so the audit answers the
general question -- would the repaired gate have changed any decision the
lifecycle actually made? -- rather than just the robot one.

Run:  python exp_gate_audit.py [--benchmarks robot letter ...] [--seeds 3]
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
from cgrow.gates import (territory_check, escrow_check, certify, escrow_sample)
from exp_continual import run_ours, CFG as MLP_CFG
from exp_pfc import evidence, train_arbiter
from exp_arbiter_source import who_and_pred

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ALL = ["robot", "letter", "satimage", "digits", "mnist", "kmnist", "fashion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=ALL)
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--escrow", type=int, default=None,
                    help="samples/class; default = whole validation split")
    ap.add_argument("--out-name", default="gate_audit.json")
    args = ap.parse_args()

    out = {}
    for bench in args.benchmarks:
        cfg = MLP_CFG[bench]
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            X, y, _ = load_real(bench)
            splits = prepare(X, y, seed=seed)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            tasks = split_tasks(y, 2)
            torch.manual_seed(seed)
            org, _ = run_ours(tasks, splits, X.shape[1], cfg, seed)
            last = np.isin(yva, tasks[-1])

            org2, rep = consolidate_protos(org, X.shape[1], cfg, seed,
                                           Xva_last=Xva[last],
                                           yva_last=yva[last])
            ref_test = org2.scores(Xte, yte)["strict"]

            rng = np.random.default_rng(seed * 41 + 9)
            X_syn, y_syn = sample_pool(org, 2500, rng)
            perm = rng.permutation(len(X_syn))
            ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
            F_tr, P_tr = evidence(org2, X_syn[tr])
            net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)
            F_ho, P_ho = evidence(org2, X_syn[ho])
            best = (0.0, None)
            for th in np.linspace(0.3, 0.9, 13):
                a, _ = who_and_pred(net, F_ho, P_ho, th)
                s = float((a == y_syn[ho]).mean())
                if s > best[0]:
                    best = (s, float(th))
            theta = best[1]

            # --- the ORIGINAL lifecycle rule (exp_lifecycle.py) --------------
            a_ho, _ = who_and_pred(net, F_ho, P_ho, theta)
            classes_syn = np.asarray(sorted(set(int(c) for c in y_syn)))
            pc_new = _strict_per_class(a_ho, a_ho != ABSTAIN, y_syn[ho],
                                       classes_syn)
            r_ref = org2.predict(X_syn[ho])
            pc_ref = _strict_per_class(r_ref["pred"], r_ref["pred"] != ABSTAIN,
                                       y_syn[ho], classes_syn)
            F_v, P_v = evidence(org2, Xva[last])
            a_v, _ = who_and_pred(net, F_v, P_v, theta)
            val_new = float((a_v == yva[last]).mean())
            val_ref = org2.scores(Xva[last], yva[last])["strict"]
            old_accept = bool(pc_new.mean() >= pc_ref.mean() + 0.01
                              and pc_new.min() >= pc_ref.min() - 0.05
                              and val_new >= val_ref)

            # --- the REPAIRED checks ---------------------------------------
            Xe, ye = escrow_sample(Xva, yva, args.escrow, seed)
            classes = np.asarray(sorted(set(int(c) for c in yte)))
            F_e, P_e = evidence(org2, Xe)
            a_e, who_e = who_and_pred(net, F_e, P_e, theta)
            ref_e = org2.predict(Xe)["pred"]
            F_te, P_te = evidence(org2, Xte)
            a_te, who_te = who_and_pred(net, F_te, P_te, theta)
            acc_te = F_te[:, 3::4] > 0.5
            inside = acc_te[np.arange(len(Xte)), who_te]

            terr = territory_check(a_te != ABSTAIN, inside)
            esc = escrow_check(a_e, ref_e, ye, classes)
            v = certify([terr, esc])
            test = float((a_te == yte).mean())

            rec = {"seed": seed, "ref_test": ref_test, "arbiter_test": test,
                   "test_delta": test - ref_test, "old_accept": old_accept,
                   "territory_passed": terr["passed"],
                   "uncertified_fraction": terr["uncertified_fraction"],
                   "escrow_passed": esc["passed"], "escrow_z": esc["mcnemar_z"],
                   "escrow_n": int(len(ye)),
                   "new_accept": bool(old_accept and v["accepted"]),
                   "failed": v["failed"],
                   "adol_accepted": rep["accepted"]}
            recs.append(rec)
            verdict = ("kept" if rec["new_accept"] == old_accept
                       else ("NOW REFUSED" if old_accept else "now allowed"))
            print(f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) ref "
                  f"{ref_test:.4f} -> arbiter {test:.4f} "
                  f"({rec['test_delta']:+.4f}) | OLD "
                  f"{'ACCEPT' if old_accept else 'refuse'} | NEW "
                  f"{'ACCEPT' if rec['new_accept'] else 'REFUSE'} "
                  f"({','.join(v['failed']) if v['failed'] else 'clean'}) "
                  f"| uncert {terr['uncertified_fraction']:.3f} z "
                  f"{esc['mcnemar_z']:+.2f} | {verdict}", flush=True)
            out[bench] = recs
            with open(os.path.join(RES, args.out_name), "w") as f:
                json.dump(out, f, indent=1, default=float)
    print(f"wrote results/{args.out_name}")


if __name__ == "__main__":
    main()
