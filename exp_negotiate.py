"""The market's real job: agreeing the CONTRACTS, not routing the queries.

Every market variant so far priced organs at INFERENCE time (who answers this
query?) and the verdict was mixed-to-negative. But the measured defect is not in
routing, it is in the BOUNDARIES: on CIFAR-ViT only 18.3% of the arbiter's
beneficial re-routes land on an organ that accepted the input, so the territories
themselves are a poor guide to who should answer. The arbiter wins by ignoring
them, and that is precisely why 94% of its gain is not contract-covered.

So use the market where the contracts are written. Each organ publishes a
pool-calibrated price curve (the one thing the market demonstrably does well:
Brier 0.71 -> 0.004 once calibrated on the shared pool), the organs bid over the
contested regions of the shared contract pool, and every region an organ LOSES is
removed from its territory by a veto. Vetoes are monotone -- an accepted set can
only shrink -- so the renegotiated contract is still a finite, checkable spec, and
first-accept routing over the new boundaries stays 100% contract-covered.

The hypothesis this tests: the arbiter's re-routing gain can be moved INSIDE the
certificate by fixing the boundaries instead of overriding them.

Three arms, all data-free (pool labels come from the contract spec itself):
  market   an organ cedes points where another organ outbids it
  oracle   an organ cedes points it gets WRONG that another organ gets RIGHT
           (upper bound on what any boundary agreement could buy)
  none     the unrenegotiated organism (control)

Preregistered before running: veto self-loss budget 0.02; a renegotiation is
ACCEPTED only if, on the synthetic holdout, per-class mean >= ref - 0.02 and
per-class floor >= ref floor - 0.05, and on the last task's real validation
strict >= ref - 0.05 (the same gate shape used for consolidation and the PFC).
Success = renegotiated cascade beats the cascade by >= +0.01 strict.

Run:  python exp_negotiate.py --benchmarks cifar_vitb [--first-seed 2 --seeds 3]
"""
from __future__ import annotations

import argparse
import copy
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
                               _strict_per_class, ProtoOrgan)
from cgrow.data import load_real, prepare, split_tasks
from exp_continual import run_ours, CFG as MLP_CFG
from exp_cifar import load_feats, CFG as CIFAR_CFG
from exp_market import fit_prices_pool, bids
from exp_pfc import evidence, train_arbiter
from exp_arbiter_source import who_and_pred

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SELF_LOSS = 0.02          # preregistered veto self-coverage budget
MARGIN = 0.10             # preregistered: how far the incumbent must be outbid
GATE_MEAN, GATE_FLOOR, GATE_VAL = 0.02, 0.05, 0.05
SUCCESS = 0.01


def accepts_matrix(org: Organism, X: np.ndarray):
    """(n, K) acceptance and (K, n) forced predictions."""
    A, P = [], []
    for o in org.organs:
        d1, m, p, d_in, vet, _ = o.evaluate(X)
        A.append(o.accept(d1, m, d_in, vet))
        P.append(p)
    return np.stack(A, axis=1), np.stack(P)


def negotiate(org: Organism, X_tr, y_tr, arm, seed, cfg,
              self_loss=SELF_LOSS, margin=MARGIN, include_c=False):
    """Return a renegotiated COPY plus a log of what each organ ceded."""
    new = copy.deepcopy(org)
    rng = np.random.default_rng(seed * 97 + 13)
    A, P = accepts_matrix(org, X_tr)
    # only points the cascade actually ROUTES to j are contestable: an organ
    # that is not the first accepter never answers them, so ceding them is a
    # pure loss of its own territory (the first formulation ceded ~2/3 of every
    # territory and every veto was correctly discarded by its self-loss check)
    routed = org.predict(X_tr)["organ"]

    if arm == "market":
        curves = fit_prices_pool(org, X_tr, y_tr)
        B, _ = bids(org, curves, X_tr)
        winner = B.argmax(axis=1)
        own = B[np.arange(len(B)), np.clip(routed, 0, None)]
        margin_thr = margin
        margin = B.max(axis=1) - own          # how much the incumbent is outbid
    else:                                     # oracle upper bound
        correct = P == y_tr[None, :]          # (K, n)
        score = correct.T * (1.0 + A)
        winner = score.argmax(axis=1)
        winner = np.where(score.max(axis=1) > 0, winner, -1)
        margin_thr = margin
        margin = np.ones(len(X_tr))

    log = []
    for j, o in enumerate(new.organs):
        if isinstance(o, ProtoOrgan) and not include_c:
            log.append({"organ": j, "ceded": 0, "vetoes": 0,
                        "why": "consolidated generalist is not a bidder"})
            continue
        lost = (routed == j) & (winner != j) & (winner >= 0) & (margin > margin_thr)
        if arm == "oracle":
            # only cede what this organ actually gets wrong
            lost &= P[j] != y_tr
        n_lost = int(lost.sum())
        if n_lost < 10:
            log.append({"organ": j, "ceded": n_lost, "vetoes": 0})
            continue
        X_self, _ = o.prox.sample(2000, rng)
        added = o.add_veto(X_tr[lost], theta_q=cfg.get("veto_theta_q", 0.985),
                           seed=seed * 7 + j, X_self=X_self,
                           max_self_loss=self_loss)
        log.append({"organ": j, "ceded": n_lost, "vetoes": int(added)})
    return new, log


def gate(new: Organism, ref: Organism, X_ho, y_ho, Xva_last, yva_last):
    classes = np.asarray(sorted(set(int(c) for c in y_ho)))
    r_new, r_ref = new.predict(X_ho), ref.predict(X_ho)
    pc_new = _strict_per_class(r_new["pred"], r_new["pred"] != ABSTAIN,
                               y_ho, classes)
    pc_ref = _strict_per_class(r_ref["pred"], r_ref["pred"] != ABSTAIN,
                               y_ho, classes)
    v_new = new.scores(Xva_last, yva_last)["strict"]
    v_ref = ref.scores(Xva_last, yva_last)["strict"]
    ok = (pc_new.mean() >= pc_ref.mean() - GATE_MEAN
          and pc_new.min() >= pc_ref.min() - GATE_FLOOR
          and v_new >= v_ref - GATE_VAL)
    return {"accepted": bool(ok), "new_mean": float(pc_new.mean()),
            "ref_mean": float(pc_ref.mean()), "new_floor": float(pc_new.min()),
            "ref_floor": float(pc_ref.min()), "val_new": float(v_new),
            "val_ref": float(v_ref)}


def build(bench, seed):
    if bench == "cifar_vitb":
        cfg = CIFAR_CFG
        splits = load_feats(seed, feats_path=os.path.join(
            RES, "cifar100_vitb.npz"))
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
        with open(os.path.join(RES, f"adol_org_vitb_{seed}.pkl"), "rb") as fh:
            org = pickle.load(fh)
        (Xtr, ytr), (Xva, yva), _ = splits
        last = np.isin(yva, tasks[-1])
        org, _ = consolidate_protos(org, Xtr.shape[1], cfg, seed,
                                    Xva_last=Xva[last], yva_last=yva[last])
    else:
        cfg = MLP_CFG[bench]
        X, y, _ = load_real(bench)
        splits = prepare(X, y, seed=seed)
        tasks = split_tasks(y, 2)
        torch.manual_seed(seed)
        org, _ = run_ours(tasks, splits, X.shape[1], cfg, seed)
    return org, splits, tasks, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["cifar_vitb"])
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pool", type=int, default=3500)
    ap.add_argument("--self-loss", type=float, nargs="+",
                    default=[0.02])
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--include-c", action="store_true",
                    help="let the consolidated generalist bid and cede too")
    ap.add_argument("--out-name", default="negotiate.json")
    args = ap.parse_args()

    out = {}
    for bench in args.benchmarks:
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            org, splits, tasks, cfg = build(bench, seed)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            last = np.isin(yva, tasks[-1])
            base = org.scores(Xte, yte)

            rng = np.random.default_rng(seed * 41 + 9)
            X_syn, y_syn = sample_pool(org, args.pool, rng)
            perm = rng.permutation(len(X_syn))
            ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]

            rec = {"seed": seed, "cascade": base["strict"],
                   "cascade_coverage": base["coverage"], "arms": {}}
            for arm, sl in [(a, b) for a in ("market", "oracle")
                            for b in args.self_loss]:
                new, log = negotiate(org, X_syn[tr], y_syn[tr], arm, seed, cfg,
                                     self_loss=sl, margin=args.margin,
                                     include_c=args.include_c)
                g = gate(new, org, X_syn[ho], y_syn[ho], Xva[last], yva[last])
                s = new.scores(Xte, yte)
                before = org.predict(Xte)["pred"]
                after = new.predict(Xte)["pred"]
                moved = (before != ABSTAIN) & (after != before)
                rec["arms"][f"{arm}@{sl}"] = {
                    "arm": arm, "self_loss": sl,
                    "strict": s["strict"], "coverage": s["coverage"],
                    "precision": s["precision"], "gate": g,
                    "delta": s["strict"] - base["strict"],
                    "ceded_total": sum(l["ceded"] for l in log),
                    "vetoes_added": sum(l["vetoes"] for l in log),
                    "moved": int(moved.sum()),
                    "moved_to_right": int((moved & (after == yte)).sum()),
                    "moved_to_wrong": int((moved & (before == yte)).sum()),
                    "log": log}

            # reference points: the free and territory-respecting arbiters
            F_tr, P_tr = evidence(org, X_syn[tr])
            net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)
            F_ho, P_ho = evidence(org, X_syn[ho])
            th = {}
            for gt in (False, True):
                best = (-1.0, None)
                for t in np.linspace(0.3, 0.9, 13):
                    a, _ = who_and_pred(net, F_ho, P_ho, t, gated=gt)
                    sc = float((a == y_syn[ho]).mean())
                    if sc > best[0]:
                        best = (sc, float(t))
                th["gated" if gt else "free"] = best[1]
            F_te, P_te = evidence(org, Xte)
            a_free, _ = who_and_pred(net, F_te, P_te, th["free"])
            a_gate, _ = who_and_pred(net, F_te, P_te, th["gated"], gated=True)
            rec["free_arbiter"] = float((a_free == yte).mean())
            rec["gated_arbiter"] = float((a_gate == yte).mean())

            recs.append(rec)
            def _pick(a, sl):
                return rec["arms"][f"{a}@{sl}"]
            m, o_ = _pick("market", args.self_loss[0]), _pick("oracle", args.self_loss[0])
            print(f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) cascade "
                  f"{base['strict']:.3f} | MARKET {m['strict']:.3f} "
                  f"({m['delta']:+.3f}, gate {m['gate']['accepted']}) | "
                  f"ORACLE {o_['strict']:.3f} ({o_['delta']:+.3f}, gate "
                  f"{o_['gate']['accepted']}) | free arb "
                  f"{rec['free_arbiter']:.3f} gated arb "
                  f"{rec['gated_arbiter']:.3f}", flush=True)
            print(f"    market: ceded {m['ceded_total']} vetoes "
                  f"{m['vetoes_added']} moved {m['moved']} "
                  f"({m['moved_to_right']} right / {m['moved_to_wrong']} wrong)"
                  f" cov {base['coverage']:.3f}->{m['coverage']:.3f} | "
                  f"oracle: ceded {o_['ceded_total']} vetoes "
                  f"{o_['vetoes_added']} moved {o_['moved']} "
                  f"({o_['moved_to_right']}/{o_['moved_to_wrong']}) cov "
                  f"{o_['coverage']:.3f}", flush=True)
            for a in ("market", "oracle"):
                line = "    %-7s " % a
                for sl in args.self_loss:
                    v = _pick(a, sl)
                    line += (f"| sl {sl}: {v['strict']:.3f} ({v['delta']:+.3f}) "
                             f"vet {v['vetoes_added']} moved {v['moved']}"
                             f"({v['moved_to_right']}/{v['moved_to_wrong']}) "
                             f"cov {v['coverage']:.3f} gate "
                             f"{'Y' if v['gate']['accepted'] else 'N'} ")
                print(line, flush=True)
            out[bench] = recs
            with open(os.path.join(RES, args.out_name), "w") as f:
                json.dump(out, f, indent=1, default=float)
    print(f"wrote results/{args.out_name}")


if __name__ == "__main__":
    main()
