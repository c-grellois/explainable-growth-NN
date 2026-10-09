"""Where does the arbiter's gain over first-accept actually come from?

On the merged twin pool the arbiter scored 0.964 against the cascade's 0.915
(+0.049). The arbiter has three powers the cascade does not, because
exp_pfc.evidence() hands it every organ's FORCED prediction and
arbiter_predict() answers with the argmax organ whether or not that organ
ACCEPTED the input:

  fill     answer where the cascade abstained (no organ accepted)
  reroute  answer differently where the cascade already answered
  drop     abstain (theta) where the cascade answered

An exact, additive decomposition: with c_i = 1 if the cascade is right (abstain
counts as wrong) and a_i = 1 if the arbiter is right, each partition's
contribution to the accuracy difference is sum(a_i - c_i)/n, and the partitions
sum to exactly delta-strict. Reported alongside:

  * how often the arbiter answers with an organ that did NOT accept -- i.e. how
    much of the gain is bought by leaving certified territory (guarantee-relevant)
  * arbiter gain at k=1 vs k=4, which separates "the arbiter fixes first-accept
    misrouting" (known since the PFC work) from "twin redundancy gives the
    arbiter something extra to work with" (the new question)

Run:  python exp_arbiter_source.py [--benchmarks digits] [--kmax 4] [--seeds 2]
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
from cgrow.adolescence import sample_pool
from exp_continual import run_ours, CFG as MLP_CFG
from exp_pfc import evidence, train_arbiter, arbiter_predict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def merge(orgs):
    m = Organism()
    m.organs = [o for g in orgs for o in g.organs]
    return m


def fit_arbiter(merged, seed):
    """Train on contract samples only; pick theta on the synthetic holdout."""
    rng = np.random.default_rng(seed * 61 + 7)
    X_syn, y_syn = sample_pool(merged, 2000, rng)
    perm = rng.permutation(len(X_syn))
    ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
    F_tr, P_tr = evidence(merged, X_syn[tr])
    net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)
    F_ho, P_ho = evidence(merged, X_syn[ho])
    thetas = {}
    for gated in (False, True):
        best = (-1.0, None)
        for th in np.linspace(0.3, 0.9, 13):
            a, _ = who_and_pred(net, F_ho, P_ho, th, gated=gated)
            sc = float((a == y_syn[ho]).mean())
            if sc > best[0]:
                best = (sc, float(th))
        thetas["gated" if gated else "free"] = best[1]
    return net, thetas


def who_and_pred(net, F, preds, theta, gated=False):
    """Replicate arbiter_predict but also return the chosen organ index.

    gated=True is the TERRITORY-RESPECTING arbiter: the choice is restricted to
    organs that ACCEPT the input, and it abstains when none do. Every answer is
    then inside some organ's certified territory, which the standard arbiter
    cannot promise -- it answers with the argmax organ's forced prediction
    whether or not that organ accepted."""
    with torch.no_grad():
        p = torch.sigmoid(net(torch.as_tensor(F, dtype=torch.float32))).numpy()
    if gated:
        acc = F[:, 3::4] > 0.5
        p = np.where(acc, p, -np.inf)
    who = p.argmax(axis=1)
    conf = p.max(axis=1)
    ans = preds[who, np.arange(preds.shape[1])]
    ok = conf >= theta
    if gated:
        ok &= np.isfinite(conf)          # nobody accepted -> abstain
    return np.where(ok, ans, ABSTAIN), who


def decompose(merged, net, thetas, Xte, yte):
    r = merged.predict(Xte)
    casc = r["pred"]
    F, preds = evidence(merged, Xte)
    theta = thetas["free"]
    arb, who = who_and_pred(net, F, preds, theta)
    # the territory-respecting variant, same net, same evidence
    garb, gwho = who_and_pred(net, F, preds, thetas["gated"], gated=True)

    # did the organ the arbiter picked actually accept this input?
    K = len(merged.organs)
    acc_flags = F[:, 3::4]                       # the per-organ accept feature
    picked_accepted = acc_flags[np.arange(len(Xte)), who] > 0.5

    c = (casc == yte).astype(int)                # abstain counts as wrong
    a = (arb == yte).astype(int)
    n = len(yte)

    c_ans, a_ans = casc != ABSTAIN, arb != ABSTAIN
    parts = {
        "same_answer": c_ans & a_ans & (arb == casc),
        "reroute": c_ans & a_ans & (arb != casc),
        "fill": (~c_ans) & a_ans,
        "drop": c_ans & (~a_ans),
        "both_abstain": (~c_ans) & (~a_ans),
    }
    out = {"cascade_strict": float(c.mean()), "arbiter_strict": float(a.mean()),
           "delta": float(a.mean() - c.mean()),
           "cascade_coverage": float(c_ans.mean()),
           "arbiter_coverage": float(a_ans.mean()),
           "theta": theta,
           "picked_organ_did_not_accept": float((~picked_accepted).mean()),
           "gain_from_outside_territory": float(
               ((a - c) * (~picked_accepted)).sum() / n),
           "channels": {}}
    gc_ = (garb == yte).astype(int)
    g_acc_flags = F[:, 3::4] > 0.5
    out["gated"] = {
        "strict": float(gc_.mean()),
        "coverage": float((garb != ABSTAIN).mean()),
        "theta": thetas["gated"],
        "delta_vs_cascade": float(gc_.mean() - (casc == yte).mean()),
        "delta_vs_free_arbiter": float(gc_.mean() - (arb == yte).mean()),
        # by construction every answer sits in an accepting organ's territory
        "answers_outside_territory": float(
            ((garb != ABSTAIN)
             & ~g_acc_flags[np.arange(len(Xte)), gwho]).mean()),
        "reroute_vs_cascade": int(((garb != ABSTAIN) & (casc != ABSTAIN)
                                   & (garb != casc)).sum()),
        "reroute_wrong_to_right": int((((casc == yte) == False)
                                       & (garb == yte)
                                       & (casc != ABSTAIN)
                                       & (garb != casc)).sum()),
        "reroute_right_to_wrong": int(((casc == yte)
                                       & ((garb == yte) == False)
                                       & (garb != ABSTAIN)
                                       & (garb != casc)).sum())}
    for name, mask in parts.items():
        contrib = float((a[mask] - c[mask]).sum() / n) if mask.any() else 0.0
        d = {"n": int(mask.sum()), "share": float(mask.mean()),
             "contribution": contrib}
        if name == "reroute" and mask.any():
            d["wrong_to_right"] = int(((c == 0) & (a == 1) & mask).sum())
            d["right_to_wrong"] = int(((c == 1) & (a == 0) & mask).sum())
            d["both_wrong"] = int(((c == 0) & (a == 0) & mask).sum())
            d["inside_territory"] = float(picked_accepted[mask].mean())
        if name == "fill" and mask.any():
            d["arbiter_acc_here"] = float(a[mask].mean())
            d["inside_territory"] = float(picked_accepted[mask].mean())
        if name == "drop" and mask.any():
            d["cascade_acc_here"] = float(c[mask].mean())
        out["channels"][name] = d
    out["channel_sum"] = float(sum(v["contribution"]
                                   for v in out["channels"].values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["digits"])
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out-name", default="arbiter_source.json")
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

            orgs = []
            for j in range(args.kmax):
                s = seed + 100 * j
                torch.manual_seed(s)
                g, _ = run_ours(tasks, splits, X.shape[1], cfg, s)
                orgs.append(g)

            per_k = []
            for k in (1, args.kmax):
                m = merge(orgs[:k])
                net, thetas = fit_arbiter(m, seed)
                d = decompose(m, net, thetas, Xte, yte)
                d["k"] = k
                d["organs"] = len(m.organs)
                per_k.append(d)
                ch = d["channels"]
                print(f"[{bench} seed {seed} k={k}] cascade "
                      f"{d['cascade_strict']:.3f} -> arbiter "
                      f"{d['arbiter_strict']:.3f} ({d['delta']:+.3f}) | "
                      f"fill {ch['fill']['contribution']:+.3f} "
                      f"reroute {ch['reroute']['contribution']:+.3f} "
                      f"drop {ch['drop']['contribution']:+.3f} | "
                      f"outside-territory answers "
                      f"{d['picked_organ_did_not_accept']:.2f} "
                      f"(gain {d['gain_from_outside_territory']:+.3f})",
                      flush=True)
                g = d["gated"]
                print(f"      TERRITORY-RESPECTING arbiter {g['strict']:.3f} "
                      f"(cov {g['coverage']:.3f}) = {g['delta_vs_cascade']:+.3f} "
                      f"vs cascade, {g['delta_vs_free_arbiter']:+.3f} vs free "
                      f"arbiter | outside-territory answers "
                      f"{g['answers_outside_territory']:.3f} | reroutes "
                      f"{g['reroute_wrong_to_right']}:"
                      f"{g['reroute_right_to_wrong']}", flush=True)
            recs.append({"seed": seed, "per_k": per_k,
                         "redundancy_bonus": float(
                             per_k[-1]["arbiter_strict"]
                             - per_k[0]["arbiter_strict"])})
            print(f"    ({time.time()-t0:.0f}s) arbiter k1 "
                  f"{per_k[0]['arbiter_strict']:.3f} -> k{args.kmax} "
                  f"{per_k[-1]['arbiter_strict']:.3f} = redundancy bonus "
                  f"{recs[-1]['redundancy_bonus']:+.3f}", flush=True)
            out[bench] = recs
            with open(os.path.join(RES, args.out_name), "w") as f:
                json.dump(out, f, indent=1, default=float)
    print(f"wrote results/{args.out_name}")


if __name__ == "__main__":
    main()
