"""B-lifecycle -- the full developmental lifecycle on EVERY benchmark.

growth (frozen organs) -> adolescence (verified prototype consolidation; the
gate may reject) -> prefrontal arbiter (trained on contract-sampled data only)
-> ultimate consolidation (relative gate + real-val check; drift audited).

Runs on the MLP ladder (digits/mnist/kmnist/fashion), the tabular suite
(letter/satimage/robot), CIFAR-100 on both backbones (cached organisms), and
the spatial sheet (sweeps/shapes/fsdd). Reports strict accuracy per stage; a
rejected stage carries the previous stage forward (that is the gate working).

Run:  python exp_lifecycle.py [--benchmarks ...] [--seeds 2]
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
from cgrow.data import load_real, prepare, split_tasks
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import consolidate_protos, sample_pool, _strict_per_class
from exp_continual import run_ours, CFG as MLP_CFG
from exp_cifar import load_feats, CFG as CIFAR_CFG
from exp_pfc import evidence, train_arbiter, arbiter_predict
from exp_arbiter_source import who_and_pred
from cgrow.curriculum import order_classes, tasks_from_order, cifar_superclass_map
from cgrow.gates import (territory_check, escrow_check,
                         escrow_sample, escrow_split, certify)
import exp_sheet as ES

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MLP_SETS = ["digits", "mnist", "kmnist", "fashion", "letter", "satimage", "robot"]
SHEET_SETS = ["sweeps", "shapes", "fsdd"]
CIFAR_SETS = ["cifar_r18", "cifar_vitb"]
POOL_TOTAL = 40000      # consolidation pool budget (= 10 organs x 4000)
ARB_TOTAL = 25000       # arbiter pool budget (= 10 organs x 2500)


def build(bench: str, seed: int, class_order_seed=None, order_policy="random",
          inc_override=None):
    """Return (org, splits, tasks, cfg, Xva_last, yva_last).

    class_order_seed permutes the class order before grouping into tasks (the
    default is ascending class index, so class-IL results could in principle be
    order-sensitive; this is the check). order_policy applies a CURRICULUM from
    cgrow.curriculum (easy / hard / grouped / spread) computed on the training
    split before any organ exists; "random" keeps the protocol's own order. A
    non-random policy is a different protocol and bypasses the cached CIFAR
    organisms (cache files carry the policy in their name)."""
    if bench in MLP_SETS:
        cfg = MLP_CFG[bench]
        X, y, C = load_real(bench)
        splits = prepare(X, y, seed=seed)
        tasks = split_tasks(y, 2)
        if class_order_seed is not None:
            n_cls = int(y.max()) + 1
            order = np.random.default_rng(class_order_seed).permutation(n_cls)
            tasks = [sorted(int(c) for c in order[i:i + 2])
                     for i in range(0, n_cls, 2)]
        if order_policy != "random":
            order = order_classes(order_policy, splits[0][0], splits[0][1],
                                  seed, inc=2)
            tasks = tasks_from_order(order, 2)
        torch.manual_seed(seed)
        org, _ = run_ours(tasks, splits, X.shape[1], cfg, seed)
    elif bench.startswith(("cifar_", "ptm_")):
        # cifar_<tag> -> results/cifar100_<tag>.npz (100 classes, 10 per task)
        # ptm_<tag>   -> results/ptm_<tag>.npz (n_classes / inc stored in the cache)
        cfg = CIFAR_CFG
        prefix, tag = bench.split("_", 1)
        feats = os.path.join(
            RES, f"cifar100_{tag}.npz" if prefix == "cifar" else f"ptm_{tag}.npz")
        meta = np.load(feats, allow_pickle=False)
        n_cls = (int(meta["n_classes"]) if "n_classes" in meta.files
                 else int(meta["ytr"].max()) + 1)
        inc = int(meta["inc"]) if "inc" in meta.files else 10
        if inc_override:                  # B-horizon: a longer stream, same order
            inc = int(inc_override)
        del meta
        splits = load_feats(seed, feats_path=feats)
        order = np.random.default_rng(seed + 400).permutation(n_cls)
        tasks = [sorted(int(c) for c in order[i:i + inc])
                 for i in range(0, n_cls, inc)]
        if order_policy != "random":
            sup = cifar_superclass_map() if (prefix == "cifar" and n_cls == 100) else None
            order = order_classes(order_policy, splits[0][0], splits[0][1],
                                  seed, inc=inc, superclass=sup)
            tasks = tasks_from_order(order, inc)
        horizon = f"_inc{inc}" if inc_override else ""
        cache = os.path.join(RES, f"adol_org_{tag}_{seed}"
                             f"{'' if order_policy == 'random' else '_ord-' + order_policy}"
                             f"{horizon}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as fh:
                org = pickle.load(fh)
        else:
            torch.manual_seed(seed)
            org, _ = run_ours(tasks, splits, splits[0][0].shape[1], cfg, seed)
            with open(cache, "wb") as fh:
                pickle.dump(org, fh)
    else:                                   # sheet (fsdd_spk = speaker-disjoint)
        cfg = ES.CFG[bench.replace("_spk", "")]
        splits_all = ES.make_data(bench, seed)
        splits, in_dim = splits_all[:3], splits_all[3]
        n_cls = int(splits[0][1].max()) + 1
        ES.TASKS = [[2 * i, 2 * i + 1] for i in range(n_cls // 2)]
        tasks = ES.TASKS
        torch.manual_seed(seed)
        org, _ = ES.run_ours(splits, in_dim, cfg, seed, listen=True)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    last = np.isin(yva, tasks[-1])
    return org, splits, tasks, cfg, Xva[last], yva[last]


def lifecycle(bench: str, seed: int, arbiter: str = "gated",
              gate: str = "repaired", escrow_per_class=None,
              escrow_split_val: bool = False, class_order_seed=None,
              order_policy: str = "random", inc_override=None):
    org, splits, tasks, cfg, Xva_last, yva_last = build(
        bench, seed, class_order_seed=class_order_seed,
        order_policy=order_policy, inc_override=inc_override)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    # independence ablation: half A of validation feeds the lifecycle's own
    # gates (adolescence, the original rule), half B is the escrow
    if escrow_split_val:
        idxA, idxB = escrow_split(Xva, yva, seed)
        Xva_g, yva_g = Xva[idxA], yva[idxA]
        m_last = np.isin(yva_g, tasks[-1])
        Xva_last, yva_last = Xva_g[m_last], yva_g[m_last]
        Xva_e, yva_e = Xva[idxB], yva[idxB]
    else:
        Xva_e, yva_e = Xva, yva
    in_dim = Xtr.shape[1]
    out = {"seed": seed, "growth": org.scores(Xte, yte)["strict"],
           "order_policy": order_policy, "tasks": tasks, "inc": len(tasks[0]),
           "n_tasks": len(tasks),
           "growth_coverage": org.scores(Xte, yte)["coverage"],
           "n_organs": len(org.organs),
           # inhibition burden of this order: vetoes added over the whole life
           "vetoes_total": int(sum(e.get("vetoes_added", 0) + e.get("vetoes", 0)
                                   for e in org.log
                                   if e.get("event") in ("tighten", "birth_veto")))}

    # B-horizon: the synthetic pools are bounded by a TOTAL budget equal to the
    # inc-10 values (10 organs x 4000 / 10 x 2500), so a 100-organ organism does
    # not need 10x the memory and every <= 10-organ run is unchanged
    n_pool = max(200, min(4000, POOL_TOTAL // len(org.organs)))
    org2, rep = consolidate_protos(org, in_dim, cfg, seed,
                                   Xva_last=Xva_last, yva_last=yva_last,
                                   Xva=Xva_e, yva=yva_e, n_per_organ=n_pool)
    out["adol_accepted"] = rep["accepted"]
    out["adol_refused_by"] = rep.get("refused_by")
    out["adol_escrow"] = rep.get("escrow")
    out["adolescence"] = org2.scores(Xte, yte)["strict"]

    # PFC arbiter over whatever survived, trained on contract samples only
    rng = np.random.default_rng(seed * 41 + 9)
    X_syn, y_syn = sample_pool(org, max(100, min(2500, ARB_TOTAL // len(org.organs))), rng)
    perm = rng.permutation(len(X_syn))
    ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
    F_tr, P_tr = evidence(org2, X_syn[tr])
    net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)
    F_ho, P_ho = evidence(org2, X_syn[ho])
    GATED = arbiter == "gated"
    best = (0.0, None)
    for theta in np.linspace(0.3, 0.9, 13):
        a_, _ = who_and_pred(net, F_ho, P_ho, theta, gated=GATED)
        s = float((a_ == y_syn[ho]).mean())
        if s > best[0]:
            best = (s, float(theta))
    theta = best[1]
    a_ho, _ = who_and_pred(net, F_ho, P_ho, theta, gated=GATED)
    out["arbiter_kind"] = arbiter
    classes = np.asarray(sorted(set(int(c) for c in y_syn)))
    pc_new = _strict_per_class(a_ho, a_ho != ABSTAIN, y_syn[ho], classes)
    r_ref = org2.predict(X_syn[ho])
    pc_ref = _strict_per_class(r_ref["pred"], r_ref["pred"] != ABSTAIN,
                               y_syn[ho], classes)
    F_v, P_v = evidence(org2, Xva_last)
    a_v, _ = who_and_pred(net, F_v, P_v, theta, gated=GATED)
    val_new = float((a_v == yva_last).mean())
    val_ref = org2.scores(Xva_last, yva_last)["strict"]
    # the arbiter must strictly IMPROVE, not merely match: a routing module that
    # only matches carries synthetic-real risk with no upside (robot taught this)
    # 2026-09-07 GPU-box fix: _strict_per_class is NaN for a class with no
    # sample in the synthetic holdout (possible on datasets with rare
    # classes, e.g. ImageNet-A; never on CIFAR). Plain mean/min would then be
    # NaN and every comparison False -> a refusal by arithmetic, not by
    # evidence. Aggregate over the observed classes instead; the rule itself
    # (strict improvement of the mean, floor within 0.05, no val regression)
    # is unchanged.
    accepted = (np.nanmean(pc_new) >= np.nanmean(pc_ref) + 0.01
                and np.nanmin(pc_new) >= np.nanmin(pc_ref) - 0.05
                and val_new >= val_ref)
    out["arbiter_classes_unobserved_in_holdout"] = int(np.isnan(pc_new).sum())
    out["pfc_accepted_old"] = bool(accepted)

    # --- the REPAIRED gate: territory (data-free) + escrow (McNemar on a
    # --- held-out validation escrow), applied on top of the original rule
    F_te, P_te = evidence(org2, Xte)
    a_te, who_te = who_and_pred(net, F_te, P_te, theta, gated=GATED)
    inside_te = (F_te[:, 3::4] > 0.5)[np.arange(len(Xte)), who_te]
    terr = territory_check(a_te != ABSTAIN, inside_te)
    Xe, ye = escrow_sample(Xva_e, yva_e, escrow_per_class, seed)
    F_e, P_e = evidence(org2, Xe)
    a_e, _ = who_and_pred(net, F_e, P_e, theta, gated=GATED)
    esc = escrow_check(a_e, org2.predict(Xe)["pred"], ye,
                       np.asarray(sorted(set(int(c) for c in yte))))
    v = certify([terr, esc])
    out["territory_passed"] = terr["passed"]
    out["uncertified_fraction"] = terr["uncertified_fraction"]
    out["escrow_passed"] = esc["passed"]
    out["escrow_z"] = esc["mcnemar_z"]
    # which part of the escrow fired (2026-09-14: a refusal at z +4.9 on the
    # inc-5 horizon could not be explained from the record without this)
    out["escrow_detail"] = {k: esc[k] for k in
                            ("flagged_classes", "hard_floor_classes", "near_floor_classes",
                             "worst_class", "worst_class_z", "z_crit_per_class",
                             "overall_new", "overall_ref", "n_samples")}
    out["repaired_failed"] = v["failed"]
    if gate == "repaired":
        accepted = bool(accepted and v["accepted"])
    out["pfc_accepted"] = accepted

    # the free arbiter on the same organism, for the comparison table
    th_f = (0.0, None)
    for theta_f in np.linspace(0.3, 0.9, 13):
        a_, _ = who_and_pred(net, F_ho, P_ho, theta_f, gated=False)
        sc = float((a_ == y_syn[ho]).mean())
        if sc > th_f[0]:
            th_f = (sc, float(theta_f))
    a_free, _ = who_and_pred(net, F_te, P_te, th_f[1], gated=False)
    out["pfc_free_test"] = float((a_free == yte).mean())
    # online fidelity certificate: the last task's REAL train data vs its own
    # organ's contract samples -- the one sampler-fidelity check available
    # without any stored data. High AUROC + synthetic gain >> real-val gain
    # marks a gate at risk (the robot case).
    from exp_sampler import disc_auroc
    last_cls = tasks[-1]
    m_last = np.isin(ytr, last_cls)
    Xr = Xtr[m_last]
    Xs, _ = org.organs[-1].prox.sample(len(Xr), rng)
    half = len(Xr) // 2
    fid = disc_auroc(Xr[:half], Xs[:half], Xr[half:], Xs[half:2 * half], seed)
    out["gate"] = {"syn_mean": float(pc_new.mean()),
                   "syn_mean_ref": float(pc_ref.mean()),
                   "syn_floor": float(pc_new.min()),
                   "syn_floor_ref": float(pc_ref.min()),
                   "val_new": val_new, "val_ref": val_ref,
                   "fidelity_auroc": fid,
                   "at_risk": bool(fid > 0.9 and
                                   (pc_new.mean() - pc_ref.mean())
                                   > 3 * max(val_new - val_ref, 0.005))}
    out["pfc_candidate_test"] = float((a_te == yte).mean())
    out["pfc"] = (out["pfc_candidate_test"] if accepted
                  else out["adolescence"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+",
                    default=MLP_SETS + CIFAR_SETS + SHEET_SETS)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--arbiter", default="gated", choices=["gated", "free"],
                    help="territory-respecting (default) or the free arbiter")
    ap.add_argument("--gate", default="repaired",
                    choices=["repaired", "old"])
    ap.add_argument("--out-name", default="lifecycle_gated.json")
    ap.add_argument("--escrow-split", action="store_true",
                    help="validation halves: A for lifecycle gates, B escrow")
    ap.add_argument("--class-order-seed", type=int, default=None,
                    help="permute class order before grouping into tasks")
    ap.add_argument("--order-policy", default="random",
                    choices=["random", "easy", "hard", "grouped", "spread"],
                    help="class-order curriculum (a different protocol; see "
                         "cgrow/curriculum.py)")
    ap.add_argument("--inc", type=int, default=None,
                    help="classes per task for cifar_/ptm_ benchmarks (B-horizon); "
                         "organisms are cached as adol_org_<tag>_<seed>_inc<k>.pkl")
    args = ap.parse_args()

    path = os.path.join(RES, args.out_name)
    all_out = {}
    if os.path.exists(path):
        with open(path) as f:
            all_out = json.load(f)
    for bench in args.benchmarks:
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            r = lifecycle(bench, seed, arbiter=args.arbiter,
                          gate=args.gate,
                          escrow_split_val=args.escrow_split,
                          class_order_seed=args.class_order_seed,
                          order_policy=args.order_policy,
                          inc_override=args.inc)
            recs.append(r)
            print(f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) "
                  f"growth {r['growth']:.3f} -> adol "
                  f"{'ok' if r['adol_accepted'] else 'REJ'} "
                  f"{r['adolescence']:.3f} -> pfc "
                  f"{'ok' if r['pfc_accepted'] else 'REJ'} {r['pfc']:.3f}"
                  f"  [cand {r['pfc_candidate_test']:.3f} | free "
                  f"{r['pfc_free_test']:.3f} | uncert "
                  f"{r['uncertified_fraction']:.3f} z {r['escrow_z']:+.2f}"
                  f"{' fail:' + ','.join(r['repaired_failed']) if r['repaired_failed'] else ''}]",
                  flush=True)
        all_out[bench] = recs
        with open(path, "w") as f:
            json.dump(all_out, f, indent=1, default=float)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
