"""B-apoptosis -- gated organ death with a certificate of redundancy.

Development so far only ever adds. Biology also removes: cells whose function
is covered elsewhere die on schedule. The organism's version:

  certificate of redundancy (organ k): sample k's OWN contract (its promised
  territory, labels known); the organism WITHOUT k must, on those samples,
  keep coverage (>= 0.95 of what it answered with k) and accuracy (within
  0.02), and must not regress on the last task's real validation. All
  thresholds fixed a priori; run once on evaluation seeds.

  death: k's trunk and vetoes are freed; its CONTRACT is archived to the
  genome -- the spec survives the body, so future sleep cycles can still
  generate from the dead organ's territory (demonstrated: one post-apoptosis
  sleep night is rebuilt from the genome and gated as usual).

Every death is audited on real test data (flips and their direction). The
payoff metric is honest compression: parameters and organ count before/after
at matched accuracy. Expected to fire mostly AFTER consolidation (a nightly
or one-shot C shadows the cascade); on unconsolidated organisms deaths should
be rare -- the certificate says so, and that is the control.

Run:  python exp_apoptosis.py [--benchmarks digits kmnist cifar_vitb]
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
from cgrow.data import load_real, prepare, split_tasks
from cgrow.organ import Organ
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import ProtoOrgan, consolidate_protos, _gate_synthetic
from cgrow.proximity import merge_pca_family
from exp_continual import run_ours, CFG as MLP_CFG
from exp_cifar import load_feats, CFG as CIFAR_CFG

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COV_KEEP, ACC_SLACK, VAL_SLACK = 0.95, 0.02, 0.02   # preregistered
CLASS_SLACK = 0.05                                   # v4 per-class floor


def without(org: Organism, idx: int) -> Organism:
    o2 = Organism()
    o2.organs = [o for i, o in enumerate(org.organs) if i != idx]
    return o2


def redundancy_certificate(org: Organism, idx: int, Xva_last, yva_last, rng,
                           anchor=None):
    """Data-free check that organ idx's territory is covered without it.

    v2 (the ratchet fix): all comparisons anchor to the ORIGINAL organism's
    scores on a FIXED pooled-territory reference set, so a sequence of deaths
    can spend at most one slack in total, not one slack each. v1 compared
    against the current (already-shrunk) organism and leaked 8-10 points over
    nine deaths -- the compounding was invisible to per-death local checks."""
    o = org.organs[idx]
    X_t, y_t = o.prox.sample(2000, rng)
    org_wo = without(org, idx)
    r_w = org.predict(X_t)
    r_o = org_wo.predict(X_t)
    ans_w = r_w["pred"] != ABSTAIN
    if not ans_w.any():
        return True, {"why": "territory already unanswered"}
    cov_kept = float((r_o["pred"][ans_w] != ABSTAIN).mean())
    acc_w = float((r_w["pred"] == y_t).mean())
    acc_o = float((r_o["pred"] == y_t).mean())
    cert = {"cov_kept": cov_kept, "acc_with": acc_w, "acc_without": acc_o}
    # every gate is evaluated (no short-circuit) so a refusal names its cause
    fails = []
    if cov_kept < COV_KEEP:
        fails.append("cov")
    if acc_o < acc_w - ACC_SLACK:
        fails.append("acc_local")
    if anchor is not None:
        X_ref, y_ref, acc_ref0, val_ref0 = anchor[:4]
        p_wo = org_wo.predict(X_ref)["pred"]
        acc_g = float((p_wo == y_ref).mean())
        val_o = org_wo.scores(Xva_last, yva_last)["strict"]
        cert.update({"global_acc": acc_g, "global_anchor": acc_ref0,
                     "val_without": val_o, "val_anchor": val_ref0})
        if acc_g < acc_ref0 - ACC_SLACK:
            fails.append("acc_global")
        if val_o < val_ref0 - VAL_SLACK:
            fails.append("val")
        # v4: PER-CLASS floors against the ORIGINAL organism -- the measured
        # leak concentrates in class pockets that escrow MEANS cannot see
        if len(anchor) > 4:
            worst_c, worst_d = None, 0.0
            for c, ref_acc in anchor[4].items():
                m = y_ref == c
                if m.sum() >= 10:
                    drop = ref_acc - float((p_wo[m] == c).mean())
                    if drop > worst_d:
                        worst_c, worst_d = int(c), float(drop)
            cert["worst_class"] = worst_c
            cert["worst_class_drop"] = worst_d
            if worst_d > CLASS_SLACK:
                fails.append("class")
    cert["fails"] = fails
    return (not fails), cert


def apoptose(org: Organism, Xva_last, yva_last, Xte, yte, seed, escrow=None):
    """Greedy gated retirement, most-shadowed organs first (share of test they
    answer, measured on contract samples). Index 0 (a consolidated C) is never
    a candidate -- it is what shadows. Returns (organism, genome, log)."""
    rng = np.random.default_rng(seed * 53 + 11)
    # the global anchor: the ORIGINAL organism's quality on a fixed reference
    # (never re-based as organs die). v3: the reference may be a small REAL
    # certification escrow (used only by gates, never for training) -- the
    # data-free version failed twice with an identified mechanism (synthetic
    # territory samples are tail-blind; see report).
    if escrow is not None:
        X_ref, y_ref = escrow
    else:
        X_ref, y_ref = zip(*[o.prox.sample(600, rng) for o in org.organs])
        X_ref, y_ref = np.concatenate(X_ref), np.concatenate(y_ref)
    p0 = org.predict(X_ref)["pred"]
    acc_ref0 = float((p0 == y_ref).mean())
    val_ref0 = org.scores(Xva_last, yva_last)["strict"]
    pc_ref0 = {int(c): float((p0[y_ref == c] == c).mean())
               for c in np.unique(y_ref) if (y_ref == c).sum() >= 10}
    anchor = (X_ref, y_ref, acc_ref0, val_ref0, pc_ref0)
    genome, log, refusals = [], [], []
    first_sweep = True
    changed = True
    while changed:
        changed = False
        # candidate order: cascade organs by ascending answering share
        pool_X = np.concatenate([o.prox.sample(400, rng)[0]
                                 for o in org.organs])
        share = np.bincount(
            org.predict(pool_X)["organ"] + 1,
            minlength=len(org.organs) + 1)[1:]
        order = np.argsort(share)
        for idx in order:
            if isinstance(org.organs[idx], ProtoOrgan):
                continue
            ok, cert = redundancy_certificate(org, int(idx), Xva_last,
                                              yva_last, rng, anchor=anchor)
            if not ok:
                if first_sweep:
                    refusals.append(
                        {"classes": [int(c) for c in
                                     org.organs[int(idx)].class_ids],
                         **{k: v for k, v in cert.items()}})
                continue
            o = org.organs[int(idx)]
            before = org.predict(Xte)["pred"]
            org = without(org, int(idx))
            after = org.predict(Xte)["pred"]
            was = before != ABSTAIN
            flips = int((was & (after != before)).sum())
            genome.append({"class_ids": [int(c) for c in o.class_ids],
                           "prox": o.prox})
            log.append({"classes": [int(c) for c in o.class_ids],
                        **{k: v for k, v in cert.items()},
                        "real_test_flips": flips,
                        "params_freed": o.n_params()})
            changed = True
            break
        first_sweep = False
    return org, genome, log, refusals


def genome_sleep(org: Organism, genome, in_dim, cfg, Xva_last, yva_last,
                 X_ho_seed):
    """Demonstrate the genome works: rebuild C from live organs PLUS archived
    contracts of the dead, gate it as usual."""
    rng = np.random.default_rng(X_ho_seed)
    sources = [o.prox for o in org.organs
               if not isinstance(o, ProtoOrgan)] + [g["prox"] for g in genome]
    Xs, ys = zip(*(p.sample(2500, rng) for p in sources))
    X_syn, y_syn = np.concatenate(Xs), np.concatenate(ys)
    perm = rng.permutation(len(X_syn))
    ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
    classes = sorted(set(int(c) for c in y_syn))
    Xn = X_syn[tr] / (np.linalg.norm(X_syn[tr], axis=1, keepdims=True) + 1e-9)
    anchors = np.stack([Xn[y_syn[tr] == c].mean(axis=0) for c in classes])
    anchors /= (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    C = ProtoOrgan(in_dim, classes, anchors,
                   min_norm=float(np.linalg.norm(X_syn[tr], axis=1).min()),
                   seed=X_ho_seed)
    C.prox = merge_pca_family(sources)
    C.prox_name = C.prox.name
    if C.solve_thresholds(X_syn[ho], y_syn[ho], p_target=cfg["p_target"],
                          require=True) is None:
        return False
    C.freeze(X_syn[ho], y_syn[ho])
    ok, _ = _gate_synthetic(C, X_syn[ho], y_syn[ho], classes, cfg, ref_org=org)
    return bool(ok)


def build(bench, seed):
    if bench.startswith("cifar_"):            # cifar_<tag> -> results/cifar100_<tag>.npz
        cfg = CIFAR_CFG
        tag = bench.split("_", 1)[1]
        splits = load_feats(seed, feats_path=os.path.join(
            RES, f"cifar100_{tag}.npz"))
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
        cache = os.path.join(RES, f"adol_org_{tag}_{seed}.pkl")
        with open(cache, "rb") as fh:
            org = pickle.load(fh)
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
    ap.add_argument("--benchmarks", nargs="+",
                    default=["digits", "kmnist", "cifar_vitb"])
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--escrow", type=int, default=0,
                    help="real samples per task kept ONLY for death gates")
    args = ap.parse_args()

    out = {}
    for bench in args.benchmarks:
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            org, splits, tasks, cfg = build(bench, seed)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            last = np.isin(yva, tasks[-1])
            # consolidate first (apoptosis is expected to fire post-C);
            # if the gate refuses, apoptosis runs on the raw cascade (control)
            org2, rep = consolidate_protos(org, Xtr.shape[1], cfg, seed,
                                           Xva_last=Xva[last],
                                           yva_last=yva[last])
            escrow = None
            if args.escrow:
                # v4: PER-CLASS escrow (args.escrow samples per class), kept
                # only for death gates -- never for training
                rng_e = np.random.default_rng(seed * 71 + 3)
                Xe, ye = [], []
                for c in np.unique(ytr):
                    m = np.flatnonzero(ytr == c)
                    take = rng_e.choice(m, size=min(args.escrow, len(m)),
                                        replace=False)
                    Xe.append(Xtr[take]); ye.append(ytr[take])
                escrow = (np.concatenate(Xe), np.concatenate(ye))
            before = org2.scores(Xte, yte)
            p_before = org2.n_params()
            org3, genome, log, refusals = apoptose(org2, Xva[last], yva[last],
                                                   Xte, yte, seed,
                                                   escrow=escrow)
            after = org3.scores(Xte, yte)
            genome_ok = (genome_sleep(org3, genome, Xtr.shape[1], cfg,
                                      Xva[last], yva[last], seed * 7 + 1)
                         if genome else None)
            rec = {"seed": seed, "consolidated": rep["accepted"],
                   "deaths": len(log), "organs_before": len(org2.organs),
                   "organs_after": len(org3.organs),
                   "params_before": p_before, "params_after": org3.n_params(),
                   "strict_before": before["strict"],
                   "strict_after": after["strict"],
                   "total_flips": sum(l["real_test_flips"] for l in log),
                   "genome_sleep_ok": genome_ok, "log": log,
                   "refusals": refusals}
            recs.append(rec)
            print(f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) "
                  f"C={'yes' if rep['accepted'] else 'no'}  deaths "
                  f"{len(log)}/{len(org2.organs) - int(rep['accepted'])}  "
                  f"params {p_before} -> {org3.n_params()}  strict "
                  f"{before['strict']:.3f} -> {after['strict']:.3f}  "
                  f"flips {rec['total_flips']}  genome-sleep {genome_ok}",
                  flush=True)
            if refusals:
                tally = {}
                for r in refusals:
                    for f in r["fails"]:
                        tally[f] = tally.get(f, 0) + 1
                worst = max((r.get("worst_class_drop", 0.0) for r in refusals),
                            default=0.0)
                print(f"    refused {len(refusals)}: {tally}  "
                      f"worst per-class drop {worst:.3f}", flush=True)
        out[bench] = recs
        with open(os.path.join(RES, "apoptosis.json"), "w") as f:
            json.dump(out, f, indent=1,
                      default=lambda o: "<prox>" if hasattr(o, "score") else str(o))
    print("wrote results/apoptosis.json")


if __name__ == "__main__":
    main()
