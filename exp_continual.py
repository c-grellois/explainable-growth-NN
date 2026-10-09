"""B2 -- class-incremental learning: split-digits and split-MNIST, single head,
no task oracle at test time (class-IL, the setting where regularization methods
collapse to ~20% in the literature).

Ours: one organ per task, grown on what the organism declines; before each growth,
existing organs are monotonically TIGHTENED against the incoming task's data (every
accept on unseen classes is a measured false accept; tightening excludes them under
a self-coverage budget estimated from each organ's stored birth statistics -- no raw
data retained). Answers never flip: retention is checked bitwise, not assumed.

Baselines: sequential fine-tune, EWC, replay (raw-data buffer), all the same MLP.

Run:  python exp_continual.py [--dataset digits|mnist] [--seeds N]
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
from cgrow.organ import Organ
from cgrow.organism import Organism, ABSTAIN
from cgrow.baselines import MLP, run_sequential, accuracy, train_plain

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)

CFG = {
    "digits": dict(hidden=(64,), feat=16, cap=2.0, epochs=60, p_target=0.97,
                   b_hidden=(128,), b_epochs=40, max_train=None, prox="pca",
                   k_centroids=5, q_cov_in=0.99, self_budget=0.0,
                   veto_theta_q=0.985),
    "mnist": dict(hidden=(256,), feat=32, cap=2.5, epochs=25, p_target=0.97,
                  b_hidden=(256,), b_epochs=15, max_train=None, prox="pca",
                  k_centroids=25, q_cov_in=0.99, self_budget=0.0,
                  veto_theta_q=0.985),
    "kmnist": dict(hidden=(256,), feat=32, cap=2.5, epochs=25, p_target=0.95,
                   b_hidden=(256,), b_epochs=15, max_train=None, prox="pca",
                   k_centroids=25, q_cov_in=0.99, self_budget=0.0,
                   veto_theta_q=0.985),
    "fashion": dict(hidden=(256,), feat=32, cap=2.5, epochs=25, p_target=0.93,
                    b_hidden=(256,), b_epochs=15, max_train=None, prox="pca",
                    k_centroids=25, q_cov_in=0.99, self_budget=0.0,
                    veto_theta_q=0.985),
    # tabular suite (openml, cached): letter = the longest task stream (13x2)
    "letter": dict(hidden=(128,), feat=32, cap=2.5, epochs=50, p_target=0.9,
                   b_hidden=(128,), b_epochs=40, max_train=None, prox="pca",
                   q_cov_in=0.99, self_budget=0.0, veto_theta_q=0.985),
    "satimage": dict(hidden=(96,), feat=24, cap=2.5, epochs=50, p_target=0.9,
                     b_hidden=(128,), b_epochs=40, max_train=None, prox="pca",
                     q_cov_in=0.99, self_budget=0.0, veto_theta_q=0.985),
    "robot": dict(hidden=(96,), feat=24, cap=2.5, epochs=50, p_target=0.9,
                  b_hidden=(128,), b_epochs=40, max_train=None, prox="pca",
                  q_cov_in=0.99, self_budget=0.0, veto_theta_q=0.985),
}


def run_ours(tasks, splits, in_dim, cfg, seed):
    self_budget = cfg.get("self_budget", 0.05)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    org = Organism()
    per_task_acc = []          # after task t: strict/forced acc on each seen task's test
    fa_log = []
    growth_flips = 0           # answer changes caused by GROWTH: must be exactly 0
    tighten_flips = {"n": 0, "to_right": 0, "to_wrong": 0}   # logged repairs
    for t, cls in enumerate(tasks):
        tr_mask = np.isin(ytr, cls)
        va_mask = np.isin(yva, cls)
        Xt_tr, yt_tr = Xtr[tr_mask], ytr[tr_mask]
        Xt_va, yt_va = Xva[va_mask], yva[va_mask]

        if org.organs:
            # every accept on the incoming task is a false accept: measure, tighten
            p_pre = org.predict(Xte)["pred"]
            r0 = org.predict(Xt_tr)
            fa_before = float((r0["pred"] != ABSTAIN).mean())
            org.tighten_against(Xt_tr, self_budget=self_budget,
                                veto=cfg.get("veto", True),
                                veto_theta_q=cfg.get("veto_theta_q", 0.97))
            r1 = org.predict(Xt_tr)
            fa_after = float((r1["pred"] != ABSTAIN).mean())
            p_post = org.predict(Xte)["pred"]
            fl = (p_pre != ABSTAIN) & (p_post != ABSTAIN) & (p_pre != p_post)
            tighten_flips["n"] += int(fl.sum())
            tighten_flips["to_right"] += int((fl & (p_post == yte)).sum())
            tighten_flips["to_wrong"] += int((fl & (p_pre == yte)).sum())
        else:
            fa_before = fa_after = 0.0
        fa_log.append({"task": t, "fa_before": fa_before, "fa_after": fa_after})

        p_before_growth = p_post if org.organs else None
        need = org.declined_mask(Xt_tr) if org.organs else np.ones(len(Xt_tr), bool)
        cand = Organ(in_dim, cfg["feat"], cfg["hidden"], cap=cfg["cap"],
                     class_ids=cls, seed=seed * 100 + t,
                     k_centroids=cfg.get("k_centroids", 5),
                     q_cov_in=cfg.get("q_cov_in", 0.995),
                     prox=cfg.get("prox", "kmeans"))
        cand.train_proto(Xt_tr[need], yt_tr[need], epochs=cfg["epochs"],
                         prune=cfg.get("prune", 0.0),
                         prune_rounds=cfg.get("prune_rounds", 1),
                         prune_epochs=cfg.get("prune_epochs"))
        grew = org.try_grow(cand, Xt_va, yt_va, p_target=cfg["p_target"],
                            gate=(t > 0), label=f"task{t}")
        # `t > 0` is not the same as "an earlier organ exists": if growth was
        # REFUSED on every earlier task the cascade is still empty, and the
        # birth veto has nothing to sample negatives from. That case never
        # arose until an organ was pruned hard enough to fail its own
        # precision target (2026-10-02, exp_prune at sparsity 0.998, which
        # crashed here on an empty concatenate).
        if (grew and len(org.organs) > 1 and cfg.get("prox") in ("pca", "mix")
                and cfg.get("veto", True)):
            # BIRTH VETO: the newborn is tightened against synthetic negatives
            # sampled from every earlier organ's contract -- the contracts are
            # data generators, so the new->old direction closes with no raw data
            rng = np.random.default_rng(seed * 10 + t)
            Xneg = np.concatenate([o.prox.sample(2000, rng)[0]
                                   for o in org.organs[:-1]])
            nv = org.organs[-1].add_veto(Xneg,
                                         theta_q=cfg.get("veto_theta_q", 0.97),
                                         seed=seed * 100 + 50 + t,
                                         X_self=Xt_va, max_self_loss=0.03)
            org.log.append({"event": "birth_veto", "task": t, "vetoes": nv})
        if p_before_growth is not None:
            p_after = org.predict(Xte)["pred"]
            was = p_before_growth != ABSTAIN
            growth_flips += int((was & (p_after != p_before_growth)).sum())

        row = []
        for i in range(t + 1):
            m = np.isin(yte, tasks[i])
            row.append(org.scores(Xte[m], yte[m]))
        per_task_acc.append(row)
    seen_mask = np.isin(yte, [c for cl in tasks for c in cl])
    final = org.scores(Xte[seen_mask], yte[seen_mask])
    return org, {"final": final, "per_task": per_task_acc,
                 "growth_flips": growth_flips, "tighten_flips": tighten_flips,
                 "false_accepts": fa_log, "n_params": org.n_params(),
                 "log": org.log}


def run_baseline(method, tasks, splits, in_dim, n_classes, cfg, seed):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    # stronger baselines (rigor pass): z-scored features (cures vanishing
    # gradients on min-max backbone features) and doubled training
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    splits = ((Xtr, ytr), (Xva, yva), (Xte, yte))
    cfg = {**cfg, "b_epochs": 2 * cfg["b_epochs"]}
    tasks_data = [(Xtr[np.isin(ytr, cls)], ytr[np.isin(ytr, cls)]) for cls in tasks]
    # per-task accuracy matrix: retrain incrementally, evaluating after each task
    from cgrow.baselines import MLP, _epochs, _fisher_diag
    import cgrow.baselines as B
    model = MLP(in_dim, cfg["b_hidden"], n_classes, seed=seed)
    anchors_prev = []
    buf_X = np.empty((0, in_dim)); buf_y = np.empty((0,), dtype=np.int64)
    acc_matrix = []
    for t, (Xt, yt) in enumerate(tasks_data):
        extra = None
        if method == "ewc" and anchors_prev:
            def extra(m, anchors_prev=anchors_prev):
                loss = 0.0
                for params_star, fisher in anchors_prev:
                    for p, ps, f in zip(m.parameters(), params_star, fisher):
                        loss = loss + (f * (p - ps) ** 2).sum()
                return 5000.0 * loss
        buffer = (buf_X, buf_y) if method == "replay" and len(buf_X) else None
        _epochs(model, Xt, yt, cfg["b_epochs"], 1e-3, 256, seed + t,
                extra_loss=extra, buffer=buffer)
        if method == "ewc":
            anchors_prev.append(([p.detach().clone() for p in model.parameters()],
                                 _fisher_diag(model, Xt, yt, seed=seed + t)))
        if method == "replay":
            rng = np.random.default_rng(seed + t)
            for c in np.unique(yt):
                sel = np.flatnonzero(yt == c)
                take = rng.choice(sel, size=min(100, len(sel)), replace=False)
                buf_X = np.concatenate([buf_X, Xt[take]])
                buf_y = np.concatenate([buf_y, yt[take]])
        row = []
        for i in range(t + 1):
            m = np.isin(yte, tasks[i])
            row.append(accuracy(model, Xte[m], yte[m])[0])
        acc_matrix.append(row)
    seen_mask = np.isin(yte, [c for cl in tasks for c in cl])
    final_acc = accuracy(model, Xte[seen_mask], yte[seen_mask])[0]
    forgetting = float(np.mean([acc_matrix[i][i] - acc_matrix[-1][i]
                                for i in range(len(tasks))]))
    return {"final_acc": final_acc, "acc_matrix": acc_matrix,
            "forgetting": forgetting, "n_params": model.n_params()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=0,
                    help="seed discipline: dev = 0-1, eval = 2+")
    args = ap.parse_args()
    cfg = CFG[args.dataset]

    X, y, C = load_real(args.dataset)
    records = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.time()
        splits = prepare(X, y, seed=seed, max_train=cfg["max_train"])
        tasks = split_tasks(y, classes_per_task=2)
        in_dim = X.shape[1]
        torch.manual_seed(seed)

        _, ours = run_ours(tasks, splits, in_dim, cfg, seed)
        # ours forgetting, strict: acc on task i after learning i vs at the end
        pt = ours["per_task"]
        forget_strict = float(np.mean([pt[i][i]["strict"] - pt[-1][i]["strict"]
                                       for i in range(len(tasks))]))
        base = {m: run_baseline(m, tasks, splits, in_dim, C, cfg, seed)
                for m in ["finetune", "ewc", "replay"]}
        # joint upper bound: all data at once -- strengthened (z-scored,
        # doubled width and epochs) so "beats joint" claims face a real joint
        (Xtr, ytr), _, (Xte, yte) = splits
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
        joint = train_plain((Xtr - mu) / sd, ytr, C,
                            hidden=tuple(2 * h for h in cfg["b_hidden"]),
                            epochs=2 * cfg["b_epochs"], seed=seed)
        joint_acc = accuracy(joint, (Xte - mu) / sd, yte)[0]

        rec = {"seed": seed, "ours": ours, "ours_forget_strict": forget_strict,
               "baselines": base, "joint_acc": joint_acc}
        records.append(rec)
        fa = ours["false_accepts"]
        print(f"[{args.dataset} seed {seed}] ({time.time()-t0:.0f}s)")
        tf = ours["tighten_flips"]
        print(f"  ours   : strict {ours['final']['strict']:.3f}  forced {ours['final']['forced']:.3f}"
              f"  cov {ours['final']['coverage']:.3f}  prec {ours['final']['precision']:.3f}"
              f"  forget(strict) {forget_strict:+.3f}  params {ours['n_params']}")
        print(f"           growth flips (must be 0): {ours['growth_flips']}  |  "
              f"tightening re-routes: {tf['n']} ({tf['to_right']} to-right, {tf['to_wrong']} to-wrong)")
        print(f"  false-accept on incoming task (before->after tightening): "
              + "  ".join(f"t{f['task']}: {f['fa_before']:.2f}->{f['fa_after']:.2f}"
                          for f in fa[1:]))
        for m in ["finetune", "ewc", "replay"]:
            b = base[m]
            print(f"  {m:8s}: final {b['final_acc']:.3f}  forgetting {b['forgetting']:+.3f}"
                  f"  params {b['n_params']}")
        print(f"  joint  : {joint_acc:.3f}")

    out = os.path.join(RES, f"continual_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump(records, f, indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
