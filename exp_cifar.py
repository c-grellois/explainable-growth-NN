"""B-cifar -- the pretrained-backbone CIFAR-100 test (the make-or-break regime).

Class-incremental learning on frozen ImageNet-pretrained ResNet-18 features
(extracted once by extract_cifar_feats.py): 10 tasks x 10 classes, single head,
no task oracle, class order shuffled per seed, abstention counted as error.

Contestants, all on IDENTICAL features:
  organism   one organ per task, PCA proximity + subspace vetoes, veto-only
             tightening, zero stored raw data, growth flips checked
  fine-tune / EWC / replay(20 per class, the standard 2000 buffer) MLP heads
  NCM        nearest class mean on L2-normalized features -- the SimpleCIL-style
             baseline that dominates this regime in the literature
  joint      upper bound

Run:  python exp_cifar.py [--seeds 2]
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
from cgrow.organism import Organism
from exp_continual import run_ours, run_baseline
from cgrow.baselines import train_plain, accuracy

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FEATS = os.path.join(RES, "cifar100_r18.npz")

CFG = dict(hidden=(256,), feat=64, cap=2.5, epochs=30, p_target=0.9,
           b_hidden=(256,), b_epochs=15, prox="pca",
           q_cov_in=0.99, self_budget=0.0, veto_theta_q=0.985)


def load_feats(seed: int, val_frac: float = 0.1, feats_path: str | None = None):
    d = np.load(feats_path or FEATS)
    Xtr_all = d["Xtr"].astype(np.float64); ytr_all = d["ytr"]
    Xte = d["Xte"].astype(np.float64); yte = d["yte"]
    lo, hi = Xtr_all.min(axis=0), Xtr_all.max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    Xtr_all = np.clip((Xtr_all - lo) / span, 0.0, 1.0)
    Xte = np.clip((Xte - lo) / span, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr_all))
    n_val = int(val_frac * len(Xtr_all))
    va, tr = perm[:n_val], perm[n_val:]
    return ((Xtr_all[tr], ytr_all[tr]), (Xtr_all[va], ytr_all[va]), (Xte, yte))


def ncm_baseline(splits, tasks):
    """Nearest class mean on L2-normalized features (SimpleCIL-style)."""
    (Xtr, ytr), _, (Xte, yte) = splits
    def norm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xn, Xt = norm(Xtr), norm(Xte)
    classes = sorted(c for cl in tasks for c in cl)
    means = np.stack([Xn[ytr == c].mean(axis=0) for c in classes])
    means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-9)
    pred = np.asarray(classes)[(Xt @ means.T).argmax(axis=1)]
    return float((pred == yte).mean())


def contracted_ncm(splits, tasks, seed):
    """NCM's topology wrapped in the contract machinery: per class, an anchor
    (the mean of L2-normalized features) and a PCA territory, both computed from
    that task's data alone (order-independent, no interference). Global
    nearest-anchor routing; accept iff inside the winning class's territory and
    the cosine margin clears tau (solved on val). The weaker retention property
    of this topology: prototypes never change, but ANSWERS may drift as classes
    arrive -- the guarantee the cascade pays for is exactly what this drops."""
    from cgrow.proximity import PCAProx
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits

    def norm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    Xn, Xv, Xt = norm(Xtr), norm(Xva), norm(Xte)
    classes = sorted(c for cl in tasks for c in cl)
    means, proxes = {}, {}
    for cl in tasks:                       # per task, per class: mean + territory
        for c in cl:
            sel = ytr == c
            means[c] = Xn[sel].mean(axis=0)
            proxes[c] = PCAProx(n_components=24, seed=seed).fit(
                Xtr[sel], ytr[sel])
    M = np.stack([means[c] for c in classes])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

    def judge(Xn_, X_raw):
        sims = Xn_ @ M.T
        order = np.argsort(-sims, axis=1)
        top, second = order[:, 0], order[:, 1]
        margin = sims[np.arange(len(Xn_)), top] - sims[np.arange(len(Xn_)), second]
        pred = np.asarray(classes)[top]
        d_in = np.array([proxes[p].score(x[None])[0]
                         for p, x in zip(pred, X_raw)])
        return pred, margin, d_in

    pv, mv, dv = judge(Xv, Xva)
    correct = pv == yva
    from cgrow.organ import conformal_quantile
    rho_in = conformal_quantile(dv[correct], 0.01)
    best = (0.0, 0.0)
    for tau in np.quantile(mv, np.linspace(0.0, 0.8, 12)):
        acc_m = (mv > tau) & (dv <= rho_in)
        if acc_m.sum() and correct[acc_m].mean() >= 0.6:
            s = float((acc_m & correct).mean())
            if s > best[0]:
                best = (s, float(tau))
    tau = best[1]
    pt, mt, dt = judge(Xt, Xte)
    acc_m = (mt > tau) & (dt <= rho_in)
    return {"strict": float(((pt == yte) & acc_m).mean()),
            "forced": float((pt == yte).mean()),
            "coverage": float(acc_m.mean()),
            "precision": float((pt[acc_m] == yte[acc_m]).mean()) if acc_m.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--feats", default=FEATS, help="feature cache npz")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    ap.add_argument("--only-cncm", action="store_true",
                    help="run only the contracted-prototype arm, append to json")
    ap.add_argument("--adolescence", action="store_true",
                    help="grow the organism, then run the verified metamorphosis")
    ap.add_argument("--adol-epochs", type=int, default=30)
    ap.add_argument("--adol-feat", type=int, default=None,
                    help="feature width of the consolidated organ")
    ap.add_argument("--adol-mode", default="trunk", choices=["trunk", "protos"],
                    help="consolidate into a trained trunk or closed-form prototypes")
    ap.add_argument("--prefix", default="cifar100",
                    help="result-file prefix (cifar100 for the CIFAR caches, "
                         "ptm for the pretrained-model suite caches)")
    ap.add_argument("--inc", type=int, default=None,
                    help="classes per task; default: the cache's inc field, else 10")
    ap.add_argument("--protocol", default="10x10", choices=["10x10", "b50"],
                    help="b50: 50-class base task then 5 x 10-class increments")
    args = ap.parse_args()

    # protocol metadata: CIFAR caches carry none (100 classes, 10 per task);
    # PTM-suite caches (gpu/extract_ptm.py) store n_classes and inc.
    _meta = np.load(args.feats, allow_pickle=False)
    N_CLS = int(_meta["n_classes"]) if "n_classes" in _meta.files \
        else int(_meta["ytr"].max()) + 1
    INC = args.inc or (int(_meta["inc"]) if "inc" in _meta.files else 10)
    del _meta

    def make_tasks(seed):
        order = np.random.default_rng(seed + 400).permutation(N_CLS)
        if args.protocol == "b50":
            return [sorted(int(c) for c in order[:50])] + [
                sorted(int(c) for c in order[50 + i * 10: 50 + (i + 1) * 10])
                for i in range(5)]
        return [sorted(int(c) for c in order[i:i + INC])
                for i in range(0, N_CLS, INC)]
    horizon = f"_inc{args.inc}" if args.inc else ""      # B-horizon file suffix
    out_name = (f"{args.prefix}{('_' + args.tag) if args.tag else ''}"
                f"{'_b50' if args.protocol == 'b50' else ''}{horizon}.json")

    if args.only_cncm:
        out = os.path.join(RES, out_name)
        with open(out) as f:
            records = json.load(f)
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            splits = load_feats(seed, feats_path=args.feats)
            tasks = make_tasks(seed)
            r = contracted_ncm(splits, tasks, seed)
            rec = next(x for x in records if x["seed"] == seed)
            rec["contracted_ncm"] = r
            print(f"[{args.prefix} seed {seed}] contracted-NCM: strict {r['strict']:.3f}"
                  f"  forced {r['forced']:.3f}  cov {r['coverage']:.3f}"
                  f"  prec {r['precision']:.3f}")
        with open(out, "w") as f:
            json.dump(records, f, indent=1, default=float)
        print(f"updated {out}")
        return

    if args.adolescence:
        from cgrow.adolescence import (consolidate_mlp, consolidate_protos,
                                       metamorphosis_audit)
        consolidate = (consolidate_protos if args.adol_mode == "protos"
                       else consolidate_mlp)
        adol_cfg = {**CFG, "adol_epochs": args.adol_epochs}
        if args.adol_feat:
            adol_cfg["feat"] = args.adol_feat
        out = os.path.join(
            RES, f"{args.prefix}_{args.tag or 'r18'}_adol"
                 f"{'_protos' if args.adol_mode == 'protos' else ''}{horizon}.json")
        records = []
        if os.path.exists(out):
            # parallel single-seed runs share this file: keep the other seeds'
            # records (2026-09-14; the inc-5 horizon runs overwrote each other
            # before this, and the lifecycle files carry those seeds instead)
            mine = range(args.first_seed, args.first_seed + args.seeds)
            with open(out) as f:
                records = [r for r in json.load(f) if r["seed"] not in mine]
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            splits = load_feats(seed, feats_path=args.feats)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            tasks = make_tasks(seed)
            cache = os.path.join(
                RES, f"adol_org_{args.tag or 'r18'}_{seed}"
                     f"{'_b50' if args.protocol == 'b50' else ''}{horizon}.pkl")
            if os.path.exists(cache):
                import pickle
                with open(cache, "rb") as fh:
                    org = pickle.load(fh)
            else:
                torch.manual_seed(seed)
                org, ours = run_ours(tasks, splits, Xtr.shape[1], CFG, seed)
                import pickle
                with open(cache, "wb") as fh:
                    pickle.dump(org, fh)
            last = np.isin(yva, tasks[-1])
            new_org, rep = consolidate(org, Xtr.shape[1], adol_cfg, seed,
                                       Xva_last=Xva[last], yva_last=yva[last],
                                       n_per_organ=max(200, min(4000, 40000 // len(org.organs))))
            before = org.scores(Xte, yte)
            rec = {"seed": seed, "before": before, "adolescence": rep}
            line = (f"[{args.prefix} adol seed {seed}] ({time.time()-t0:.0f}s) "
                    f"before strict {before['strict']:.3f}")
            if rep["accepted"]:
                after = new_org.scores(Xte, yte)
                audit = metamorphosis_audit(org, new_org, Xte, yte)
                solo = Organism(); solo.organs = [new_org.organs[0]]
                rec.update({"after": after, "audit": audit,
                            "c_alone": solo.scores(Xte, yte)})
                line += (f"  after {after['strict']:.3f}  share_C "
                         f"{audit['share_answered_by_C']:.2f}  flips "
                         f"{audit['flips_to_right']}R:{audit['flips_to_wrong']}W  "
                         f"C-alone {rec['c_alone']['strict']:.3f}")
            else:
                line += f"  REJECTED ({rep.get('why')})"
            print(line)
            records.append(rec)
        with open(out, "w") as f:
            json.dump(records, f, indent=1, default=float)
        print(f"wrote {out}")
        return

    records = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.time()
        splits = load_feats(seed, feats_path=args.feats)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
        tasks = make_tasks(seed)
        in_dim = Xtr.shape[1]
        torch.manual_seed(seed)

        org, ours = run_ours(tasks, splits, in_dim, CFG, seed)
        base = {m: run_baseline(m, tasks, splits, in_dim, N_CLS, CFG, seed)
                for m in ["finetune", "ewc", "replay"]}
        joint = train_plain(Xtr, ytr, N_CLS, hidden=CFG["b_hidden"],
                            epochs=CFG["b_epochs"], seed=seed)
        joint_acc = accuracy(joint, Xte, yte)[0]
        ncm = ncm_baseline(splits, tasks)

        rec = {"seed": seed, "ours": ours,
               "baselines": {m: base[m] for m in base},
               "joint": joint_acc, "ncm": ncm}
        records.append(rec)
        f = ours["final"]
        print(f"[{args.prefix} seed {seed}] ({time.time()-t0:.0f}s)")
        print(f"  organism: strict {f['strict']:.3f}  forced {f['forced']:.3f}"
              f"  cov {f['coverage']:.3f}  prec {f['precision']:.3f}"
              f"  growth flips {ours['growth_flips']}  params {ours['n_params']}")
        print("  fa (before->after tightening): " + "  ".join(
            f"t{x['task']}:{x['fa_before']:.2f}->{x['fa_after']:.2f}"
            for x in ours["false_accepts"][1:]))
        for m in ["finetune", "ewc", "replay"]:
            print(f"  {m:8s}: final {base[m]['final_acc']:.3f}")
        print(f"  NCM     : {ncm:.3f}")
        print(f"  joint   : {joint_acc:.3f}")

    out = os.path.join(RES, out_name)
    with open(out, "w") as f:
        json.dump(records, f, indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
