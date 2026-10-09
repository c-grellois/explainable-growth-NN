"""B-sampler -- how faithful is the contract's data generator?

The gaussian-in-subspace sampler is the measured binding constraint on
adolescence-at-scale and on drift-forecast calibration. This experiment scores
samplers directly, per task, before any consumer is rerun:

  disc AUROC   a small classifier trained real-vs-synthetic, scored on held-out
               real and fresh synthetic. 0.5 = indistinguishable (perfect
               sampler); 1.0 = the vacuity regime that sank raw-pixel vetoes.
  downstream   a classifier trained ONLY on labeled synthetic data, evaluated
               on real held-out data of the same classes, against the same
               classifier trained on real data -- the adolescence sub-problem.

Compared: pca (single gaussian per class, the incumbent) vs mix (diagonal GMM
in whitened subspace coords, k<=5 per class).

Run:  python exp_sampler.py [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import load_real, prepare, split_tasks
from cgrow.proximity import make_prox
from exp_cifar import load_feats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def get_setup(name, seed):
    if name in ("mnist", "fashion"):
        X, y, C = load_real(name)
        (Xtr, ytr), (Xva, yva), _ = prepare(X, y, seed=seed)
        tasks = split_tasks(y, 2)[:3]
    else:  # cifar_vitb
        (Xtr, ytr), (Xva, yva), _ = load_feats(
            seed, feats_path=os.path.join(RES, "cifar100_vitb.npz"))
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(3)]
    return (Xtr, ytr), (Xva, yva), tasks


def disc_auroc(X_real_tr, X_syn_tr, X_real_te, X_syn_te, seed):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200,
                        random_state=seed)
    Xt = np.concatenate([X_real_tr, X_syn_tr])
    yt = np.concatenate([np.zeros(len(X_real_tr)), np.ones(len(X_syn_tr))])
    clf.fit(Xt, yt)
    p = clf.predict_proba(np.concatenate([X_real_te, X_syn_te]))[:, 1]
    lab = np.concatenate([np.zeros(len(X_real_te)), np.ones(len(X_syn_te))])
    r = p.argsort().argsort().astype(np.float64)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n0 * n1))


def downstream_acc(X_syn, y_syn, X_real_tr, y_real_tr, Xva, yva, cls, seed):
    from sklearn.neural_network import MLPClassifier
    m = np.isin(yva, cls)
    out = {}
    for tag, (Xs, ys) in [("synthetic", (X_syn, y_syn)),
                          ("real", (X_real_tr, y_real_tr))]:
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=250,
                            random_state=seed)
        clf.fit(Xs, ys)
        out[tag] = float((clf.predict(Xva[m]) == yva[m]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["mnist", "fashion", "cifar_vitb"])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    KINDS = {"pca": ("pca", {}), "mix": ("mix", {}),
             "mix48": ("mix", {"n_components": 48})}
    out = {}
    for ds in args.datasets:
        rows = {k: {"auroc": [], "down_syn": [], "down_real": []}
                for k in KINDS}
        for seed in range(args.seeds):
            (Xtr, ytr), (Xva, yva), tasks = get_setup(ds, seed)
            rng = np.random.default_rng(seed * 17 + 3)
            pools = {k: ([], []) for k in KINDS}
            for cls in tasks:
                m_tr = np.isin(ytr, cls)
                m_va = np.isin(yva, cls)
                Xt, yt = Xtr[m_tr], ytr[m_tr]
                for label, (kind, kw) in KINDS.items():
                    prox = make_prox(kind, seed=seed, **kw).fit(Xt, yt)
                    X_syn, y_syn = prox.sample(len(Xt), rng)
                    rows[label]["auroc"].append(disc_auroc(
                        Xt[: len(X_syn) // 2], X_syn[: len(X_syn) // 2],
                        Xva[m_va], X_syn[len(X_syn) // 2:
                                         len(X_syn) // 2 + int(m_va.sum())],
                        seed))
                    pools[label][0].append(X_syn)
                    pools[label][1].append(y_syn)
            # the CONSOLIDATION probe: train on the pooled synthetic data from
            # every task's contract, evaluate on real val of all pooled classes
            all_cls = [c for cl in tasks for c in cl]
            m_all = np.isin(ytr, all_cls)
            for label in KINDS:
                X_syn = np.concatenate(pools[label][0])
                y_syn = np.concatenate(pools[label][1])
                d = downstream_acc(X_syn, y_syn, Xtr[m_all], ytr[m_all],
                                   Xva, yva, all_cls, seed)
                rows[label]["down_syn"].append(d["synthetic"])
                rows[label]["down_real"].append(d["real"])
        out[ds] = {k: {m: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                       for m, v in r.items()} for k, r in rows.items()}
        print(f"[{ds}]")
        for label in KINDS:
            r = out[ds][label]
            print(f"  {label:6s}: disc AUROC {r['auroc']['mean']:.3f} "
                  f"+- {r['auroc']['sd']:.3f}   consolidation synth "
                  f"{r['down_syn']['mean']:.3f} vs real {r['down_real']['mean']:.3f}"
                  f"   (gap {r['down_real']['mean'] - r['down_syn']['mean']:+.3f})")

    p = os.path.join(RES, "sampler.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
