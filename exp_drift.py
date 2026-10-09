"""B-drift -- the intermediate guarantee: audited prototype drift.

The guarantee ladder's middle rung. Global nearest-prototype routing (NCM
accuracy) where an answer can change ONLY at a prototype-addition event -- and
each event is audited twice:

  real       flips among previously-answered test samples (to-right / to-wrong)
  predicted  flip rate forecast BEFORE the event from contract-sampled old
             territories alone (no real old data) -- can the organism certify
             its own drift in advance?

Between events, answers are bitwise stable (nothing changes). Final accuracy is
identical to contracted prototypes by construction (the final prototype set is
order-independent); the content of this experiment is the certificate.

Run:  python exp_drift.py [--feats results/cifar100_r18.npz] [--tag r18] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.proximity import PCAProx
from exp_cifar import load_feats, FEATS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


class AuditedPrototypes:
    def __init__(self, seed: int):
        self.classes, self.means, self.proxes = [], [], {}
        self.seed = seed

    def add_task(self, Xt, yt, cls):
        for c in cls:
            sel = yt == c
            self.classes.append(c)
            self.means.append(norm(Xt[sel]).mean(axis=0))
            self.proxes[c] = PCAProx(n_components=24, seed=self.seed).fit(
                Xt[sel], yt[sel])

    def predict(self, X):
        M = norm(np.stack(self.means))
        sims = norm(X) @ M.T
        return np.asarray(self.classes)[sims.argmax(axis=1)]

    def sample_old(self, rng, n_per_class=120):
        Xs = [self.proxes[c].sample(n_per_class, rng)[0] for c in self.classes]
        return np.concatenate(Xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=FEATS)
    ap.add_argument("--tag", default="r18")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    records = []
    for seed in range(args.seeds):
        splits = load_feats(seed, feats_path=args.feats)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
        rng_order = np.random.default_rng(seed + 400)
        order = rng_order.permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
        rng = np.random.default_rng(seed * 13 + 1)

        model = AuditedPrototypes(seed)
        events = []
        pred_te = None
        for t, cls in enumerate(tasks):
            m = np.isin(ytr, cls)
            if t > 0:
                # PREDICTED drift, before the event, from contracts alone
                X_old = model.sample_old(rng)
                before_syn = model.predict(X_old)
            model.add_task(Xtr[m], ytr[m], cls)
            new_pred = model.predict(Xte)
            ev = {"task": t}
            if t > 0:
                after_syn = model.predict(X_old)
                ev["predicted_flip_rate"] = float((before_syn != after_syn).mean())
                flips = pred_te != new_pred
                ev["real_flip_rate"] = float(flips.mean())
                # the PROMISED region: test samples of classes seen before this
                # event -- the only region old contracts speak about
                seen = np.isin(yte, [c for cl in tasks[:t] for c in cl])
                ev["real_flip_rate_promised"] = float(flips[seen].mean())
                ev["to_right"] = int((flips & (new_pred == yte)).sum())
                ev["to_wrong"] = int((flips & (pred_te == yte)).sum())
            events.append(ev)
            pred_te = new_pred

        final_forced = float((pred_te == yte).mean())
        pr = [e["predicted_flip_rate"] for e in events if "real_flip_rate" in e]
        re = [e["real_flip_rate_promised"] for e in events if "real_flip_rate" in e]
        corr = float(np.corrcoef(pr, re)[0, 1]) if len(pr) > 2 else None
        ratio = float(np.mean(np.array(re) / np.maximum(np.array(pr), 1e-9)))
        rec = {"seed": seed, "final_forced": final_forced, "events": events,
               "pred_vs_real_corr_promised": corr, "real_over_pred_ratio": ratio,
               "total_to_right": sum(e.get("to_right", 0) for e in events),
               "total_to_wrong": sum(e.get("to_wrong", 0) for e in events)}
        records.append(rec)
        print(f"[drift {args.tag} seed {seed}] final forced {final_forced:.3f}  "
              f"corr(pred, real-on-promised) {corr if corr is None else round(corr, 3)}"
              f"  real/pred ratio {ratio:.2f}  "
              f"flips {rec['total_to_right']} right : {rec['total_to_wrong']} wrong")
        print("  per event (pred -> promised -> all): " + "  ".join(
            f"t{e['task']}: {e['predicted_flip_rate']:.3f}->"
            f"{e['real_flip_rate_promised']:.3f}->{e['real_flip_rate']:.3f}"
            for e in events if "real_flip_rate" in e))

    p = os.path.join(RES, f"drift_{args.tag}.json")
    with open(p, "w") as f:
        json.dump(records, f, indent=1, default=float)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
