"""B-pfc -- the full lifecycle: growth -> adolescence -> prefrontal arbiter ->
ultimate consolidation.

Biology's ordering (sensory consolidates first, executive control matures last,
operating plastically over frozen parts before consolidating itself -- Gogtay
2004) mapped onto the organism:

  phase 1  growth: one organ per task, frozen at birth (cached from B-vit)
  phase 2  adolescence: verified consolidation into prototypes (veto-capable
           merged territory)
  phase 3  PFC: an ARBITER -- a small net that reads every organ's evidence
           (distance, margin, territory score, accept bit) and learns WHO TO
           TRUST, trained purely on contract-sampled data (labels known, each
           organ's correctness computable). It answers nothing itself; it
           routes. Its errors are routing errors -- exactly the measured gap
           between the cascade and the prototype regime.
  phase 4  ultimate consolidation: the arbiter faces the same verify-then-
           freeze gate as adolescence (relative per-class gate on the synthetic
           holdout + last-task real val), its answer changes vs the
           pre-arbiter organism are audited (a single logged event), and the
           whole organism returns to full strictness.

Run:  python exp_pfc.py [--seeds 2]
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
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import (consolidate_protos, sample_pool,
                               _strict_per_class)
from exp_cifar import load_feats, CFG

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FEATS = os.path.join(RES, "cifar100_vitb.npz")
BIG = 100.0


def evidence(org: Organism, X: np.ndarray):
    """Per-organ evidence matrix (n, 4K) and forced predictions (K, n)."""
    feats, preds = [], []
    for o in org.organs:
        d1, m, p, d_in, vet, _ = o.evaluate(X)
        acc = o.accept(d1, m, d_in, vet).astype(np.float64)
        feats += [d1, np.minimum(m, BIG), d_in, acc]
        preds.append(p)
    return np.stack(feats, axis=1), np.stack(preds)


class Arbiter(nn.Module):
    def __init__(self, in_dim, k, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                 nn.Linear(64, k))

    def forward(self, x):
        return self.net(x)


def train_arbiter(F, correct, seed, epochs=25):
    net = Arbiter(F.shape[1], correct.shape[0], seed=seed)
    Ft = torch.as_tensor(F, dtype=torch.float32)
    Yt = torch.as_tensor(correct.T.astype(np.float32))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Ft), generator=g)
        for s in range(0, len(Ft), 256):
            idx = perm[s:s + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(Ft[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def arbiter_predict(net, F, preds, theta=None):
    p = torch.sigmoid(net(torch.as_tensor(F, dtype=torch.float32))).numpy()
    who = p.argmax(axis=1)
    conf = p.max(axis=1)
    ans = preds[who, np.arange(preds.shape[1])]
    if theta is not None:
        ans = np.where(conf >= theta, ans, ABSTAIN)
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    records = []
    for seed in range(args.seeds):
        t0 = time.time()
        splits = load_feats(seed, feats_path=FEATS)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
        with open(os.path.join(RES, f"adol_org_vitb_{seed}.pkl"), "rb") as fh:
            org = pickle.load(fh)
        s_grow = org.scores(Xte, yte)

        # phase 2: adolescence (veto-capable merged territory)
        last = np.isin(yva, tasks[-1])
        org2, rep = consolidate_protos(org, Xtr.shape[1], CFG, seed,
                                       Xva_last=Xva[last], yva_last=yva[last])
        s_adol = org2.scores(Xte, yte)

        # phase 3: the PFC arbiter, trained on contract-sampled data only
        rng = np.random.default_rng(seed * 41 + 9)
        X_syn, y_syn = sample_pool(org, 3500, rng)     # from the original organs
        perm = rng.permutation(len(X_syn))
        ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
        F_tr, P_tr = evidence(org2, X_syn[tr])
        correct_tr = P_tr == y_syn[tr][None, :]
        net = train_arbiter(F_tr, correct_tr, seed)

        # solve the abstention threshold on the synthetic holdout
        F_ho, P_ho = evidence(org2, X_syn[ho])
        best = (0.0, None)
        for theta in np.linspace(0.3, 0.9, 13):
            a = arbiter_predict(net, F_ho, P_ho, theta)
            s = float((a == y_syn[ho]).mean())
            if s > best[0]:
                best = (s, float(theta))
        theta = best[1]

        # phase 4: ultimate consolidation -- verify, audit, freeze
        a_ho = arbiter_predict(net, F_ho, P_ho, theta)
        classes = sorted(set(int(c) for c in y_syn))
        pc_new = _strict_per_class(a_ho, a_ho != ABSTAIN, y_syn[ho],
                                   np.asarray(classes))
        r_ref = org2.predict(X_syn[ho])
        pc_ref = _strict_per_class(r_ref["pred"], r_ref["pred"] != ABSTAIN,
                                   y_syn[ho], np.asarray(classes))
        gate = {"new_mean": float(pc_new.mean()), "ref_mean": float(pc_ref.mean()),
                "new_floor": float(pc_new.min()), "ref_floor": float(pc_ref.min())}
        accepted = (pc_new.mean() >= pc_ref.mean() - 0.02
                    and pc_new.min() >= pc_ref.min() - 0.05)
        # last-task real val check
        F_v, P_v = evidence(org2, Xva[last])
        val_new = float((arbiter_predict(net, F_v, P_v, theta) == yva[last]).mean())
        val_ref = org2.scores(Xva[last], yva[last])["strict"]
        gate.update({"val_new": val_new, "val_ref": val_ref})
        accepted = accepted and (val_new >= val_ref - 0.05)

        rec = {"seed": seed, "growth": s_grow, "adolescence": s_adol,
               "adol_accepted": rep["accepted"], "gate": gate,
               "pfc_accepted": accepted, "theta": theta}
        if accepted:
            F_te, P_te = evidence(org2, Xte)
            ans = arbiter_predict(net, F_te, P_te, theta)
            forced = arbiter_predict(net, F_te, P_te, None)
            pre = org2.predict(Xte)["pred"]
            was = pre != ABSTAIN
            now = ans != ABSTAIN
            flips = was & now & (ans != pre)
            rec["final"] = {
                "strict": float((ans == yte).mean()),
                "forced": float((forced == yte).mean()),
                "coverage": float(now.mean()),
                "precision": float((ans[now] == yte[now]).mean()) if now.any() else 0.0}
            rec["audit"] = {
                "flips": int(flips.sum()),
                "to_right": int((flips & (ans == yte)).sum()),
                "to_wrong": int((flips & (pre == yte)).sum())}
        records.append(rec)
        print(f"[pfc seed {seed}] ({time.time()-t0:.0f}s) "
              f"growth {s_grow['strict']:.3f} -> adolescence {s_adol['strict']:.3f}"
              + (f" -> PFC {rec['final']['strict']:.3f} (forced "
                 f"{rec['final']['forced']:.3f}, cov {rec['final']['coverage']:.3f})"
                 if accepted else "  PFC REJECTED"))
        if accepted:
            au = rec["audit"]
            print(f"  ultimate consolidation: gate mean {gate['new_mean']:.3f} vs "
                  f"ref {gate['ref_mean']:.3f}  val {gate['val_new']:.3f} vs "
                  f"{gate['val_ref']:.3f}  |  logged flips {au['flips']} "
                  f"({au['to_right']}R:{au['to_wrong']}W)")

    p = os.path.join(RES, "pfc.json")
    with open(p, "w") as f:
        json.dump(records, f, indent=1, default=float)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
