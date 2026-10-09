"""Measurement: learned tightening (veto) from contract-sampled positives.

When task B arrives, an old organ serving classes A falsely accepts some of B.
Scalar tightening moves three knobs; a VETO is a small capped discriminator
trained on
    positives: synthetic samples from the organ's own contract (prox.sample)
    negatives: the real incoming samples the organ falsely accepts
and conjoined with the accept rule (accept only if the veto passes), so the
accepted set can only shrink -- a tightening event with a learned shape.

This experiment measures, per (dataset, organ-task A, incoming-task B):
  exclusion   fraction of B's falsely-accepted val samples the veto rejects
  self-loss   fraction of A's real val samples the veto wrongly rejects
              (in deployment only synthetic positives exist; the gap between
              the synthetic estimate and this real number is the calibration
              gap, reported explicitly)
  transfer    exclusion on task C (classes never seen by the veto) -- does a
              veto against B also exclude later classes?

Run:  python exp_veto.py [--datasets mnist fashion] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import load_real, prepare, split_tasks
from cgrow.organ import SpectralMLP, DTYPE
from cgrow.proximity import make_prox

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)

THETA_Q = 0.97      # veto threshold: quantile of veto score on synthetic positives


def train_veto(X_pos, X_neg, seed=0, hidden=128, cap=4.0, epochs=12):
    net = SpectralMLP([X_pos.shape[1], hidden, 1], cap=cap, seed=seed)
    Xall = torch.as_tensor(np.concatenate([X_pos, X_neg]), dtype=DTYPE)
    lab = torch.cat([torch.zeros(len(X_pos)), torch.ones(len(X_neg))])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xall), generator=g)
        for s in range(0, len(Xall), 256):
            idx = perm[s:s + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(Xall[idx]).squeeze(1), lab[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            net.project()
    return net


@torch.no_grad()
def vscore(net, X):
    return torch.sigmoid(net(torch.as_tensor(X, dtype=DTYPE))).numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--prox", default="pca", choices=["pca", "kmeans"])
    args = ap.parse_args()

    out = {}
    for ds in args.datasets:
        X, y, C = load_real(ds)
        rows = []
        for seed in range(args.seeds):
            (Xtr, ytr), (Xva, yva), _ = prepare(X, y, seed=seed)
            tasks = split_tasks(y, 2)
            rng = np.random.default_rng(seed)
            for a in range(len(tasks) - 1):
                A = tasks[a]
                own_tr = np.isin(ytr, A)
                prox = make_prox("pca", seed=seed).fit(Xtr[own_tr], ytr[own_tr])

                def proj_nearest(Xq):
                    """Project each sample into the coordinates of its NEAREST
                    class territory (the class whose veto would judge it)."""
                    best = np.full(len(Xq), np.inf)
                    ci = np.zeros(len(Xq), dtype=int)
                    for j, (mu, V, lam, rq, wq) in enumerate(zip(
                            prox.mu, prox.V, prox.lam, prox.r_q, prox.w_q)):
                        Z = Xq - mu
                        p = Z @ V.T
                        s = np.maximum(np.linalg.norm(Z - p @ V, axis=1) / rq,
                                       np.sqrt(((p ** 2) / lam).sum(axis=1)) / wq)
                        take = s < best
                        best[take] = s[take]; ci[take] = j
                    P = np.stack([(Xq[i] - prox.mu[ci[i]]) @ prox.V[ci[i]].T
                                  for i in range(len(Xq))])
                    # whiten so the veto sees N(0, I)-like positives
                    return P / np.sqrt(np.stack([prox.lam[c] for c in ci])), ci

                for b in range(a + 1, len(tasks)):
                    B = tasks[b]
                    Xb_tr = Xtr[np.isin(ytr, B)]
                    # per-class vetoes, all judged in whitened subspace coords
                    Pb, cb = proj_nearest(Xb_tr)
                    vetoes, thetas = [], []
                    q = prox.V[0].shape[0]
                    for j in range(len(prox.classes)):
                        Z_syn = rng.normal(0, 1.0, (3000, q))     # exact spec model
                        neg = Pb[cb == j]
                        if len(neg) < 10:
                            vetoes.append(None); thetas.append(None); continue
                        v = train_veto(Z_syn, neg, seed=seed * 10 + a + 7 * j)
                        vetoes.append(v)
                        thetas.append(np.quantile(vscore(v, Z_syn), THETA_Q))

                    def excluded(Xq):
                        P, ci = proj_nearest(Xq)
                        outm = np.zeros(len(Xq), bool)
                        for j, (v, th) in enumerate(zip(vetoes, thetas)):
                            if v is None:
                                continue
                            sel = ci == j
                            if sel.any():
                                outm[sel] = vscore(v, P[sel]) > th
                        return outm

                    excl_B = float(excluded(Xva[np.isin(yva, B)]).mean())
                    self_real = float(excluded(Xva[np.isin(yva, A)]).mean())
                    transfer = None
                    if b + 1 < len(tasks):
                        transfer = float(
                            excluded(Xva[np.isin(yva, tasks[b + 1])]).mean())
                    rows.append({"seed": seed, "A": A, "B": B,
                                 "excl_B": excl_B, "self_loss_real": self_real,
                                 "self_loss_syn": 1.0 - THETA_Q,
                                 "transfer_next": transfer})
        e = [r["excl_B"] for r in rows]
        sl = [r["self_loss_real"] for r in rows]
        tr = [r["transfer_next"] for r in rows if r["transfer_next"] is not None]
        out[ds] = {"rows": rows,
                   "exclusion": {"mean": float(np.mean(e)), "sd": float(np.std(e))},
                   "self_loss_real": {"mean": float(np.mean(sl)), "sd": float(np.std(sl))},
                   "transfer_next": {"mean": float(np.mean(tr)), "sd": float(np.std(tr))}}
        print(f"[{ds}] veto ({args.prox} sampler): exclusion of B "
              f"{np.mean(e):.3f} +- {np.std(e):.3f}   real self-loss "
              f"{np.mean(sl):.3f} +- {np.std(sl):.3f}   transfer to next task "
              f"{np.mean(tr):.3f}")

    with open(os.path.join(RES, f"veto_{args.prox}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {os.path.join(RES, f'veto_{args.prox}.json')}")


if __name__ == "__main__":
    main()
