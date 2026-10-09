"""B-head -- the contracted head: a RanPAC-style random-projection ridge readout
whose answers carry a ball contract.

RanPAC's own ablations put our contracted prototypes at its "NCM only" level and
show that its head alone -- a random projection of the frozen features to M
dimensions, a ReLU, and a ridge classifier solved in closed form from
accumulated second-order statistics -- buys +3.6 to +10.6 points on frozen
features (the "No Phase 1" row). This experiment asks whether that head can be
put under a contract without losing the gain.

Head (order-invariant, exemplar-free, never trained by gradient):
  h = relu(x_n W),  W ~ N(0, 1) fixed by seed, x_n = x / ||x||
  G += h^T h,  Q += h^T onehot(y)   accumulated task by task
  W_out = (G + lambda I)^-1 Q       lambda chosen on the validation split
Contract (same machinery as the contracted prototypes):
  territory  per-class PCA territory in the FEATURE space (24 components,
             fitted from that class's own data at its arrival, never updated)
  margin     top-1 minus top-2 head score must clear tau (solved on validation)
  accept iff inside the predicted class's territory (conformal radius) and
             margin > tau; otherwise abstain
Reported per seed: forced (plain head accuracy, comparable to RanPAC's
"No Phase 1" row on the same backbone), strict (abstention = error), coverage,
precision, and -- the property this topology does NOT have -- answer flips:
test answers on already-seen classes that changed when a later task arrived
(the ridge weights are re-solved after every task; the cascade's count is 0).

Run:  python exp_head.py --cache cifar100_vitb [--M 2000 10000] [--first-seed 2 --seeds 3]
      python exp_head.py --cache ptm_imr_vitb ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_cifar import load_feats
from cgrow.organ import conformal_quantile
from cgrow.proximity import PCAProx

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def project(X, W):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return np.maximum(Xn @ W, 0.0)


def solve(G, Q, lam):
    M = G.shape[0]
    return np.linalg.solve(G + lam * np.eye(M, dtype=G.dtype), Q)


def run(cache, M, seed, tasks_override=None, inc_override=None, return_model=False):
    feats = os.path.join(RES, f"{cache}.npz")
    meta = np.load(feats, allow_pickle=False)
    n_cls = int(meta["n_classes"]) if "n_classes" in meta.files else int(meta["ytr"].max()) + 1
    inc = int(meta["inc"]) if "inc" in meta.files else 10
    if inc_override:                      # B-horizon: a longer stream, same order
        inc = int(inc_override)
    del meta
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_feats(seed, feats_path=feats)
    order = np.random.default_rng(seed + 400).permutation(n_cls)
    tasks = [sorted(int(c) for c in order[i:i + inc]) for i in range(0, n_cls, inc)]
    d = Xtr.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, M)).astype(np.float64)
    # training projections are never held whole: G and Q are accumulated in
    # chunks (OmniBenchmark: 90k x 10k would be 7 GB); val/test projections are
    # kept for scoring
    Hva, Hte = project(Xva, W), project(Xte, W)
    G = np.zeros((M, M)); Q = np.zeros((M, n_cls))
    proxes = {}
    prev_pred = None
    flips = 0
    curve = []
    t0 = time.time()
    # the lambda grid needs |LAMBDAS| solves of an M x M system per task; for a
    # wide head that is the whole cost, so the grid runs only at the last task
    # and intermediate tasks (which serve the flip count) use lambda = 1
    grid_every_task = M <= 4000
    for t, cls in enumerate(tasks):
        idx = np.flatnonzero(np.isin(ytr, cls))
        for s0 in range(0, len(idx), 4000):
            ii = idx[s0:s0 + 4000]
            H = project(Xtr[ii], W)
            G += H.T @ H
            Y = np.zeros((len(ii), n_cls)); Y[np.arange(len(ii)), ytr[ii]] = 1.0
            Q += H.T @ Y
        for c in cls:                       # territory from this class's data, once
            sel = ytr == c
            proxes[c] = PCAProx(n_components=24, seed=seed).fit(Xtr[sel], ytr[sel])
        seen = sorted(c for cl in tasks[:t + 1] for c in cl)
        # lambda on validation of seen classes (forced accuracy)
        mv = np.isin(yva, seen)
        best = (-1, None, None)
        grid = LAMBDAS if (grid_every_task or t == len(tasks) - 1) else [1.0]
        for lam in grid:
            Wo = solve(G, Q, lam)
            acc = float((np.asarray(seen)[(Hva[mv] @ Wo[:, seen]).argmax(1)] == yva[mv]).mean())
            if acc > best[0]:
                best = (acc, lam, Wo)
        _, lam, Wo = best
        mt = np.isin(yte, seen)
        S = Hte[mt] @ Wo[:, seen]
        pred = np.asarray(seen)[S.argmax(1)]
        if prev_pred is not None:
            # answers on classes seen before this task
            prev_mask = np.isin(yte[mt], prev_seen)
            # align: prev_pred was over yte[prev_mt]; recompute on same points
            flips += int((pred[prev_mask] != prev_pred_full[mt][prev_mask]).sum())
        prev_pred_full = np.full(len(yte), -1); prev_pred_full[mt] = pred
        prev_pred = pred; prev_seen = seen
        curve.append({"task": t, "lambda": lam, "forced_seen": float((pred == yte[mt]).mean())})
    # ---- final: contract on top of the head -------------------------------
    seen = sorted(c for cl in tasks for c in cl)
    Sv = Hva @ Wo[:, seen]; St = Hte @ Wo[:, seen]
    def judge(S, X):
        o = np.argsort(-S, axis=1)
        top, second = o[:, 0], o[:, 1]
        margin = S[np.arange(len(S)), top] - S[np.arange(len(S)), second]
        pred = np.asarray(seen)[top]
        d_in = np.array([proxes[p].score(x[None])[0] for p, x in zip(pred, X)])
        return pred, margin, d_in
    pv, mv_, dv = judge(Sv, Xva)
    correct = pv == yva
    rho_in = conformal_quantile(dv[correct], 0.01)
    best = (0.0, 0.0)
    for tau in np.quantile(mv_, np.linspace(0.0, 0.8, 12)):
        acc_m = (mv_ > tau) & (dv <= rho_in)
        if acc_m.sum() and correct[acc_m].mean() >= 0.6:
            s = float((acc_m & correct).mean())
            if s > best[0]:
                best = (s, float(tau))
    tau = best[1]
    pt, mt_, dt = judge(St, Xte)
    acc_m = (mt_ > tau) & (dt <= rho_in)
    strict = float((acc_m & (pt == yte)).mean())
    forced = float((pt == yte).mean())
    cov = float(acc_m.mean())
    prec = float((pt[acc_m] == yte[acc_m]).mean()) if acc_m.any() else 0.0
    # margin-only contract (no territory) for reference
    acc_mo = mt_ > tau
    strict_margin_only = float((acc_mo & (pt == yte)).mean())
    rec = {"seed": seed, "cache": cache, "M": M, "n_classes": n_cls, "inc": inc,
            "lambda": lam, "forced": forced, "strict": strict, "coverage": cov,
            "precision": prec, "tau": tau, "rho_in": float(rho_in),
            "strict_margin_only": strict_margin_only,
            "answer_flips_under_growth": int(flips), "n_test": int(len(yte)),
            "head_params": int(W.size + Wo.size), "gram_floats": int(G.size),
            "curve": curve, "elapsed_s": time.time() - t0}
    if return_model:            # not JSON-serialisable; consumers pop it
        rec["_model"] = {"W": W, "Wo": Wo, "seen": seen, "proxes": proxes,
                         "tau": tau, "rho_in": float(rho_in), "lambda": lam}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cifar100_vitb")
    ap.add_argument("--M", type=int, nargs="+", default=[2000, 10000])
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--inc", type=int, default=None,
                    help="classes per task; default the cache's inc field. "
                         "B-horizon: writes head_<cache>_inc<k>.json")
    args = ap.parse_args()
    suffix = f"_inc{args.inc}" if args.inc else ""
    out = os.path.join(RES, args.out_name or f"head_{args.cache}{suffix}.json")
    records = []
    for M in args.M:
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            r = run(args.cache, M, seed, inc_override=args.inc)
            records.append(r)
            print(f"[head {args.cache} M={M} seed {seed}] ({r['elapsed_s']:.0f}s) forced {r['forced']:.3f} "
                  f"| contracted strict {r['strict']:.3f} cov {r['coverage']:.3f} prec {r['precision']:.3f} "
                  f"(margin-only strict {r['strict_margin_only']:.3f}) | lambda {r['lambda']} "
                  f"| answer flips under growth {r['answer_flips_under_growth']}/{r['n_test']}", flush=True)
            with open(out, "w") as f:
                json.dump(records, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
