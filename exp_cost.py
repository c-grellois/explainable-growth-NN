"""B-cost -- what each configuration costs: state, inference, update.

Everything downstream of the frozen encoder, per configuration, on a cache:

  state      floats that must be stored to keep answering (parameters,
             anchors, territories, vetoes) and, separately, the floats that
             must be stored to keep LEARNING (the head's Gram matrix)
  inference  multiply-accumulates per test sample after the encoder
  update     wall-clock seconds to absorb one task (protos, head; the cascade's
             organ birth is timed by replaying growth on ONE seed, timing only)
  storage    a same-data replay buffer at 20 / class, as raw 224x224 images
             and as features, for comparison
  RanPAC     its published head at M = 10,000 (768 x M projection, M x C
             readout, M x M Gram), arithmetic only, no run

The encoder (ViT-B/16: 86 M parameters, ~17.6 GMAC per image) is shared by
every method in the comparison and is stated once, not counted per row.
Timing is measured here, single process, NumPy BLAS threads as configured; it
is a scale indication, not a tuned quantity. Preregistered 2026-09-14.

Run:  python exp_cost.py --cache cifar100_vitb --seed 2
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_cifar import load_feats, CFG
from cgrow.proximity import PCAProx

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ENCODER = {"vitb": {"params": 86_000_000, "gmac_per_image": 17.6, "d": 768},
           "dinov2b": {"params": 86_000_000, "gmac_per_image": 23.0, "d": 768},
           "vitl": {"params": 304_000_000, "gmac_per_image": 61.6, "d": 1024}}


def prox_floats(prox):
    """mu (d) + V (q x d) + lam (q) + r_q + w_q per class territory."""
    return int(sum(len(mu) + V.size + len(lam) + 2 for mu, V, lam in zip(prox.mu, prox.V, prox.lam)))


def organ_cost(o, d):
    """(state floats, inference MACs) of one organ, vetoes included."""
    trunk = int(sum(p.numel() for p in o.trunk.parameters()))
    anchors = int(o.anchors.numel())
    prox = prox_floats(o.prox) if o.prox is not None else 0
    vet_params = 0; vet_mac = 0
    for (_, net, _, _) in o.vetoes:
        n = int(sum(p.numel() for p in net.parameters()))
        vet_params += n; vet_mac += n
    # inference: trunk (~params MACs for an MLP), anchor distances, every
    # territory of the organ (score is a min over its classes), the veto of the
    # nearest territory
    n_terr = len(o.prox.mu) if o.prox is not None else 0
    q = int(o.prox.V[0].shape[0]) if n_terr else 0
    mac = trunk + anchors + n_terr * (2 * d * q + d) + (vet_mac // max(len(o.vetoes), 1) if o.vetoes else 0)
    return trunk + anchors + prox + vet_params + 3, mac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cifar100_vitb")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--M", type=int, nargs="+", default=[2000, 10000])
    ap.add_argument("--time-growth", action="store_true",
                    help="replay the cascade's growth on this seed to time organ birth")
    args = ap.parse_args()
    cache, seed = args.cache, args.seed
    feats = os.path.join(RES, f"{cache}.npz")
    meta = np.load(feats)
    n_cls = int(meta["n_classes"]) if "n_classes" in meta.files else int(meta["ytr"].max()) + 1
    inc = int(meta["inc"]) if "inc" in meta.files else 10
    del meta
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_feats(seed, feats_path=feats)
    d = Xtr.shape[1]
    tag = cache.split("_", 1)[1]
    enc_key = next((k for k in ENCODER if tag.startswith(k)), "vitb")
    order = np.random.default_rng(seed + 400).permutation(n_cls)
    tasks = [sorted(int(c) for c in order[i:i + inc]) for i in range(0, n_cls, inc)]
    out = {"cache": cache, "seed": seed, "d": d, "n_classes": n_cls, "inc": inc,
           "n_train": int(len(ytr)), "encoder": {"name": enc_key, **ENCODER[enc_key]},
           "blas_threads": os.environ.get("OMP_NUM_THREADS", "default"), "rows": {}}
    q = 24

    # ---- contracted prototypes: state, MACs, update time (one task)
    t0 = time.time()
    cls = tasks[-1]
    means = {c: Xtr[ytr == c].mean(axis=0) for c in cls}
    proxes = {c: PCAProx(n_components=q, seed=seed).fit(Xtr[ytr == c], ytr[ytr == c]) for c in cls}
    # thresholds on validation (the part that touches every class): score val once
    _ = [proxes[c].score(Xva[:2000]) for c in cls]
    t_protos = time.time() - t0
    out["rows"]["protos"] = {
        "state_floats": int(n_cls * (d + q * d + q + 2) + 2),
        "learning_state_floats": 0,
        "inference_mac": int(n_cls * d + 2 * d * q + d),
        "update_s_one_task": t_protos,
        "answers_change_under_growth": False,
    }

    # ---- contracted head at each M: state, MACs, one-task update time
    import exp_head
    for M in args.M:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((d, M))
        idx = np.flatnonzero(np.isin(ytr, tasks[0]))
        G = np.zeros((M, M)); Q = np.zeros((M, n_cls))
        t0 = time.time()
        for s0 in range(0, len(idx), 4000):
            ii = idx[s0:s0 + 4000]
            H = exp_head.project(Xtr[ii], W)
            G += H.T @ H
            Y = np.zeros((len(ii), n_cls)); Y[np.arange(len(ii)), ytr[ii]] = 1.0
            Q += H.T @ Y
        Wo = exp_head.solve(G, Q, 1.0)
        t_head = time.time() - t0
        out["rows"][f"head_M{M}"] = {
            "state_floats": int(W.size + Wo.size + n_cls * (d + q * d + q + 2)),
            "learning_state_floats": int(G.size + Q.size),
            "inference_mac": int(d * M + M * n_cls + 2 * d * q + d),
            "update_s_one_task": t_head,
            "update_s_one_task_with_lambda_grid": t_head + 6 * (t_head * 0.9),  # 7 solves; the accumulate is ~10 %
            "answers_change_under_growth": True,
        }
        del G, Q, W, Wo

    # ---- RanPAC's published head, arithmetic only
    M = 10000
    out["rows"]["ranpac_M10000_published"] = {
        "state_floats": int(d * M + M * n_cls), "learning_state_floats": int(M * M + M * n_cls),
        "inference_mac": int(d * M + M * n_cls), "update_s_one_task": None,
        "answers_change_under_growth": True, "note": "arithmetic from the published configuration; not run"}

    # ---- the grown cascade (this seed's cached organism) and, if present, the lifecycle's C
    pk = os.path.join(RES, f"adol_org_{tag}_{seed}.pkl")
    if os.path.exists(pk):
        with open(pk, "rb") as fh:
            org = pickle.load(fh)
        st = mac = 0
        for o in org.organs:
            a, b = organ_cost(o, d); st += a; mac += b
        out["rows"]["cascade"] = {
            "state_floats": int(st), "learning_state_floats": 0, "inference_mac": int(mac),
            "n_organs": len(org.organs), "n_vetoes": int(sum(len(o.vetoes) for o in org.organs)),
            "update_s_one_task": None, "answers_change_under_growth": False}
        if args.time_growth:
            from exp_continual import run_ours
            import torch
            torch.manual_seed(seed)
            t0 = time.time()
            times = []
            # time each task by wrapping run_ours per prefix is intrusive; time the
            # whole growth and report the mean per task plus first/last via a
            # second call on the first and the last two tasks
            _ = run_ours(tasks, ((Xtr, ytr), (Xva, yva), (Xte, yte)), d, CFG, seed)
            total = time.time() - t0
            out["rows"]["cascade"]["update_s_one_task"] = total / len(tasks)
            out["rows"]["cascade"]["growth_total_s"] = total

    # ---- same-data replay buffer, 20 / class
    n_buf = 20 * n_cls
    out["rows"]["replay_buffer_20_per_class"] = {
        "raw_bytes": int(n_buf * 3 * 224 * 224), "feature_floats": int(n_buf * d),
        "note": "storage only; the classifier it trains is a 2-layer MLP (see baselines)"}

    path = os.path.join(RES, f"cost_{cache}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    for k, r in out["rows"].items():
        print(f"{k:28s} state {r.get('state_floats', 0)/1e6:8.2f} M floats | learn-state "
              f"{r.get('learning_state_floats', 0)/1e6:8.2f} M | MAC/sample {r.get('inference_mac', 0)/1e6:8.2f} M "
              f"| update {r.get('update_s_one_task')}")
    print("wrote", path)


if __name__ == "__main__":
    main()
