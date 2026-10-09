"""B-sheet -- the organism on ASDbrain's spatial substrate, benchmarked.

Class-incremental learning (4 tasks x 2 classes, single head, no task oracle)
on ASDbrain's own stream generators:

  sweeps  temporal spectrogram sequences (B, T, 16) -- the recurrent substrate
          earns its keep
  shapes  static 12x12 patterns driven for all T steps

Ours: one SheetOrgan per task -- a spatially placed recurrent population wired
under the hinge wiring cost, listening (directionally) to frozen predecessors,
pruned cost-weightedly, behind the same PCA-proximity + subspace-veto contract
as the MLP organism. Baselines: a monolithic recurrent sheet of the same total
unit count, trained sequentially (fine-tune / EWC / replay) and jointly.

Beyond accuracy, the spatial port measures what only anatomy can offer:
  * modularity Q of the pooled wiring under the organ partition, and mean
    (weight-weighted) edge length vs the random-pairing control;
  * a lesion matrix: silence organ k, measure per-task damage -- selectivity
    on the diagonal, and the off-diagonal cost of listening (forward transfer
    priced in fragility), via a listen-on/off ablation.

Run:  python exp_sheet.py [--streams sweeps shapes] [--seeds 2]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ASDbrain supplies the sheet stream generators. It is looked for next to this
# repository (../ASDbrain) unless ASDBRAIN_DIR says otherwise; when it is absent
# (e.g. the GPU box, which runs only the feature-cache benchmarks) the import is
# deferred so exp_lifecycle still loads, and the sheet benchmarks fail with a
# clear message only if actually requested.
sys.path.insert(0, os.environ.get(
    "ASDBRAIN_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ASDbrain")))
try:
    from growth_by_need.streams import make_shapes, make_sweeps
except ImportError:                       # pragma: no cover
    def _missing(*a, **k):
        raise ImportError("sheet benchmarks need ASDbrain (set ASDBRAIN_DIR)")
    make_shapes = make_sweeps = _missing

from cgrow.sheet_organ import SheetOrgan, place_center
from cgrow.organism import Organism, ABSTAIN
from cgrow.baselines import _epochs, _fisher_diag

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)

T = 8
N_UNITS = 40          # per organ; the monolith gets 4x this
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7]]
CFG = {
    "sweeps": dict(static=False, epochs=60, p_target=0.95, veto_theta_q=0.985,
                   q_cov_in=0.99, b_epochs=40),
    "shapes": dict(static=True, epochs=60, p_target=0.95, veto_theta_q=0.985,
                   q_cov_in=0.99, b_epochs=40),
    # real audio: FSDD spoken digits as 8x16 log-spectrograms (build_fsdd.py)
    "fsdd": dict(static=False, epochs=120, p_target=0.85, veto_theta_q=0.985,
                 q_cov_in=0.99, b_epochs=80),
}


def make_data(kind: str, seed: int, n: int = 11000):
    if kind == "fsdd_spk":
        # SPEAKER-DISJOINT split: one held-out speaker for test, one for
        # validation, the remaining four for training. The default random
        # split shares speakers across splits, so it can reward speaker
        # identity rather than the spoken digit; this is the honest version.
        d = np.load(os.path.join(RES, "fsdd_spec.npz"))
        X, y, spk = d["X"].astype(np.float64), d["y"], d["speakers"]
        names = np.array(sorted(set(spk.tolist())))
        rng = np.random.default_rng(seed * 1000 + 7)
        names = names[rng.permutation(len(names))]
        te, va = spk == names[0], spk == names[1]
        tr = ~(te | va)
        return ((X[tr], y[tr]), (X[va], y[va]), (X[te], y[te]), 16)
    if kind == "fsdd":
        d = np.load(os.path.join(RES, "fsdd_spec.npz"))
        X, y = d["X"].astype(np.float64), d["y"]
        perm = np.random.default_rng(seed * 1000 + 7).permutation(len(X))
        X, y = X[perm], y[perm]
        n, n_te, n_va = len(X), 600, 400
        return ((X[: n - n_te - n_va], y[: n - n_te - n_va]),
                (X[n - n_te - n_va: n - n_te], y[n - n_te - n_va: n - n_te]),
                (X[n - n_te:], y[n - n_te:]), 16)
    gen = torch.Generator().manual_seed(seed * 1000 + 7)
    if kind == "sweeps":
        x, y = make_sweeps(n, gen, T=T)
        X = x.reshape(n, -1).numpy().astype(np.float64)     # flat (n, T*16)
        in_dim = 16
    else:
        x, y = make_shapes(n, gen)
        X = x.numpy().astype(np.float64)                    # flat (n, 144)
        in_dim = x.shape[-1]
    y = y.numpy().astype(np.int64)
    n_te, n_va = 2500, 1500
    return ((X[: n - n_te - n_va], y[: n - n_te - n_va]),
            (X[n - n_te - n_va: n - n_te], y[n - n_te - n_va: n - n_te]),
            (X[n - n_te:], y[n - n_te:]), in_dim)


# --------------------------------------------------------------------------- #
# ours
# --------------------------------------------------------------------------- #

def run_ours(splits, in_dim, cfg, seed, listen=True):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    org = Organism()
    fa_log = []
    growth_flips = 0
    for t, cls in enumerate(TASKS):
        trm, vam = np.isin(ytr, cls), np.isin(yva, cls)
        Xt_tr, yt_tr = Xtr[trm], ytr[trm]
        Xt_va, yt_va = Xva[vam], yva[vam]
        if org.organs:
            r0 = org.predict(Xt_tr)
            fa_before = float((r0["pred"] != ABSTAIN).mean())
            org.tighten_against(Xt_tr, self_budget=0.0,
                                veto_theta_q=cfg["veto_theta_q"])
            fa_after = float((org.predict(Xt_tr)["pred"] != ABSTAIN).mean())
        else:
            fa_before = fa_after = 0.0
        fa_log.append({"task": t, "fa_before": fa_before, "fa_after": fa_after})

        p_before = org.predict(Xte)["pred"] if org.organs else None
        existing = (torch.cat([o.pos for o in org.organs])
                    if org.organs else None)
        cand = SheetOrgan(in_dim, T, N_UNITS, cls,
                          center=place_center(existing),
                          prev_organs=list(org.organs), seed=seed * 100 + t,
                          listen=listen, q_cov_in=cfg["q_cov_in"],
                          static=cfg["static"])
        need = org.declined_mask(Xt_tr) if org.organs else np.ones(len(Xt_tr), bool)
        cand.train_proto(Xt_tr[need], yt_tr[need], epochs=cfg["epochs"])
        org.try_grow(cand, Xt_va, yt_va, p_target=cfg["p_target"],
                     gate=(t > 0), label=f"task{t}")
        if p_before is not None:
            p_after = org.predict(Xte)["pred"]
            was = p_before != ABSTAIN
            growth_flips += int((was & (p_after != p_before)).sum())
    final = org.scores(Xte, yte)
    return org, {"final": final, "growth_flips": growth_flips,
                 "false_accepts": fa_log, "n_params": org.n_params()}


# --------------------------------------------------------------------------- #
# monolithic recurrent baseline
# --------------------------------------------------------------------------- #

class MonoSheet(nn.Module):
    def __init__(self, in_dim, T, n, n_classes, static, seed=0, g=0.9):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.in_dim, self.T, self.n, self.static = in_dim, T, n, static
        self.W = nn.Parameter(torch.randn(n, n, generator=gen) * (g / math.sqrt(n)))
        self.U = nn.Parameter(torch.randn(n, in_dim, generator=gen) / math.sqrt(in_dim))
        self.b = nn.Parameter(torch.zeros(n))
        self.head = nn.Linear(n, n_classes)

    def forward(self, xflat):
        if self.static:
            X3 = xflat.unsqueeze(1).expand(-1, self.T, -1)
        else:
            X3 = xflat.reshape(len(xflat), self.T, self.in_dim)
        h = torch.zeros(len(xflat), self.n)
        acc = 0.0
        for t in range(self.T):
            h = torch.tanh(h @ self.W.t() + X3[:, t] @ self.U.t() + self.b)
            acc = acc + h
        return self.head(acc / self.T)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def acc_of(model, X, y):
    pred = model(torch.as_tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return float((pred == y).mean())


def run_baseline(method, splits, in_dim, cfg, seed):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    n_cls = 1 + max(c for cl in TASKS for c in cl)
    model = MonoSheet(in_dim, T, len(TASKS) * N_UNITS, n_cls,
                      cfg["static"], seed=seed)
    anchors_prev = []
    buf_X = np.empty((0, Xtr.shape[1])); buf_y = np.empty((0,), dtype=np.int64)
    for t, cls in enumerate(TASKS):
        m = np.isin(ytr, cls)
        Xt, yt = Xtr[m], ytr[m]
        extra = None
        if method == "ewc" and anchors_prev:
            def extra(mod, anchors_prev=anchors_prev):
                loss = 0.0
                for ps, fs in anchors_prev:
                    for p, p0, f in zip(mod.parameters(), ps, fs):
                        loss = loss + (f * (p - p0) ** 2).sum()
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
    return {"final_acc": acc_of(model, Xte, yte), "n_params": model.n_params()}


def run_joint(splits, in_dim, cfg, seed):
    (Xtr, ytr), _, (Xte, yte) = splits
    n_cls = 1 + max(c for cl in TASKS for c in cl)
    model = MonoSheet(in_dim, T, len(TASKS) * N_UNITS, n_cls,
                      cfg["static"], seed=seed)
    _epochs(model, Xtr, ytr, cfg["b_epochs"], 1e-3, 256, seed)
    return acc_of(model, Xte, yte)


# --------------------------------------------------------------------------- #
# anatomy
# --------------------------------------------------------------------------- #

def anatomy(org: Organism):
    """Modularity Q under the organ partition, and weighted mean edge length
    vs the random-pairing control."""
    w_all, d_all, blocks = [], [], []
    for i, o in enumerate(org.organs):
        w, d = o.edges()
        w_all.append(w); d_all.append(d)
        # within-organ edges: self block; listening edges cross blocks
        blocks.append((i, w[: int(o.edge_mask_self.sum())]))
    w = torch.cat(w_all); d = torch.cat(d_all)
    m = float(w.sum())
    within = sum(float(b.sum()) for _, b in blocks)
    # degree null: organs' out/in strength products (self rows only own units)
    # simple two-level Q: fraction within-organ minus expected under random
    # placement of the same strength across all unit pairs
    n_units = sum(o.n for o in org.organs)
    n_pairs_within = sum(o.n * o.n for o in org.organs)
    q_null = n_pairs_within / (n_units * n_units)
    Q = within / m - q_null
    mean_len = float((w * d).sum() / m)
    pos = torch.cat([o.pos for o in org.organs])
    ctrl = float(torch.cdist(pos, pos).mean())
    return {"Q": Q, "within_frac": within / m, "q_null": q_null,
            "mean_edge_len": mean_len, "random_pair_len": ctrl}


def lesion_matrix(org: Organism, Xte, yte):
    """Row k: strict accuracy per task after silencing organ k."""
    base = [org.scores(Xte[np.isin(yte, cls)], yte[np.isin(yte, cls)])["strict"]
            for cls in TASKS]
    rows = []
    for k in range(len(org.organs)):
        o2 = copy.deepcopy(org)
        ok = o2.organs[k]
        with torch.no_grad():
            ok.W_self.zero_(); ok.U.zero_(); ok.b.zero_()
            if ok.W_prev is not None:
                ok.W_prev.zero_()
        ok.rho = -1.0            # silenced organ accepts nothing
        rows.append([o2.scores(Xte[np.isin(yte, cls)], yte[np.isin(yte, cls)])["strict"]
                     for cls in TASKS])
    return {"base": base, "lesioned": rows}


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=["sweeps", "shapes"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--out-name", default="sheet.json",
                    help="write here instead of results/sheet.json")
    args = ap.parse_args()

    global TASKS
    out = {}
    for kind in args.streams:
        # "<stream>_spk" = the speaker-disjoint variant of a stream's config
        cfg = CFG[kind.replace("_spk", "")]
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            splits_all = make_data(kind, seed)
            n_cls = int(splits_all[0][1].max()) + 1
            TASKS = [[2 * i, 2 * i + 1] for i in range(n_cls // 2)]
            splits, in_dim = splits_all[:3], splits_all[3]
            torch.manual_seed(seed)
            org, ours = run_ours(splits, in_dim, cfg, seed, listen=True)
            anat = anatomy(org)
            les = lesion_matrix(org, splits[2][0], splits[2][1])
            _, ours_nl = run_ours(splits, in_dim, cfg, seed, listen=False)
            base = {m: run_baseline(m, splits, in_dim, cfg, seed)
                    for m in ["finetune", "ewc", "replay"]}
            joint = run_joint(splits, in_dim, cfg, seed)
            rec = {"seed": seed, "ours": ours, "ours_nolisten": ours_nl,
                   "anatomy": anat, "lesion": les,
                   "baselines": base, "joint": joint}
            recs.append(rec)
            f = ours["final"]
            print(f"[{kind} seed {seed}] ({time.time()-t0:.0f}s)")
            print(f"  ours(listen) : strict {f['strict']:.3f}  forced {f['forced']:.3f}"
                  f"  cov {f['coverage']:.3f}  prec {f['precision']:.3f}"
                  f"  gflips {ours['growth_flips']}  params {ours['n_params']}")
            fn = ours_nl["final"]
            print(f"  ours(deaf)   : strict {fn['strict']:.3f}  cov {fn['coverage']:.3f}"
                  f"  prec {fn['precision']:.3f}  gflips {ours_nl['growth_flips']}")
            print("  fa (before->after): " + "  ".join(
                f"t{x['task']}: {x['fa_before']:.2f}->{x['fa_after']:.2f}"
                for x in ours["false_accepts"][1:]))
            print(f"  anatomy      : Q {anat['Q']:.3f} (within {anat['within_frac']:.3f}"
                  f" vs null {anat['q_null']:.3f})  edge len {anat['mean_edge_len']:.3f}"
                  f" vs random {anat['random_pair_len']:.3f}")
            for m in ["finetune", "ewc", "replay"]:
                print(f"  {m:12s}: final {base[m]['final_acc']:.3f}  params {base[m]['n_params']}")
            print(f"  joint        : {joint:.3f}")
        out[kind] = recs

    p = os.path.join(RES, args.out_name)
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
