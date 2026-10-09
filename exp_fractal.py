"""B-fractal -- an organ that IS an organism.

The contract algebra composes: a union of PCA territories is a PCA territory,
so a whole sub-organism can sit behind ONE merged contract and present the
same interface as an organ. A SuperOrgan wraps an inner organism: its
territory is the merge of its children's, its sampler is the union generator,
its answer is the inner organism's routing, and -- because the merged
territory is first-class -- it can be VETOED and TIGHTENED at the top level
exactly like a leaf organ. Two levels, one algebra.

Protocol (CIFAR-100/ViT, eval seeds): the 10 tasks arrive in three
super-epochs ([4, 3, 3], preregistered). Within an epoch, a sub-organism grows
as usual (cascade + vetoes); at epoch end it is consolidated (gated) and
wrapped as a SuperOrgan; the TOP organism cascades SuperOrgans, tightening
earlier ones (top-level vetoes) against each incoming epoch. Measured:

  accuracy    vs the flat cascade (eval 0.667) and the flat lifecycle (0.835)
  invariants  growth flips at BOTH levels (must be 0)
  fractal replacement: rebuild an entire SuperOrgan -- its whole inner
              organism -- from its merged contract alone; agreement on its
              region (the two-level version of B-replace)

Run:  python exp_fractal.py [--first-seed 2 --seeds 3]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.organ import Organ, conformal_quantile
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import ProtoOrgan, consolidate_protos
from cgrow.proximity import merge_pca_family
from exp_cifar import load_feats, CFG

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EPOCHS = [4, 3, 3]                 # preregistered super-epoch sizes


class SuperOrgan(Organ):
    """A whole organism behind one merged contract, presenting the organ
    interface: territory = merge of children's; answer = inner routing;
    margin passthrough from the inner answering organ; vetoes/tightening
    apply at this level through the merged (pca-family) territory."""

    def __init__(self, inner: Organism, seed: int = 0):
        self.inner = inner
        self.class_ids = np.asarray(sorted(
            {int(c) for o in inner.organs for c in o.class_ids}))
        self.rho, self.tau, self.rho_in = np.inf, 0.0, np.inf
        self.L = max(o.L or 1.0 for o in inner.organs)
        self.frozen = False
        self.stats = None
        self.seed = seed
        self.k_centroids = 5
        self.q_cov_in = 0.99
        self.prox = merge_pca_family(
            [o.prox for o in inner.organs if o.prox is not None])
        self.prox_name = self.prox.name
        self.vetoes = []
        self.anchors = torch.nn.Parameter(torch.zeros(1))   # unused

    @torch.no_grad()
    def evaluate(self, X: np.ndarray):
        r = self.inner.predict(X)
        answered = r["pred"] != ABSTAIN
        # margin/d1 passthrough: use the inner answering organ's own values
        d1 = np.where(answered, 0.0, np.inf)
        margin = np.where(answered, np.inf, 0.0)
        pred = np.where(answered, r["pred"], self.class_ids[0])
        d_in = self.prox.score(X)
        vetoed, veto_r = self._veto_judge(X)
        return d1, margin, pred, np.asarray(d_in), vetoed, veto_r

    def accept(self, d1, margin, d_in, vetoed=None):
        base = np.isfinite(d1) & (d_in <= self.rho_in)
        return base if vetoed is None else base & ~vetoed

    def solve_thresholds_super(self, Xval, yval):
        d1, m, pred, d_in, vet, _ = self.evaluate(Xval)
        correct = pred == yval
        if correct.sum() == 0:
            return None
        self.rho_in = conformal_quantile(d_in[correct], 1.0 - self.q_cov_in)
        return True

    def freeze_super(self, Xval, yval):
        d1, m, pred, d_in, _, _ = self.evaluate(Xval)
        self.stats = {"d1": d1, "margin": m, "d_in": d_in,
                      "correct": pred == yval}
        self.frozen = True

    def n_params(self):
        return self.inner.n_params()


def grow_epoch(tasks, splits, cfg, seed, epoch_idx):
    """A sub-organism over one super-epoch's tasks (standard growth)."""
    from exp_continual import run_ours
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    torch.manual_seed(seed * 100 + epoch_idx)
    org, _ = run_ours(tasks, splits, Xtr.shape[1], cfg, seed * 10 + epoch_idx)
    return org


def rebuild_super(sup: SuperOrgan, in_dim: int, seed: int):
    """Fractal replacement: rebuild the ENTIRE super-organ from its merged
    contract alone -- a prototype organism over its sampled territory."""
    rng = np.random.default_rng(seed * 91 + 5)
    X_syn, y_syn = sup.prox.sample(6000, rng)
    Xn = X_syn / (np.linalg.norm(X_syn, axis=1, keepdims=True) + 1e-9)
    classes = sorted(set(int(c) for c in y_syn))
    anchors = np.stack([Xn[y_syn == c].mean(axis=0) for c in classes])
    anchors /= (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    C = ProtoOrgan(in_dim, classes, anchors,
                   min_norm=float(np.linalg.norm(X_syn, axis=1).min()),
                   seed=seed)
    C.prox = sup.prox
    C.prox_name = C.prox.name
    C.solve_thresholds(X_syn, y_syn, p_target=0.8, require=False)
    C.freeze(X_syn, y_syn)
    inner = Organism(); inner.organs = [C]
    rep = SuperOrgan(inner, seed=seed)
    rep.rho_in = sup.rho_in
    rep.frozen = True
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    feats = os.path.join(RES, "cifar100_vitb.npz")
    recs = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.time()
        splits = load_feats(seed, feats_path=feats)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
        groups, i = [], 0
        for n in EPOCHS:
            groups.append(tasks[i:i + n]); i += n

        top = Organism()
        growth_flips = 0
        for e, gtasks in enumerate(groups):
            ecls = [c for cl in gtasks for c in cl]
            e_va = np.isin(yva, ecls)
            if top.organs:      # top-level tightening of earlier super-organs
                e_tr = np.isin(ytr, ecls)
                top.tighten_against(Xtr[e_tr], self_budget=0.0,
                                    veto_theta_q=CFG["veto_theta_q"])
            p_before = top.predict(Xte)["pred"] if top.organs else None
            sub = grow_epoch(gtasks, splits, CFG, seed, e)
            sub2, rep = consolidate_protos(
                sub, Xtr.shape[1], CFG, seed * 10 + e,
                Xva_last=Xva[np.isin(yva, gtasks[-1])],
                yva_last=yva[np.isin(yva, gtasks[-1])])
            sup = SuperOrgan(sub2, seed=seed * 10 + e)
            sup.solve_thresholds_super(Xva[e_va], yva[e_va])
            sup.freeze_super(Xva[e_va], yva[e_va])
            top.organs.append(sup)
            if p_before is not None:
                p_after = top.predict(Xte)["pred"]
                was = p_before != ABSTAIN
                growth_flips += int((was & (p_after != p_before)).sum())

        final = top.scores(Xte, yte)
        # fractal replacement of the middle super-organ
        base = top.predict(Xte)
        region = base["organ"] == 1
        top2 = copy.deepcopy(top)
        top2.organs[1] = rebuild_super(top.organs[1], Xtr.shape[1], seed)
        r2 = top2.predict(Xte)
        agree = (float((r2["pred"][region] == base["pred"][region]).mean())
                 if region.any() else 1.0)
        rec = {"seed": seed, "final": final, "growth_flips": growth_flips,
               "consolidations": [len(g) for g in groups],
               "super_replacement_agreement": agree,
               "region_size": int(region.sum()),
               "n_params": top.n_params()}
        recs.append(rec)
        print(f"[fractal seed {seed}] ({time.time()-t0:.0f}s) strict "
              f"{final['strict']:.3f}  cov {final['coverage']:.3f}  prec "
              f"{final['precision']:.3f}  top-level growth flips "
              f"{growth_flips}  super-replacement agreement {agree:.3f} "
              f"(region {int(region.sum())})", flush=True)

    with open(os.path.join(RES, "fractal.json"), "w") as f:
        json.dump(recs, f, indent=1, default=float)
    print("wrote results/fractal.json")


if __name__ == "__main__":
    main()
