"""B-head-blocks -- a contracted head that never re-solves.

`exp_head.py` showed the accuracy lever: a random-projection ridge readout under
a ball contract reaches or passes the published exemplar-free methods. It pays
for that with the one property the cascade has and no published method reports:
its weights are re-solved after every task, so 8 to 155 already-given answers
per 100 test points change over a run, against exactly 0 for the cascade.

This asks whether both can be had. Instead of one ridge over all classes,
re-solved as classes arrive, each task gets its OWN ridge block, solved from that
task's data alone and then FROZEN, exactly as an organ is frozen:

  block t : W_t = (G_t + lambda_t I)^-1 Q_t   over task t's classes only,
            G_t, Q_t accumulated from task t's training rows only,
            lambda_t and the margin tau_t solved on task t's validation rows only
  contract: accept iff the query lies inside the PCA territory of the class the
            block predicts AND the block's top-two margin clears tau_t
  answer  : first-accept over blocks in birth order, as the organism does

Because a block is solved once and never touched, and blocks are consulted in
birth order, appending a block cannot change an answer already given. The script
checks that bitwise (`growth_flips`, which must be 0) rather than assuming it.

Accounting follows the flat protocol exactly, and the distinction matters: an
earlier block must still be TIGHTENED against each incoming task or it answers
everything (without tightening, strict accuracy is 0.148 at coverage 1.00 on
CIFAR-ViT). Tightening is monotone per block -- tau only rises, rho_in only
falls -- so a block's own answer can become an abstention but never a different
answer; in a cascade, however, that abstention can fall through to a later
block, which may answer differently. Those are logged repairs
(`tighten_to_abstain`, `tighten_to_other`, with the right/wrong split), exactly
as `exp_continual.py` logs `tighten_flips` separately from `growth_flips`.
--no-tighten ablates the tightening.

Run:  python exp_head_blocks.py --cache cifar100_vitb --M 2000 10000 --first-seed 2 --seeds 3
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
ABSTAIN = -1


def project(X, W):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return np.maximum(Xn @ W, 0.0)


class Block:
    """One frozen ridge readout over its own task's classes, plus its contract."""

    def __init__(self, classes, Wo, tau, rho_in, proxes):
        self.classes = np.asarray(classes)
        self.Wo = Wo                  # (M, |classes|)
        self.tau = float(tau)
        self.rho_in = float(rho_in)
        self.proxes = proxes          # class -> PCAProx
        self.vetoes = []              # (class_idx, centre, radius) monotone

    def judge(self, H, X):
        S = H @ self.Wo
        o = np.argsort(-S, axis=1)
        top = o[:, 0]
        second = o[:, 1] if S.shape[1] > 1 else o[:, 0]
        margin = S[np.arange(len(S)), top] - S[np.arange(len(S)), second]
        pred = self.classes[top]
        d_in = np.array([self.proxes[p].score(x[None])[0] for p, x in zip(pred, X)])
        acc = (margin > self.tau) & (d_in <= self.rho_in)
        if self.vetoes:
            for ci, centre, radius in self.vetoes:
                sel = pred == self.classes[ci]
                if sel.any():
                    dist = np.linalg.norm(X[sel] - centre, axis=1)
                    v = np.zeros(len(X), bool); v[np.flatnonzero(sel)[dist < radius]] = True
                    acc &= ~v
        return pred, acc, margin, d_in

    def tighten_against(self, H_fa, X_fa, self_budget=0.02, n_grid=8):
        """Monotone tightening against foreign data, mirroring
        Organ.solve_tightening: tau may only RISE and rho_in only FALL, so an
        answer can become an abstention but never a different answer. Chooses
        the pair excluding the most false accepts subject to losing at most
        self_budget of the block's own birth coverage. Returns (excluded, loss)."""
        s = self.stats
        own = (s["margin"] > self.tau) & (s["d_in"] <= self.rho_in)
        n_own = max(int(own.sum()), 1)
        _, fa_acc, m_fa, din_fa = self.judge(H_fa, X_fa)
        if fa_acc.sum() == 0:
            return 0, 0.0

        def grid_up(v, cur):
            v = v[np.isfinite(v)]
            if not len(v):
                return [cur]
            g = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_grid)))
            return [*g[g >= cur], cur]

        def grid_down(v, cur):
            v = v[np.isfinite(v)]
            if not len(v):
                return [cur]
            g = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_grid)))
            return [*g[g <= cur], cur]

        best = (0, self.tau, self.rho_in, 0.0)
        for tau in grid_up(m_fa[fa_acc], self.tau):
            for rin in grid_down(din_fa[fa_acc], self.rho_in):
                keep_own = (s["margin"] > tau) & (s["d_in"] <= rin)
                loss = 1.0 - keep_own.sum() / n_own
                if loss > self_budget:
                    continue
                excl = int(fa_acc.sum() - ((m_fa > tau) & (din_fa <= rin)).sum())
                if excl > best[0]:
                    best = (excl, float(tau), float(rin), float(loss))
        excluded, self.tau, self.rho_in, loss = best
        return excluded, loss

    def n_params(self):
        return int(self.Wo.size)


def cascade(blocks, H, X):
    """First-accept over blocks in birth order."""
    n = len(X)
    pred = np.full(n, ABSTAIN, dtype=np.int64)
    forced = np.full(n, ABSTAIN, dtype=np.int64)
    who = np.full(n, -1, dtype=np.int64)
    best_margin = np.full(n, -np.inf)
    open_mask = np.ones(n, bool)
    for i, b in enumerate(blocks):
        p, a, m, _ = b.judge(H, X)
        take = open_mask & a
        pred[take] = p[take]; who[take] = i
        open_mask &= ~a
        better = m > best_margin          # forced fallback: most confident block
        forced[better] = p[better]; best_margin[better] = m[better]
    return pred, np.where(pred != ABSTAIN, pred, forced), who


def run(cache, M, seed, tighten=True, self_budget=0.02):
    feats = os.path.join(RES, f"{cache}.npz")
    meta = np.load(feats, allow_pickle=False)
    n_cls = int(meta["n_classes"]) if "n_classes" in meta.files else None
    inc = int(meta["inc"]) if "inc" in meta.files else 10
    del meta
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_feats(seed, feats_path=feats)
    if n_cls is None:
        n_cls = int(ytr.max()) + 1
    order = np.random.default_rng(seed + 400).permutation(n_cls)
    tasks = [sorted(int(c) for c in order[i:i + inc]) for i in range(0, n_cls, inc)]
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((Xtr.shape[1], M)).astype(np.float64)
    Hva, Hte = project(Xva, W), project(Xte, W)

    blocks = []
    tighten_log = []
    growth_flips = 0
    tighten_to_abstain = tighten_to_other = tighten_to_right = tighten_to_wrong = 0
    curve = []
    t0 = time.time()
    for t, cls in enumerate(tasks):
        tr = np.flatnonzero(np.isin(ytr, cls))
        va = np.flatnonzero(np.isin(yva, cls))
        # ---- this task's ridge, from this task's rows only -----------------
        G = np.zeros((M, M)); Q = np.zeros((M, len(cls)))
        col = {c: j for j, c in enumerate(cls)}
        for s0 in range(0, len(tr), 4000):
            ii = tr[s0:s0 + 4000]
            H = project(Xtr[ii], W)
            G += H.T @ H
            Y = np.zeros((len(ii), len(cls)))
            Y[np.arange(len(ii)), [col[int(c)] for c in ytr[ii]]] = 1.0
            Q += H.T @ Y
        proxes = {c: PCAProx(n_components=24, seed=seed).fit(Xtr[ytr == c], ytr[ytr == c])
                  for c in cls}
        # lambda on this task's validation rows only (never later tasks')
        best = (-1.0, None, None)
        for lam in LAMBDAS:
            Wo = np.linalg.solve(G + lam * np.eye(M), Q)
            acc = float((np.asarray(cls)[(Hva[va] @ Wo).argmax(1)] == yva[va]).mean())
            if acc > best[0]:
                best = (acc, lam, Wo)
        own_val_acc, lam, Wo = best
        # tau and rho_in from this task's validation rows, as the organs do
        S = Hva[va] @ Wo
        o = np.argsort(-S, axis=1); top = o[:, 0]
        second = o[:, 1] if S.shape[1] > 1 else o[:, 0]
        margin = S[np.arange(len(S)), top] - S[np.arange(len(S)), second]
        pred_v = np.asarray(cls)[top]
        correct = pred_v == yva[va]
        d_in = np.array([proxes[p].score(x[None])[0] for p, x in zip(pred_v, Xva[va])])
        rho_in = conformal_quantile(d_in[correct], 0.01) if correct.any() else np.inf
        tau_best = (0.0, 0.0)
        for cand in np.quantile(margin, np.linspace(0.0, 0.8, 12)):
            a = (margin > cand) & (d_in <= rho_in)
            if a.sum() and correct[a].mean() >= 0.6:
                s = float((a & correct).mean())
                if s > tau_best[0]:
                    tau_best = (s, float(cand))
        b = Block(cls, Wo, tau_best[1], rho_in, proxes)
        b.stats = {"margin": margin, "d_in": d_in}      # own validation, at birth
        # ---- accounting, split exactly as the flat protocol splits it ------
        # TIGHTENING may turn an answer into an abstention, and in a cascade a
        # fall-through can then be answered differently by a later block: those
        # are logged repairs, not a broken invariant. GROWTH -- appending the
        # new block -- must never change an answer already given, and that is
        # what is checked bitwise below.
        p_before = cascade(blocks, Hte, Xte)[0] if blocks else None
        if tighten and blocks:
            H_fa = project(Xtr[tr], W)
            for eb in blocks:
                excl, loss = eb.tighten_against(H_fa, Xtr[tr],
                                                self_budget=self_budget)
                tighten_log.append({"task": t, "excluded": excl, "self_loss": loss})
        if p_before is not None:
            p_tight = cascade(blocks, Hte, Xte)[0]
            was = p_before != ABSTAIN
            tighten_to_abstain += int((was & (p_tight == ABSTAIN)).sum())
            ch = was & (p_tight != ABSTAIN) & (p_tight != p_before)
            tighten_to_other += int(ch.sum())
            tighten_to_right += int((ch & (p_tight == yte)).sum())
            tighten_to_wrong += int((ch & (p_before == yte)).sum())
        else:
            p_tight = None
        blocks.append(b)
        if p_tight is not None:
            p_grown = cascade(blocks, Hte, Xte)[0]
            was = p_tight != ABSTAIN
            growth_flips += int((was & (p_grown != p_tight)).sum())

        seen = sorted(c for cl in tasks[:t + 1] for c in cl)
        mt = np.isin(yte, seen)
        pred, forced, who = cascade(blocks, Hte, Xte)
        sc = {"task": t, "lambda": lam, "own_val_acc": own_val_acc,
              "tau": b.tau, "strict_seen": float((pred[mt] == yte[mt]).mean()),
              "coverage_seen": float((pred[mt] != ABSTAIN).mean())}
        curve.append(sc)

    pred, forced, who = cascade(blocks, Hte, Xte)
    ans = pred != ABSTAIN
    out = {"seed": seed, "cache": cache, "M": M, "mode": "blocks",
           "tighten": bool(tighten), "self_budget": self_budget,
           "tighten_excluded": int(sum(x["excluded"] for x in tighten_log)),
           "tighten_self_loss_max": max([x["self_loss"] for x in tighten_log] or [0.0]),
           "n_classes": n_cls, "inc": inc, "n_blocks": len(blocks),
           "strict": float((pred == yte).mean()),
           "forced": float((forced == yte).mean()),
           "coverage": float(ans.mean()),
           "precision": float((pred[ans] == yte[ans]).mean()) if ans.any() else 0.0,
           "growth_flips": int(growth_flips),
           "tighten_to_abstain": int(tighten_to_abstain),
           "tighten_to_other": int(tighten_to_other),
           "tighten_to_right": int(tighten_to_right),
           "tighten_to_wrong": int(tighten_to_wrong),
           "n_test": int(len(yte)),
           "head_params": int(W.size + sum(b.n_params() for b in blocks)),
           "tighten_events": len(tighten_log),
           "curve": curve, "elapsed_s": time.time() - t0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cifar100_vitb")
    ap.add_argument("--M", type=int, nargs="+", default=[2000, 10000])
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-tighten", action="store_true",
                    help="ablate the monotone tightening (blocks never meet later data)")
    ap.add_argument("--self-budget", type=float, default=0.02)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    out = os.path.join(RES, args.out_name or
                       f"headblocks_{args.cache}{'_notighten' if args.no_tighten else ''}.json")
    recs = []
    for M in args.M:
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            r = run(args.cache, M, seed, tighten=not args.no_tighten,
                    self_budget=args.self_budget)
            recs.append(r)
            print(f"[blocks {args.cache} M={M} seed {seed}] ({r['elapsed_s']:.0f}s) "
                  f"strict {r['strict']:.3f} forced {r['forced']:.3f} cov {r['coverage']:.3f} "
                  f"prec {r['precision']:.3f} | blocks {r['n_blocks']} tightenings {r['tighten_events']} "
                  f"(excluded {r['tighten_excluded']}, worst self-loss {r['tighten_self_loss_max']:.3f}) "
                  f"| GROWTH FLIPS {r['growth_flips']} "
                  f"| tightening: {r['tighten_to_abstain']} -> abstain, {r['tighten_to_other']} -> other "
                  f"({r['tighten_to_right']}R:{r['tighten_to_wrong']}W)",
                  flush=True)
            with open(out, "w") as f:
                json.dump(recs, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
