"""B-market -- routing by auction: organs bid self-calibrated confidence.

First-accept routing is an accident of birth order; the arbiter fixed it with
a centrally TRAINED router. The market fixes it decentrally: each organ
publishes a PRICE CURVE -- a logistic map from its own evidence (margin,
territory score, anchor distance) to P(my answer is correct) -- fitted at
birth from its stored contract statistics (organ.stats). No new data, no
central training; the contract gains a price field. Inference is an auction:
the highest bid answers; a reserve price (swept on the contract-sampled pool
only) sets abstention.

Guarantee tier: intermediate -- answers may change only when a new bidder
enters (logged events); routing itself involves no mutable trained state
beyond each organ's frozen price curve.

Self-pricing fails (anti-calibrated: an organ fitted on its own distribution
has never seen a negative). The MARKET-MAKER is the diagnosis test: identical
auction, identical per-organ logistic price model, but calibrated on the
SHARED contract pool so every organ sees the others' territory as negatives.
Decentralized model, joint data -- it separates "prices need joint data" from
"routing needs a central network".

Measured, eval seeds 2-4, everything preregistered: strict accuracy of
  cascade < self-priced market < market-maker < arbiter?
on the raw cascade and on the consolidated ([C]+cascade) organism, plus
per-organ win shares and price-curve calibration (Brier score on the pool).

Run:  python exp_market.py [--benchmarks digits kmnist cifar_vitb]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import load_real, prepare, split_tasks
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import consolidate_protos, sample_pool
from exp_continual import run_ours, CFG as MLP_CFG
from exp_cifar import load_feats, CFG as CIFAR_CFG
from exp_pfc import evidence, train_arbiter, arbiter_predict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BIG = 100.0


def fit_prices(org: Organism):
    """Each organ's price curve: logistic on its stored birth statistics."""
    from sklearn.linear_model import LogisticRegression
    curves = []
    for o in org.organs:
        s = o.stats
        F = np.stack([np.minimum(s["margin"], BIG), s["d_in"], s["d1"]], axis=1)
        y = s["correct"].astype(int)
        if y.sum() in (0, len(y)):          # degenerate birth val
            curves.append(("const", float(y.mean())))
            continue
        lr = LogisticRegression(max_iter=300).fit(F, y)
        curves.append(("lr", lr))
    return curves


def fit_prices_pool(org: Organism, X_pool: np.ndarray, y_pool: np.ndarray):
    """The MARKET-MAKER: same decentralized price model (one tiny logistic per
    organ, same three evidence features), but calibrated on the SHARED contract
    pool -- joint data, decentralized model. Isolates how much of the arbiter's
    advantage is central modeling vs simply seeing everyone's negatives."""
    from sklearn.linear_model import LogisticRegression
    curves = []
    for o in org.organs:
        d1, m, p, d_in, vet, _ = o.evaluate(X_pool)
        F = np.stack([np.minimum(m, BIG), d_in, d1], axis=1)
        y = (p == y_pool).astype(int)
        if y.sum() in (0, len(y)):
            curves.append(("const", float(y.mean())))
            continue
        curves.append(("lr", LogisticRegression(max_iter=300).fit(F, y)))
    return curves


def bids(org: Organism, curves, X: np.ndarray):
    """(n, K) bid matrix and (K, n) forced predictions."""
    B, P = [], []
    for o, (kind, c) in zip(org.organs, curves):
        d1, m, p, d_in, vet, _ = o.evaluate(X)
        F = np.stack([np.minimum(m, BIG), d_in, d1], axis=1)
        b = (np.full(len(X), c) if kind == "const"
             else c.predict_proba(F)[:, 1])
        b = np.where(vet, 0.0, b)          # a vetoed answer bids nothing
        B.append(b); P.append(p)
    return np.stack(B, axis=1), np.stack(P)


def market_predict(B, P, reserve=None):
    who = B.argmax(axis=1)
    ans = P[who, np.arange(P.shape[1])]
    if reserve is not None:
        ans = np.where(B.max(axis=1) >= reserve, ans, ABSTAIN)
    return ans, who


def market_predict_indexed(B, P, lam, reserve=None, c_idx=0, fallback="c"):
    """The C-INDEXED auction: the consolidated generalist is the price
    reference, and a specialist must outbid lam x C's bid to take the answer.
    A flat reserve asks "is anyone confident?"; an indexed reserve asks "is
    this specialist more confident than the generalist, here?" -- the question
    a pool-calibrated price can actually answer.

    fallback="c":       C answers when no specialist outbids it.
    fallback="abstain": nobody answers -- isolates the ROUTING effect from C's
                        answering power (mandatory where C's gate refused it).
    """
    n, K = B.shape
    spec = np.array([j for j in range(K) if j != c_idx])
    Bs = B[:, spec]
    j = Bs.argmax(axis=1)
    b_spec = Bs[np.arange(n), j]
    p_spec = P[spec][j, np.arange(n)]
    b_c, p_c = B[:, c_idx], P[c_idx]
    win = b_spec >= lam * b_c
    if fallback == "c":
        ans = np.where(win, p_spec, p_c)
        b_win = np.where(win, b_spec, b_c)
    else:
        ans = np.where(win, p_spec, ABSTAIN)
        b_win = np.where(win, b_spec, 0.0)
    if reserve is not None:
        ans = np.where(b_win >= reserve, ans, ABSTAIN)
    return ans, win


LAMS = np.linspace(0.4, 2.0, 17)
RESERVES = np.linspace(0.2, 0.9, 15)


def _sweep_indexed(B, P, y, c_idx, fallback):
    """Sweep (lambda, reserve) on the contract-sampled holdout only."""
    best = (-1.0, None, None)
    for lam in LAMS:
        for r in RESERVES:
            a, _ = market_predict_indexed(B, P, lam, r, c_idx, fallback)
            s = float((a == y).mean())
            if s > best[0]:
                best = (s, float(lam), float(r))
    return best


def run_market(org: Organism, splits, seed, arbiter_too=True, pricing="self",
               c_idx=None):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    rng = np.random.default_rng(seed * 61 + 7)
    # pool for pricing/reserve/calibration/arbiter (contract samples only)
    X_syn, y_syn = sample_pool(org, 2000, rng)
    perm = rng.permutation(len(X_syn))
    ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
    curves = (fit_prices_pool(org, X_syn[tr], y_syn[tr])
              if pricing == "pool" else fit_prices(org))
    B_ho, P_ho = bids(org, curves, X_syn[ho])
    best = (0.0, None)
    for r in np.linspace(0.2, 0.9, 15):
        a, _ = market_predict(B_ho, P_ho, r)
        s = float((a == y_syn[ho]).mean())
        if s > best[0]:
            best = (s, float(r))
    reserve = best[1]
    # calibration of the price curves on the pool
    conf = B_ho.max(axis=1)
    a0, _ = market_predict(B_ho, P_ho, None)
    brier = float(np.mean((conf - (a0 == y_syn[ho])) ** 2))

    B_te, P_te = bids(org, curves, Xte)
    ans, who = market_predict(B_te, P_te, reserve)
    out = {"market_strict": float((ans == yte).mean()),
           "market_forced": float((market_predict(B_te, P_te)[0] == yte).mean()),
           "market_coverage": float((ans != ABSTAIN).mean()),
           "reserve": reserve, "pool_brier": brier,
           "win_share": np.bincount(who, minlength=len(org.organs)
                                    ).tolist()}
    if c_idx is not None:
        for fb in ("c", "abstain"):
            s_ho, lam, r = _sweep_indexed(B_ho, P_ho, y_syn[ho], c_idx, fb)
            a, win = market_predict_indexed(B_te, P_te, lam, r, c_idx, fb)
            out[f"indexed_{fb}"] = {
                "strict": float((a == yte).mean()),
                "coverage": float((a != ABSTAIN).mean()),
                "lam": lam, "reserve": r, "pool_holdout": s_ho,
                "spec_win_rate": float(win.mean())}
    if arbiter_too:
        F_tr, P_tr = evidence(org, X_syn[tr])
        net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed)
        F_ho2, P_ho2 = evidence(org, X_syn[ho])
        bta = (0.0, None)
        for th in np.linspace(0.3, 0.9, 13):
            s = float((arbiter_predict(net, F_ho2, P_ho2, th) == y_syn[ho]).mean())
            if s > bta[0]:
                bta = (s, float(th))
        F_te2, P_te2 = evidence(org, Xte)
        a2 = arbiter_predict(net, F_te2, P_te2, bta[1])
        out["arbiter_strict"] = float((a2 == yte).mean())
    return out


def c_alone(C, Xte, yte):
    o = Organism()
    o.organs = [C]
    s = o.scores(Xte, yte)
    return {"strict": s["strict"], "forced": s.get("forced")}


def build(bench, seed):
    if bench.startswith("cifar_"):            # cifar_<tag> -> results/cifar100_<tag>.npz
        cfg = CIFAR_CFG
        tag = bench.split("_", 1)[1]
        splits = load_feats(seed, feats_path=os.path.join(
            RES, f"cifar100_{tag}.npz"))
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
        with open(os.path.join(RES, f"adol_org_{tag}_{seed}.pkl"), "rb") as fh:
            org = pickle.load(fh)
    else:
        cfg = MLP_CFG[bench]
        X, y, _ = load_real(bench)
        splits = prepare(X, y, seed=seed)
        tasks = split_tasks(y, 2)
        torch.manual_seed(seed)
        org, _ = run_ours(tasks, splits, X.shape[1], cfg, seed)
    return org, splits, tasks, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+",
                    default=["digits", "kmnist", "cifar_vitb"])
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pricing", default="self", choices=["self", "pool"])
    ap.add_argument("--out-name", default="market.json")
    ap.add_argument("--force-c", action="store_true",
                    help="expose a gate-REFUSED consolidated organ as a price "
                         "reference only (never trusted as an answerer)")
    args = ap.parse_args()

    out = {}
    for bench in args.benchmarks:
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            org, splits, tasks, cfg = build(bench, seed)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
            cascade_strict = org.scores(Xte, yte)["strict"]
            # same auction, two pricing schemes, one arbiter, one pool
            self_raw = run_market(org, splits, seed, arbiter_too=False,
                                  pricing="self")
            pool_raw = run_market(org, splits, seed, arbiter_too=True,
                                  pricing="pool")
            last = np.isin(yva, tasks[-1])
            org2, rep = consolidate_protos(org, Xtr.shape[1], cfg, seed,
                                           Xva_last=Xva[last],
                                           yva_last=yva[last],
                                           force=args.force_c)
            self_con = pool_con = None
            have_c = rep["accepted"] or rep.get("forced", False)
            if have_c:
                self_con = run_market(org2, splits, seed, arbiter_too=False,
                                      pricing="self")
                # C sits at index 0; index the reserve to its bid
                pool_con = run_market(org2, splits, seed, arbiter_too=True,
                                      pricing="pool", c_idx=0)
            rec = {"seed": seed, "cascade": cascade_strict,
                   "raw": self_raw, "raw_pool": pool_raw,
                   "consolidated": self_con, "consolidated_pool": pool_con,
                   "C_accepted": rep["accepted"],
                   "C_forced": bool(rep.get("forced", False)),
                   "C_why": rep.get("why"),
                   # C's own accuracy as a standalone answerer: a refused C can
                   # still be a good price reference, and this is the contrast
                   "C_alone": (c_alone(org2.organs[0], Xte, yte)
                               if have_c else None)}
            recs.append(rec)
            line = (f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) cascade "
                    f"{cascade_strict:.3f} | self {self_raw['market_strict']:.3f}"
                    f" (brier {self_raw['pool_brier']:.3f}) | maker "
                    f"{pool_raw['market_strict']:.3f} (brier "
                    f"{pool_raw['pool_brier']:.3f}) | arbiter "
                    f"{pool_raw['arbiter_strict']:.3f}")
            if pool_con:
                tag = "+C" if rep["accepted"] else "+C(FORCED)"
                line += (f" | {tag} self {self_con['market_strict']:.3f} maker "
                         f"{pool_con['market_strict']:.3f} arbiter "
                         f"{pool_con['arbiter_strict']:.3f}")
                ic = pool_con.get("indexed_c")
                ia = pool_con.get("indexed_abstain")
                if ic:
                    line += (f" | INDEXED c {ic['strict']:.3f} "
                             f"(lam {ic['lam']:.2f}, spec-win "
                             f"{ic['spec_win_rate']:.2f}) abstain "
                             f"{ia['strict']:.3f} (cov {ia['coverage']:.2f})"
                             f" | C alone {rec['C_alone']['strict']:.3f}")
            print(line, flush=True)
            # write after EVERY seed: long batteries get torn down mid-run
            out[bench] = recs
            with open(os.path.join(RES, args.out_name), "w") as f:
                json.dump(out, f, indent=1, default=float)
        out[bench] = recs
        with open(os.path.join(RES, args.out_name), "w") as f:
            json.dump(out, f, indent=1, default=float)
    print(f"wrote results/{args.out_name}")


if __name__ == "__main__":
    main()
