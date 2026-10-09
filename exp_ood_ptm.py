"""B-ood -- declining as a measured capability.

The contract says every accepted answer comes from inside a satisfied
specification and that the system may decline. This turns that into the two
numbers the OOD and selective-prediction literatures report, on the same frozen
features as the benchmark runs, for every certified configuration and for the
standard same-feature baselines.

Preregistered on 2026-09-14 (REPORT.md, B-ood), before any number was seen:

  score   Every certified configuration accepts iff a conjunction of
          thresholded quantities holds. Its OOD score is the certificate
          slack  s(x) = max_i q_i(x) / theta_i  (margin enters as tau/margin),
          so  s <= 1  <=>  accept  (up to the boundary and vetoes, which are
          applied as-is). For the cascade  s = min over non-vetoing organs.
  ID      the benchmark's test split; OOD = every results/ood_*_<tag>.npz
          plus cifar10_<tag>.npz when the ID set is CIFAR-100
  metrics AUROC (OOD positive), FPR at 95 % TPR (ID kept), and the operating
          point the contract actually uses: ID coverage vs OOD accept rate.
          Selective prediction on ID: risk-coverage curve, AURC, risk at the
          contract's own coverage, all with s as the selector.
  baselines (fit on the seed's train split, scored on the same points)
          NCM-cosine (-max cos), kNN-1 on L2-normalised features (Sun 2022),
          Mahalanobis with shared covariance (Lee 2018), MSP of a joint linear
          probe (softmax response; a joint-training upper-bound comparator).
  seeds   evaluation seeds 2-4. No threshold is tuned on the OOD data.
  reading the hypothesis is that the certificate slack is at least as good an
          OOD score as NCM-cosine and that kNN / Mahalanobis may be stronger;
          the operating-point row (what the deployed contract actually
          declines) is the one the partner document can quote.

Run:  python exp_ood.py --cache cifar100_vitb --first-seed 2 --seeds 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_cifar import load_feats, CFG
from cgrow.organ import conformal_quantile
from cgrow.organism import ABSTAIN
from cgrow.proximity import PCAProx

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EPS = 1e-12


# ----------------------------------------------------------------- metrics
def auroc(s_id, s_ood):
    """AUROC of score s for OOD-positive (higher s = more OOD), rank based."""
    s = np.concatenate([s_id, s_ood]); y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]
    order = np.argsort(s, kind="mergesort"); s, y = s[order], y[order]
    # average ranks for ties
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fpr_at_tpr(s_id, s_ood, tpr=0.95):
    """Fraction of OOD accepted when the threshold keeps `tpr` of ID."""
    thr = np.quantile(s_id, tpr)
    return float((s_ood <= thr).mean())


def risk_coverage(s, correct):
    """Selective risk at every coverage when answering the lowest-s first."""
    order = np.argsort(s, kind="mergesort")
    err = (~correct[order]).astype(np.float64)
    n = len(s)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return cov, risk


def aurc(s, correct):
    cov, risk = risk_coverage(s, correct)
    return float(np.trapezoid(risk, cov))


def risk_at(s, correct, coverage):
    cov, risk = risk_coverage(s, correct)
    k = max(int(np.searchsorted(cov, coverage)), 0)
    return float(risk[min(k, len(risk) - 1)])


# ----------------------------------------------------------------- scaling
def scaler(feats_path):
    d = np.load(feats_path)
    Xtr = d["Xtr"].astype(np.float64)
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    return lambda X: np.clip((X.astype(np.float64) - lo) / span, 0.0, 1.0)


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


# ----------------------------------------------------------------- configurations
def cascade_slack(org, X):
    """Certificate slack of a first-accept cascade: min over organs of the
    organ's slack, an organ that vetoes the point contributing +inf."""
    best = np.full(len(X), np.inf)
    for o in org.organs:
        d1, m, p, d_in, vet, _ = o.evaluate(X)
        tau = max(float(o.tau), 0.0)
        s = np.maximum(d1 / max(float(o.rho), EPS), d_in / max(float(o.rho_in), EPS))
        if tau > 0:
            s = np.maximum(s, tau / np.maximum(m, EPS))
        s = np.where(vet, np.inf, s)
        best = np.minimum(best, s)
    return best


def fit_protos(splits, seed):
    (Xtr, ytr), (Xva, yva), _ = splits
    classes = np.unique(ytr)
    Xn, Xv = l2n(Xtr), l2n(Xva)
    means = np.stack([Xn[ytr == c].mean(axis=0) for c in classes])
    means = l2n(means)
    proxes = {int(c): PCAProx(n_components=24, seed=seed).fit(Xtr[ytr == c], ytr[ytr == c])
              for c in classes}

    def judge(X):
        sims = l2n(X) @ means.T
        o = np.argsort(-sims, axis=1)
        top, second = o[:, 0], o[:, 1]
        margin = sims[np.arange(len(X)), top] - sims[np.arange(len(X)), second]
        pred = classes[top]
        d_in = np.array([proxes[int(p)].score(x[None])[0] for p, x in zip(pred, X)])
        return pred, margin, d_in

    pv, mv, dv = judge(Xva)
    correct = pv == yva
    rho_in = conformal_quantile(dv[correct], 0.01)
    best = (0.0, 0.0)
    for tau in np.quantile(mv, np.linspace(0.0, 0.8, 12)):
        acc_m = (mv > tau) & (dv <= rho_in)
        if acc_m.sum() and correct[acc_m].mean() >= 0.6:
            s = float((acc_m & correct).mean())
            if s > best[0]:
                best = (s, float(tau))
    tau = best[1]

    def slack(X):
        pred, margin, d_in = judge(X)
        s = d_in / max(rho_in, EPS)
        if tau > 0:
            s = np.maximum(s, tau / np.maximum(margin, EPS))
        return pred, s, -sims_max(X)

    def sims_max(X):
        return (l2n(X) @ means.T).max(axis=1)

    return slack, judge


def fit_head(cache, seed, M):
    import exp_head
    rec = exp_head.run(cache, M, seed, return_model=True)
    m = rec["_model"]
    W, Wo, seen, proxes, tau, rho_in = m["W"], m["Wo"], m["seen"], m["proxes"], m["tau"], m["rho_in"]

    def slack(X):
        S = exp_head.project(X, W) @ Wo[:, seen]
        o = np.argsort(-S, axis=1)
        top, second = o[:, 0], o[:, 1]
        margin = S[np.arange(len(S)), top] - S[np.arange(len(S)), second]
        pred = np.asarray(seen)[top]
        d_in = np.array([proxes[int(p)].score(x[None])[0] for p, x in zip(pred, X)])
        s = d_in / max(rho_in, EPS)
        if tau > 0:
            s = np.maximum(s, tau / np.maximum(margin, EPS))
        return pred, s
    return slack


# ----------------------------------------------------------------- baselines
def fit_baselines(splits, seed):
    (Xtr, ytr), _, _ = splits
    classes = np.unique(ytr)
    Xn = l2n(Xtr)
    means = l2n(np.stack([Xn[ytr == c].mean(axis=0) for c in classes]))
    # Mahalanobis, shared covariance, on raw scaled features
    mu = np.stack([Xtr[ytr == c].mean(axis=0) for c in classes])
    Z = Xtr - mu[np.searchsorted(classes, ytr)]
    cov = Z.T @ Z / len(Z) + 1e-3 * np.eye(Xtr.shape[1])
    P = np.linalg.inv(cov)
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(max_iter=300, C=1.0, n_jobs=1).fit(Xtr, ytr)

    def scores(X):
        out = {}
        s = l2n(X) @ means.T
        out["ncm_cos"] = -s.max(axis=1)
        # kNN-1 distance on normalised features, chunked
        d = np.empty(len(X))
        Xq = l2n(X)
        for i in range(0, len(X), 2000):
            q = Xq[i:i + 2000]
            dd = 2.0 - 2.0 * (q @ Xn.T)
            d[i:i + 2000] = np.sqrt(np.maximum(dd.min(axis=1), 0.0))
        out["knn1"] = d
        md = np.empty((len(X), len(classes)))
        for j, m_ in enumerate(mu):
            D = X - m_
            md[:, j] = np.einsum("ij,jk,ik->i", D, P, D)
        out["mahalanobis"] = md.min(axis=1)
        pr = probe.predict_proba(X)
        out["probe_msp"] = -pr.max(axis=1)
        out["_probe_pred"] = probe.classes_[pr.argmax(axis=1)]
        out["_ncm_pred"] = classes[s.argmax(axis=1)]
        return out
    return scores


# ----------------------------------------------------------------- main
def run(cache, seed, M_head=2000, with_lifecycle=True, extra_ood=()):
    t0 = time.time()
    feats = os.path.join(RES, f"{cache}.npz")
    splits = load_feats(seed, feats_path=feats)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    scale = scaler(feats)
    tag = cache.split("_", 1)[1] if cache.startswith("cifar100_") else cache.split("_", 1)[1]
    # OOD sets: every ood_*_<tag>.npz, plus cifar10_<tag> for a CIFAR-100 ID set,
    # plus any explicitly named cache (e.g. the CLINC out-of-scope queries)
    ood = {}
    for p in sorted(glob.glob(os.path.join(RES, f"ood_*_{tag}.npz"))):
        name = os.path.basename(p)[len("ood_"):-len(f"_{tag}.npz")]
        ood[name] = scale(np.load(p)["Xte"])
    for stem in extra_ood:
        ood[stem.replace("ood_", "", 1)] = scale(np.load(os.path.join(RES, f"{stem}.npz"))["Xte"])
    if cache.startswith("cifar100_") and os.path.exists(os.path.join(RES, f"cifar10_{tag}.npz")):
        ood["cifar10"] = scale(np.load(os.path.join(RES, f"cifar10_{tag}.npz"))["Xte"])
    if not ood:
        raise SystemExit(f"no OOD cache for tag {tag}")

    configs = {}
    # cascade: the grown organism this seed's lifecycle used
    pk = os.path.join(RES, f"adol_org_{tag}_{seed}.pkl")
    if os.path.exists(pk):
        with open(pk, "rb") as fh:
            org = pickle.load(fh)
        configs["cascade"] = (lambda X, org=org: (org.predict(X)["pred"], cascade_slack(org, X)))
        if with_lifecycle:
            from cgrow.adolescence import consolidate_protos
            order = np.random.default_rng(seed + 400).permutation(int(ytr.max()) + 1)
            n_cls = int(ytr.max()) + 1
            inc = 10 if n_cls == 100 else int(np.load(feats)["inc"])
            last_cls = sorted(int(c) for c in order[(n_cls // inc - 1) * inc:])
            last = np.isin(yva, last_cls)
            org2, rep = consolidate_protos(org, Xtr.shape[1], CFG, seed,
                                           Xva_last=Xva[last], yva_last=yva[last],
                                           Xva=Xva, yva=yva)
            if rep["accepted"]:
                configs["lifecycle"] = (lambda X, org=org2: (org.predict(X)["pred"], cascade_slack(org, X)))
            else:
                print(f"  [seed {seed}] consolidation refused ({rep.get('why')}); lifecycle arm = cascade")
    protos_slack, _ = fit_protos(splits, seed)
    configs["protos"] = lambda X: protos_slack(X)[:2]
    head_slack = fit_head(cache, seed, M_head)
    configs["head"] = head_slack
    base = fit_baselines(splits, seed)

    out = {"seed": seed, "cache": cache, "ood_sets": {k: int(len(v)) for k, v in ood.items()},
           "n_id_test": int(len(yte)), "configs": {}, "baselines": {}}
    # --- certified configurations
    for name, fn in configs.items():
        pred_id, s_id = fn(Xte)
        acc_id = s_id <= 1.0
        if name in ("cascade", "lifecycle"):
            acc_id = pred_id != ABSTAIN         # the real operating point, vetoes included
        correct = pred_id == yte
        rec = {"id_coverage": float(acc_id.mean()),
               "id_strict": float((acc_id & correct).mean()),
               "id_precision": float(correct[acc_id].mean()) if acc_id.any() else 0.0,
               "forced_acc": float(correct.mean()),
               "aurc": aurc(s_id, correct),
               "risk_at_own_coverage": risk_at(s_id, correct, float(acc_id.mean())),
               "risk_at_80": risk_at(s_id, correct, 0.8),
               "risk_at_90": risk_at(s_id, correct, 0.9),
               "ood": {}}
        for oname, Xo in ood.items():
            pred_o, s_o = fn(Xo)
            acc_o = s_o <= 1.0
            if name in ("cascade", "lifecycle"):
                acc_o = pred_o != ABSTAIN
            rec["ood"][oname] = {"auroc": auroc(s_id, s_o),
                                 "fpr95": fpr_at_tpr(s_id, s_o, 0.95),
                                 "ood_accept_rate": float(acc_o.mean())}
        out["configs"][name] = rec
        print(f"  [{cache} seed {seed}] {name:9s} cov {rec['id_coverage']:.3f} strict {rec['id_strict']:.3f} "
              f"AURC {rec['aurc']:.4f} | " + " ".join(
                  f"{o}: AUROC {r['auroc']:.3f} FPR95 {r['fpr95']:.3f} accept {r['ood_accept_rate']:.3f}"
                  for o, r in rec["ood"].items()), flush=True)
    # --- baselines
    # 2026-09-18: the OOD sets are scored ONCE and reused across the four
    # baselines. They used to be re-scored inside the bname loop, which meant
    # four kNN passes over the whole training set per set (166k rows where 42k
    # suffice, and the bulk of the runtime). `base` is a deterministic closure
    # over the fitted means / covariance / probe, so the numbers are unchanged;
    # checked by scoring a set twice and comparing exactly before the re-run.
    b_id = base(Xte)
    b_ood = {oname: base(Xo) for oname, Xo in ood.items()}
    for bname in ("ncm_cos", "knn1", "mahalanobis", "probe_msp"):
        pred = b_id["_probe_pred"] if bname == "probe_msp" else b_id["_ncm_pred"]
        correct = pred == yte
        rec = {"forced_acc": float(correct.mean()), "aurc": aurc(b_id[bname], correct),
               "risk_at_80": risk_at(b_id[bname], correct, 0.8),
               "risk_at_90": risk_at(b_id[bname], correct, 0.9), "ood": {}}
        for oname in ood:
            b_o = b_ood[oname]
            rec["ood"][oname] = {"auroc": auroc(b_id[bname], b_o[bname]),
                                 "fpr95": fpr_at_tpr(b_id[bname], b_o[bname], 0.95)}
        out["baselines"][bname] = rec
        print(f"  [{cache} seed {seed}] {bname:11s} acc {rec['forced_acc']:.3f} AURC {rec['aurc']:.4f} | "
              + " ".join(f"{o}: AUROC {r['auroc']:.3f} FPR95 {r['fpr95']:.3f}" for o, r in rec["ood"].items()),
              flush=True)
    out["elapsed_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cifar100_vitb")
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--M", type=int, default=2000)
    ap.add_argument("--no-lifecycle", action="store_true")
    ap.add_argument("--ood", nargs="*", default=[],
                    help="extra OOD caches (stems under results/) whose names do not "
                         "follow ood_<name>_<tag>, e.g. ood_clinc_oos_mpnet")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    out = os.path.join(RES, args.out_name or f"ood_{args.cache}.json")
    recs = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        recs.append(run(args.cache, seed, args.M, not args.no_lifecycle, extra_ood=args.ood))
        with open(out, "w") as f:
            json.dump(recs, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
