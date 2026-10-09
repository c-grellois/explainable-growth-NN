"""B-sleep -- nightly micro-consolidations instead of one adolescence.

Biology consolidates nightly, not once: sleep replays the day into cortex
(systems consolidation) and rehearses control. The organism's version, using
only what it owns:

  day    the usual wake phase: tighten (vetoes) -> grow the task's organ ->
         birth-veto. Cascade organs frozen at birth, as always.
  sleep  rebuild the consolidated organ C from ALL current contracts, in
         closed form (prototype anchors from contract samples; merged
         veto-capable territory). Because C is rebuilt every night it can
         NEVER suffer the unseen-classes channel that broke mid-life
         adolescence -- each night's C knows every class seen so far.
  dream  (--dream) refit the prefrontal arbiter nightly on the contract pool.

Every night is gated (the relative per-class gate vs the current organism +
the day's real validation) and audited (answer flips logged); a refused night
leaves the organism unchanged -- sleep deprivation is safe. Contract samples
are always drawn from the ORIGINAL per-task organs, never from C, so there is
no generational synthetic-of-synthetic drift.

Measured: the ANYTIME curve (strict accuracy on seen classes after each day
and each night) against the one-shot lifecycle's endpoint, plus per-night
audit totals. Hyperparameters are inherited unchanged from the frozen
adolescence/arbiter configs; run once on evaluation seeds 2-4.

Run:  python exp_sleep.py [--benchmarks digits kmnist cifar_vitb] [--dream]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import load_real, prepare, split_tasks
from cgrow.organ import Organ
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import (ProtoOrgan, _gate_synthetic, _strict_per_class,
                               sample_pool)
from cgrow.proximity import merge_pca_family
from exp_continual import CFG as MLP_CFG
from exp_cifar import load_feats, CFG as CIFAR_CFG
from exp_pfc import evidence, train_arbiter, arbiter_predict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


from cgrow.gates import _mcnemar, DriftLedger
from exp_arbiter_source import who_and_pred

# the territory-respecting arbiter is the certified lifecycle's mechanism;
# --gated switches every arbiter call in this file to it
ARB_GATED = False


def _arb(net, F, P, theta):
    if ARB_GATED:
        return who_and_pred(net, F, P, theta, gated=True)[0]
    return arbiter_predict(net, F, P, theta)


def escrow_verdict(pred_new, pred_ref, y, ledger, name):
    """The frozen escrow check (McNemar z >= -1.645 on the whole validation
    split) plus the LIFETIME budget: a night that passes its own test is still
    refused if accepting it would push cumulative accepted drift below
    -budget. Returns (ok, refused_by, b, c, z)."""
    b, c, z = _mcnemar(pred_new, pred_ref, y)
    if z < -1.645:
        return False, "escrow", b, c, z
    if ledger is not None and ledger.would_breach(b, c, len(y)):
        return False, "ledger", b, c, z
    if ledger is not None:
        ledger.record(name, b, c, len(y))
    return True, None, b, c, z


def day_phase(cascade: Organism, C, cls, splits, cfg, seed, t):
    """Wake: tighten cascade organs, grow the task's organ, birth-veto.
    Routing during the day includes the current C (if any) in front."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    trm, vam = np.isin(ytr, cls), np.isin(yva, cls)
    Xt_tr, yt_tr = Xtr[trm], ytr[trm]
    current = view(cascade, C)
    if cascade.organs:
        current.tighten_against(Xt_tr, self_budget=0.0,
                                veto_theta_q=cfg["veto_theta_q"])
    need = (current.declined_mask(Xt_tr) if cascade.organs
            else np.ones(len(Xt_tr), bool))
    cand = Organ(Xtr.shape[1], cfg["feat"], cfg["hidden"], cap=cfg["cap"],
                 class_ids=cls, seed=seed * 100 + t,
                 q_cov_in=cfg["q_cov_in"], prox=cfg["prox"])
    cand.train_proto(Xt_tr[need], yt_tr[need], epochs=cfg["epochs"])
    cascade.try_grow(cand, Xva[vam], yva[vam], p_target=cfg["p_target"],
                     gate=(t > 0), label=f"task{t}")
    if t > 0 and cascade.organs:
        rng = np.random.default_rng(seed * 10 + t)
        Xneg = np.concatenate([o.prox.sample(2000, rng)[0]
                               for o in cascade.organs[:-1]])
        cascade.organs[-1].add_veto(Xneg, theta_q=cfg["veto_theta_q"],
                                    seed=seed * 100 + 50 + t,
                                    X_self=Xva[vam], max_self_loss=0.03)


def view(cascade: Organism, C, arbiter=None):
    org = Organism()
    org.organs = ([C] + list(cascade.organs)) if C is not None \
        else list(cascade.organs)
    return org


def scores_with(cascade, C, arbiter, theta, X, y):
    org = view(cascade, C)
    if arbiter is None:
        return org.scores(X, y)["strict"], org
    F, P = evidence(org, X)
    ans = _arb(arbiter, F, P, theta)
    return float((ans == y).mean()), org


def sleep_phase(cascade: Organism, C_prev, splits, cfg, seed, t,
                dream: bool, arb_prev, ledger=None, escrow=None):
    """Night: rebuild C closed-form from all contracts; optionally refit the
    arbiter. Both gated relative to the current organism; audited."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    rng = np.random.default_rng(seed * 31 + 5 + 977 * t)
    X_syn, y_syn = sample_pool(cascade, 2500, rng)
    perm = rng.permutation(len(X_syn))
    ho, tr = perm[: len(X_syn) // 5], perm[len(X_syn) // 5:]
    classes = sorted(set(int(c) for o in cascade.organs for c in o.class_ids))
    last_cls = cascade.organs[-1].class_ids
    vam = np.isin(yva, last_cls)

    Xn = X_syn[tr] / (np.linalg.norm(X_syn[tr], axis=1, keepdims=True) + 1e-9)
    anchors = np.stack([Xn[y_syn[tr] == c].mean(axis=0) for c in classes])
    anchors /= (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    C = ProtoOrgan(Xtr.shape[1], classes, anchors,
                   min_norm=float(np.linalg.norm(X_syn[tr], axis=1).min()),
                   seed=seed)
    C.prox = merge_pca_family([o.prox for o in cascade.organs])
    C.prox_name = C.prox.name
    solved = C.solve_thresholds(X_syn[ho], y_syn[ho],
                                p_target=cfg.get("p_target", 0.9), require=True)
    night = {"task": t, "sleep_accepted": False, "dream_accepted": False,
             "flips": 0}
    current = view(cascade, C_prev)
    if solved is not None:
        C.freeze(X_syn[ho], y_syn[ho])
        ok, _ = _gate_synthetic(C, X_syn[ho], y_syn[ho], classes, cfg,
                                ref_org=current)
        if ok:
            cand = view(cascade, C)
            old_s = current.scores(Xva[vam], yva[vam])["strict"]
            new_s = cand.scores(Xva[vam], yva[vam])["strict"]
            if new_s >= old_s - cfg.get("adol_val_slack", 0.05):
                ok_e = True
                if escrow is not None:
                    Xe, ye = escrow
                    ok_e, why, b, c, z = escrow_verdict(
                        cand.predict(Xe)["pred"], current.predict(Xe)["pred"],
                        ye, ledger, f"sleep t{t}")
                    night["sleep_escrow_z"] = z
                    if not ok_e:
                        night["sleep_refused_by"] = why
                if ok_e:
                    p0 = current.predict(Xte)["pred"]
                    p1 = cand.predict(Xte)["pred"]
                    was = p0 != ABSTAIN
                    night["flips"] = int((was & (p1 != p0)).sum())
                    night["sleep_accepted"] = True
                    C_prev = C
    C_out = C_prev

    arb, theta = arb_prev
    if dream and len(cascade.organs) >= 2:
        org2 = view(cascade, C_out)
        F_tr, P_tr = evidence(org2, X_syn[tr])
        net = train_arbiter(F_tr, P_tr == y_syn[tr][None, :], seed + t)
        F_ho, P_ho = evidence(org2, X_syn[ho])
        best = (0.0, None)
        for th in np.linspace(0.3, 0.9, 13):
            s = float((_arb(net, F_ho, P_ho, th) == y_syn[ho]).mean())
            if s > best[0]:
                best = (s, float(th))
        a_ho = _arb(net, F_ho, P_ho, best[1])
        cls_arr = np.asarray(classes)
        pc_new = _strict_per_class(a_ho, a_ho != ABSTAIN, y_syn[ho], cls_arr)
        r_ref = org2.predict(X_syn[ho])
        pc_ref = _strict_per_class(r_ref["pred"], r_ref["pred"] != ABSTAIN,
                                   y_syn[ho], cls_arr)
        F_v, P_v = evidence(org2, Xva[vam])
        v_new = float((_arb(net, F_v, P_v, best[1]) == yva[vam]).mean())
        v_ref = org2.scores(Xva[vam], yva[vam])["strict"]
        if (pc_new.mean() >= pc_ref.mean() + 0.01
                and pc_new.min() >= pc_ref.min() - 0.05 and v_new >= v_ref):
            ok_e = True
            if escrow is not None:
                Xe, ye = escrow
                F_e, P_e = evidence(org2, Xe)
                ok_e, why, b, c, z = escrow_verdict(
                    _arb(net, F_e, P_e, best[1]), org2.predict(Xe)["pred"],
                    ye, ledger, f"dream t{t}")
                night["dream_escrow_z"] = z
                if not ok_e:
                    night["dream_refused_by"] = why
            if ok_e:
                arb, theta = net, best[1]
                night["dream_accepted"] = True
    return C_out, (arb, theta), night


def run(bench: str, seed: int, dream: bool, budget=None, use_escrow=False):
    if bench.startswith("cifar_"):            # cifar_<tag> -> results/cifar100_<tag>.npz
        cfg = CIFAR_CFG
        tag = bench.split("_", 1)[1]
        splits = load_feats(seed, feats_path=os.path.join(
            RES, f"cifar100_{tag}.npz"))
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10])
                 for i in range(10)]
    else:
        cfg = MLP_CFG[bench]
        X, y, _ = load_real(bench)
        splits = prepare(X, y, seed=seed)
        tasks = split_tasks(y, 2)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    torch.manual_seed(seed)

    cascade = Organism()
    C, arb = None, (None, None)
    curve, nights = [], []
    # the escrow is the WHOLE held-out validation split (frozen gate spec);
    # the ledger, if any, caps the organism's lifetime accepted drift on it
    escrow = (Xva, yva) if (use_escrow or budget is not None) else None
    ledger = DriftLedger(budget) if budget is not None else None
    for t, cls in enumerate(tasks):
        day_phase(cascade, C, cls, splits, cfg, seed, t)
        arb = (None, None)      # growth invalidates routing knowledge;
                                # only the next night's dream re-establishes it
        seen = np.isin(yte, [c for cl in tasks[: t + 1] for c in cl])
        s_day, _ = scores_with(cascade, C, arb[0], arb[1],
                               Xte[seen], yte[seen])
        C, arb, night = sleep_phase(cascade, C, splits, cfg, seed, t,
                                    dream, arb, ledger=ledger, escrow=escrow)
        s_night, _ = scores_with(cascade, C, arb[0], arb[1],
                                 Xte[seen], yte[seen])
        curve.append({"task": t, "day": s_day, "night": s_night})
        nights.append(night)
    final, _ = scores_with(cascade, C, arb[0], arb[1], Xte, yte)
    refused = [n.get("sleep_refused_by") for n in nights] + \
              [n.get("dream_refused_by") for n in nights]
    return {"seed": seed, "curve": curve, "nights": nights, "final": final,
            "sleep_accepted": sum(n["sleep_accepted"] for n in nights),
            "dream_accepted": sum(n["dream_accepted"] for n in nights),
            "total_flips": sum(n["flips"] for n in nights),
            "refused_by_escrow": refused.count("escrow"),
            "refused_by_ledger": refused.count("ledger"),
            "ledger": ledger.summary() if ledger is not None else None,
            "gated_arbiter": ARB_GATED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+",
                    default=["digits", "kmnist", "cifar_vitb"])
    ap.add_argument("--dream", action="store_true")
    ap.add_argument("--gated", action="store_true",
                    help="territory-respecting arbiter for the dreams")
    ap.add_argument("--escrow", action="store_true",
                    help="add the frozen escrow check to every night's gate")
    ap.add_argument("--budget", type=float, default=None,
                    help="LIFETIME accepted-drift budget on the escrow "
                         "(implies --escrow); preregistered value 0.02")
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    out = {}
    global ARB_GATED
    ARB_GATED = bool(args.gated)
    suffix = (("_dream" if args.dream else "") + ("_gated" if args.gated else "")
              + ("_escrow" if (args.escrow and args.budget is None) else "")
              + (f"_budget{args.budget}" if args.budget is not None else ""))
    path = os.path.join(RES, f"sleep{suffix}.json")
    if os.path.exists(path):
        with open(path) as f:
            out = json.load(f)
    for bench in args.benchmarks:
        recs = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            t0 = time.time()
            r = run(bench, seed, args.dream, budget=args.budget,
                    use_escrow=args.escrow)
            if r["ledger"] is not None or args.escrow:
                led = r["ledger"]
                print(f"    escrow refusals {r['refused_by_escrow']} | ledger "
                      f"refusals {r['refused_by_ledger']}"
                      + (f" | cumulative accepted drift {led['cumulative']:+.4f}"
                         f" of budget -{led['budget']}" if led else ""),
                      flush=True)
            recs.append(r)
            arc = "  ".join(f"t{c['task']}:{c['day']:.2f}>{c['night']:.2f}"
                            for c in r["curve"])
            print(f"[{bench} seed {seed}] ({time.time()-t0:.0f}s) final "
                  f"{r['final']:.3f}  sleep {r['sleep_accepted']}/"
                  f"{len(r['nights'])}  dream {r['dream_accepted']}  "
                  f"flips {r['total_flips']}", flush=True)
            print(f"  {arc}", flush=True)
        out[bench] = recs
        with open(path, "w") as f:
            json.dump(out, f, indent=1, default=float)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
