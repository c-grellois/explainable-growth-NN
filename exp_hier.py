"""B-hier -- coarse-to-fine growth: containers first, refinement inside them.

A curriculum in the developmental sense the user named: learning starts with
basic concepts ("containers") and refinement is introduced later. CIFAR-100
supplies the hierarchy (20 superclasses x 5 fine classes). This is a DIFFERENT
PROTOCOL from flat class-IL and is never placed in the same table as the
random-order anchors; its internal references are the flat cascade under the
superclass-GROUPED order (same information order, no containers; exp_lifecycle
--order-policy grouped), the order-invariant contracted prototypes / NCM at the
fine level, and joint bounds.

Stage A  containers: the 20 superclasses arrive as `--container-tasks` tasks
         (default 4 x 5), each a frozen organ with a contract, grown by need
         and tightened by vetoes exactly as in the flat protocol (run_ours on
         coarse labels).
Stage B  refinement: superclasses are refined `--refine-inc` at a time (default
         2 -> 10 refinement tasks). For each superclass s a CHILD organ over its
         5 fine classes is trained ONLY on the points the container organism
         routes to s (the child's territory is nested in the parent's by
         routing), with its own contract solved on the same routed validation
         points. Containers are never touched; children are frozen at birth.

Inference (hierarchical, first-accept at each level):
  container cascade -> coarse answer c (or abstain)
  if c has a child: child -> fine answer (or abstain -> GRADED answer: the
  coarse label only)
Metrics on the test set after every refinement task and at the end:
  fine_strict     fine answer correct (abstention or coarse-only = error)
  fine_forced     nearest-anchor fallback at both levels
  coarse_strict   container answer correct
  graded          fractions answered at fine level / coarse only / abstained,
                  and coarse accuracy of the coarse-only answers
  fine_flips      fine answers on already-refined superclasses that CHANGED when
                  a later superclass was refined (must be 0: frozen children,
                  untouched containers)

Run:  python exp_hier.py [--first-seed 2 --seeds 3] [--tag vitb]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.organism import Organism, ABSTAIN
from cgrow.organ import Organ
from cgrow.curriculum import cifar_superclass_map, cifar_superclass_names


def feature_containers(Xtr, ytr, n_groups=20, size=5, seed=0):
    """Group fine classes into balanced containers by their feature-space class
    means: k-means on the L2-normalized means, then a greedy balanced assignment
    (each class to the nearest centroid with room). Returns {fine: container}."""
    from sklearn.cluster import KMeans
    n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xn = n(np.asarray(Xtr, dtype=np.float64))
    classes = sorted(int(c) for c in np.unique(ytr))
    M = n(np.stack([Xn[ytr == c].mean(axis=0) for c in classes]))
    km = KMeans(n_clusters=n_groups, n_init=10, random_state=seed).fit(M)
    D = ((M[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
    # greedy balanced: process (class, centroid) pairs by increasing distance
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    room = [size] * n_groups
    assign = {}
    for ci, gi in order:
        c = classes[ci]
        if c in assign or room[gi] == 0:
            continue
        assign[c] = int(gi); room[gi] -= 1
        if len(assign) == len(classes):
            break
    return assign
from exp_continual import run_ours
from exp_cifar import load_feats, CFG

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def child_for(container_org, s, fine_classes, Xtr, ytr_f, ytr_c, Xva, yva_f,
              yva_c, cfg, seed, in_dim):
    """Grow one child organ for superclass s on the points the container
    organism routes to s. Returns (Organism with one organ or None, report)."""
    r_tr = container_org.predict(Xtr)["pred"]
    r_va = container_org.predict(Xva)["pred"]
    m_tr = (r_tr == s) & (ytr_c == s)          # routed to s and truly s
    m_va = (r_va == s) & (yva_c == s)
    rep = {"superclass": int(s), "fine_classes": [int(c) for c in fine_classes],
           "n_train_routed": int(m_tr.sum()), "n_val_routed": int(m_va.sum()),
           "container_recall_train": float((r_tr[ytr_c == s] == s).mean())}
    if m_tr.sum() < 5 * len(fine_classes) or m_va.sum() < len(fine_classes):
        rep["why"] = "too few routed samples"
        return None, rep
    child = Organism()
    cand = Organ(in_dim, cfg["feat"], cfg["hidden"], cap=cfg["cap"],
                 class_ids=list(fine_classes), seed=seed * 100 + 50 + int(s),
                 k_centroids=cfg.get("k_centroids", 5),
                 q_cov_in=cfg.get("q_cov_in", 0.995), prox=cfg.get("prox", "kmeans"))
    cand.train_proto(Xtr[m_tr], ytr_f[m_tr], epochs=cfg["epochs"])
    grew = child.try_grow(cand, Xva[m_va], yva_f[m_va], p_target=cfg["p_target"],
                          gate=False, label=f"child{s}")
    rep["grew"] = bool(grew)
    if not grew:
        rep["why"] = child.log[-1].get("why")
        return None, rep
    g = child.log[-1]
    rep.update({k: g[k] for k in ("cov_on_declined", "prec_on_declined", "rho", "tau")})
    rep["n_params"] = child.n_params()
    return child, rep


def proto_child_for(container_org, s, fine_classes, Xtr, ytr_f, ytr_c, Xva, yva_f,
                    yva_c, cfg, seed, in_dim):
    """Prototype child: per fine class an anchor (mean of L2-normalized routed
    points) and a PCA territory, thresholds solved on the routed validation
    points. Closed form, no trunk (the contracted-prototype topology at the
    child level)."""
    from cgrow.adolescence import ProtoOrgan
    from cgrow.proximity import PCAProx
    r_tr = container_org.predict(Xtr)["pred"]
    r_va = container_org.predict(Xva)["pred"]
    m_tr = (r_tr == s) & (ytr_c == s)
    m_va = (r_va == s) & (yva_c == s)
    rep = {"superclass": int(s), "fine_classes": [int(c) for c in fine_classes],
           "n_train_routed": int(m_tr.sum()), "n_val_routed": int(m_va.sum()),
           "container_recall_train": float((r_tr[ytr_c == s] == s).mean()),
           "child": "proto"}
    if m_tr.sum() < 5 * len(fine_classes) or m_va.sum() < len(fine_classes):
        rep["why"] = "too few routed samples"
        return None, rep
    X, y = Xtr[m_tr], ytr_f[m_tr]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    present = [c for c in fine_classes if (y == c).sum() >= 3]
    if len(present) < 2:
        rep["why"] = "fewer than two fine classes with >= 3 routed samples"
        return None, rep
    anchors = np.stack([Xn[y == c].mean(axis=0) for c in present])
    anchors /= (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    C = ProtoOrgan(in_dim, present, anchors,
                   min_norm=float(np.linalg.norm(X, axis=1).min()), seed=seed * 100 + 50 + int(s))
    keep = np.isin(y, present)
    C.prox = PCAProx(n_components=24, seed=seed).fit(X[keep], y[keep])
    C.prox_name = "pca"
    child = Organism()
    grew = child.try_grow(C, Xva[m_va], yva_f[m_va], p_target=cfg["p_target"],
                          gate=False, label=f"protochild{s}")
    rep["grew"] = bool(grew)
    if not grew:
        rep["why"] = child.log[-1].get("why")
        return None, rep
    g = child.log[-1]
    rep.update({k: g[k] for k in ("cov_on_declined", "prec_on_declined", "rho", "tau")})
    rep["n_params"] = child.n_params()
    rep["classes_dropped"] = [int(c) for c in fine_classes if c not in present]
    return child, rep


def _proto_organ(X, y, classes, seed, in_dim):
    from cgrow.adolescence import ProtoOrgan
    from cgrow.proximity import PCAProx
    present = [c for c in classes if (y == c).sum() >= 3]
    if len(present) < 1:
        return None
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    anchors = np.stack([Xn[y == c].mean(axis=0) for c in present])
    anchors /= (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    C = ProtoOrgan(in_dim, present, anchors, min_norm=float(np.linalg.norm(X, axis=1).min()), seed=seed)
    keep = np.isin(y, present)
    C.prox = PCAProx(n_components=24, seed=seed).fit(X[keep], y[keep])
    C.prox_name = "pca"
    return C


def proto_two_step_child_for(container_org, s, fine_classes, Xtr, ytr_f, ytr_c, Xva, yva_f,
                             yva_c, Xte, yte_f, cfg, seed, in_dim, split=(3, 2)):
    """Two-step refinement with PROTOTYPE organs: step 1 a prototype organ over
    the first 3 fine classes; step 2 tighten it against the 2 new classes'
    routed points (monotone vetoes), grow a second prototype organ on what it
    declines, and give the newborn birth vetoes against samples from the
    first organ's territory. Fine answers already given cannot change (checked
    on the routed test points)."""
    r_tr = container_org.predict(Xtr)["pred"]
    r_va = container_org.predict(Xva)["pred"]
    r_te = container_org.predict(Xte)["pred"]
    m_tr = (r_tr == s) & (ytr_c == s)
    m_va = (r_va == s) & (yva_c == s)
    m_te = r_te == s
    rep = {"superclass": int(s), "fine_classes": [int(c) for c in fine_classes],
           "n_train_routed": int(m_tr.sum()), "n_val_routed": int(m_va.sum()),
           "container_recall_train": float((r_tr[ytr_c == s] == s).mean()),
           "child": "proto-two-step"}
    if m_tr.sum() < 5 * len(fine_classes) or m_va.sum() < len(fine_classes) or m_te.sum() == 0:
        rep["why"] = "too few routed samples"
        return None, rep
    fc = sorted(int(c) for c in fine_classes)
    steps = [fc[:split[0]], fc[split[0]:]]
    Xr, yr, Xv, yv, Xt = Xtr[m_tr], ytr_f[m_tr], Xva[m_va], yva_f[m_va], Xte[m_te]
    child = Organism()
    vetoes = 0; flips = 0; fa = []
    tighten_abstain = 0; tighten_flips = 0
    for k, cls in enumerate(steps):
        mtr, mva = np.isin(yr, cls), np.isin(yv, cls)
        if k > 0:
            before = child.predict(Xt)["pred"]
            fa0 = float((child.predict(Xr[mtr])["pred"] != ABSTAIN).mean())
            out = child.tighten_against(Xr[mtr], self_budget=0.0, veto_theta_q=cfg["veto_theta_q"])
            vetoes += int(sum(o[2] for o in out))
            fa1 = float((child.predict(Xr[mtr])["pred"] != ABSTAIN).mean())
            fa.append((fa0, fa1))
            after = child.predict(Xt)["pred"]
            was = before != ABSTAIN
            # tightening is monotone: an answer may become an abstention (logged
            # as such, exactly like the flat protocol's tighten_flips) but must
            # never become a different answer
            tighten_abstain += int((was & (after == ABSTAIN)).sum())
            tighten_flips += int((was & (after != ABSTAIN) & (after != before)).sum())
            need = child.declined_mask(Xr[mtr])
        else:
            need = np.ones(int(mtr.sum()), bool)
        if need.sum() < 3:
            continue
        C = _proto_organ(Xr[mtr][need], yr[mtr][need], cls, seed * 100 + 50 + int(s) + k, in_dim)
        if C is None:
            continue
        p_before = child.predict(Xt)["pred"] if child.organs else None
        grew = child.try_grow(C, Xv[mva], yv[mva], p_target=cfg["p_target"], gate=False,
                              label=f"protostep{k}")
        if grew and k > 0:
            rng = np.random.default_rng(seed * 10 + int(s))
            Xneg = np.concatenate([o.prox.sample(1000, rng)[0] for o in child.organs[:-1]])
            vetoes += child.organs[-1].add_veto(Xneg, theta_q=cfg["veto_theta_q"],
                                                seed=seed * 100 + 70 + int(s),
                                                X_self=Xv[mva], max_self_loss=0.03)
        if p_before is not None:
            p_after = child.predict(Xt)["pred"]
            was = p_before != ABSTAIN
            flips += int((was & (p_after != p_before)).sum())
    rep.update({"grew": len(child.organs) > 0, "n_organs": len(child.organs), "steps": steps,
                "within_growth_flips": flips, "within_vetoes": vetoes,
                "within_tighten_abstentions": tighten_abstain,
                "within_tighten_answer_changes": tighten_flips,
                "within_fa_before_after": fa, "n_params": child.n_params()})
    if not child.organs:
        rep["why"] = "no organ grew"
        return None, rep
    return child, rep


def two_step_child_for(container_org, s, fine_classes, Xtr, ytr_f, ytr_c, Xva, yva_f,
                       yva_c, Xte, yte_f, cfg, seed, in_dim, split=(3, 2)):
    """Two-step refinement inside a container: the fine classes arrive in two
    tasks (3 then 2). The child is a mini-organism grown by run_ours on the
    routed points, so the second step exercises the usual within-container
    inhibition (tightening vetoes on the first child organ, birth vetoes on the
    second) and its growth flips are checked."""
    r_tr = container_org.predict(Xtr)["pred"]
    r_va = container_org.predict(Xva)["pred"]
    r_te = container_org.predict(Xte)["pred"]
    m_tr = (r_tr == s) & (ytr_c == s)
    m_va = (r_va == s) & (yva_c == s)
    m_te = r_te == s
    rep = {"superclass": int(s), "fine_classes": [int(c) for c in fine_classes],
           "n_train_routed": int(m_tr.sum()), "n_val_routed": int(m_va.sum()),
           "container_recall_train": float((r_tr[ytr_c == s] == s).mean()),
           "child": "two-step"}
    if m_tr.sum() < 5 * len(fine_classes) or m_va.sum() < len(fine_classes) or m_te.sum() == 0:
        rep["why"] = "too few routed samples"
        return None, rep
    fc = sorted(int(c) for c in fine_classes)
    tasks = [fc[:split[0]], fc[split[0]:]]
    sub = ((Xtr[m_tr], ytr_f[m_tr]), (Xva[m_va], yva_f[m_va]), (Xte[m_te], yte_f[m_te]))
    torch.manual_seed(seed * 100 + 50 + int(s))
    child, r = run_ours(tasks, sub, in_dim, cfg, seed * 100 + 50 + int(s))
    rep.update({"grew": len(child.organs) > 0, "n_organs": len(child.organs),
                "steps": tasks, "within_growth_flips": r["growth_flips"],
                "within_vetoes": int(sum(e.get("vetoes_added", 0) + e.get("vetoes", 0)
                                         for e in child.log if e.get("event") in ("tighten", "birth_veto"))),
                "within_fa_before_after": [(x["fa_before"], x["fa_after"]) for x in r["false_accepts"]],
                "n_params": child.n_params()})
    if not child.organs:
        rep["why"] = "no organ grew"
        return None, rep
    return child, rep


def hier_predict(container_org, children, X):
    r = container_org.predict(X)
    coarse, coarse_forced = r["pred"], r["forced"]
    n = len(X)
    fine = np.full(n, ABSTAIN, dtype=np.int64)
    fine_forced = np.full(n, ABSTAIN, dtype=np.int64)
    for s, child in children.items():
        m = coarse == s
        if m.any():
            rc = child.predict(X[m])
            fine[m] = rc["pred"]
            fine_forced[m] = rc["forced"]
        # forced path: containers that abstained but whose forced coarse is s
        mf = (coarse == ABSTAIN) & (coarse_forced == s)
        if mf.any():
            fine_forced[mf] = child.predict(X[mf])["forced"]
    return coarse, coarse_forced, fine, fine_forced


def hier_scores(container_org, children, X, yf, yc, refined):
    coarse, coarse_forced, fine, fine_forced = hier_predict(container_org, children, X)
    seen = np.isin(yc, list(refined)) if refined else np.zeros(len(X), bool)
    out = {
        "coarse_strict": float((coarse == yc).mean()),
        "coarse_forced": float((coarse_forced == yc).mean()),
        "coarse_coverage": float((coarse != ABSTAIN).mean()),
        "fine_strict": float((fine == yf).mean()),
        "fine_forced": float((fine_forced == yf).mean()),
        "answered_fine": float((fine != ABSTAIN).mean()),
        "answered_coarse_only": float(((fine == ABSTAIN) & (coarse != ABSTAIN)).mean()),
        "abstained": float((coarse == ABSTAIN).mean()),
        "fine_precision": float((fine[fine != ABSTAIN] == yf[fine != ABSTAIN]).mean())
                          if (fine != ABSTAIN).any() else 0.0,
    }
    co = (fine == ABSTAIN) & (coarse != ABSTAIN)
    out["coarse_only_coarse_acc"] = float((coarse[co] == yc[co]).mean()) if co.any() else 0.0
    if seen.any():
        out["fine_strict_on_refined"] = float((fine[seen] == yf[seen]).mean())
        out["fine_forced_on_refined"] = float((fine_forced[seen] == yf[seen]).mean())
    return out, fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="vitb")
    ap.add_argument("--container-tasks", type=int, default=4)
    ap.add_argument("--refine-inc", type=int, default=2)
    ap.add_argument("--child", default="organ", choices=["organ", "proto", "two-step", "proto-two-step"],
                    help="organ: trained trunk (default); proto: closed-form prototype "
                         "child; two-step: 3+2 fine classes via run_ours inside the container")
    ap.add_argument("--containers", default="taxonomy", choices=["taxonomy", "feature"],
                    help="taxonomy: CIFAR-100 superclasses; feature: 20 balanced "
                         "groups of 5 fine classes clustered by class mean")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    out_name = args.out_name or (f"hier_cifar_{args.tag}{'' if args.containers == 'taxonomy' else '_feat'}"
                                 f"{'' if args.child == 'organ' else '_' + args.child.replace('-', '')}.json")
    sup = cifar_superclass_map()
    names = cifar_superclass_names()
    fine_of = {s: sorted(c for c in range(100) if sup[c] == s) for s in range(20)}
    feats = os.path.join(RES, f"cifar100_{args.tag}.npz")
    records = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.time()
        splits = load_feats(seed, feats_path=feats)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
        if args.containers == "feature":
            sup = feature_containers(Xtr, ytr, seed=seed)
            fine_of = {s: sorted(c for c in range(100) if sup[c] == s) for s in range(20)}
            names = [f"group {s}" for s in range(20)]
        to_c = np.vectorize(sup.get)
        ytr_c, yva_c, yte_c = to_c(ytr), to_c(yva), to_c(yte)
        in_dim = Xtr.shape[1]
        # ---- stage A: containers, incrementally, by need ---------------
        order_c = np.random.default_rng(seed + 400).permutation(20)
        per = int(np.ceil(20 / args.container_tasks))
        tasks_c = [sorted(int(s) for s in order_c[i:i + per]) for i in range(0, 20, per)]
        cache = os.path.join(RES, f"hier_containers_{args.tag}_{args.containers}_s{seed}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as fh:
                cont, rec_c = pickle.load(fh)
            print(f"[hier seed {seed}] containers loaded from {os.path.basename(cache)}", flush=True)
        else:
            torch.manual_seed(seed)
            cont, rec_c = run_ours(tasks_c, ((Xtr, ytr_c), (Xva, yva_c), (Xte, yte_c)),
                                   in_dim, CFG, seed)
            with open(cache, "wb") as fh:
                pickle.dump((cont, rec_c), fh)
        vet_c = int(sum(e.get("vetoes_added", 0) + e.get("vetoes", 0) for e in cont.log
                        if e.get("event") in ("tighten", "birth_veto")))
        stageA = {"tasks": tasks_c, "growth_flips": rec_c["growth_flips"],
                  "vetoes_total": vet_c, "n_params": cont.n_params(),
                  **{f"coarse_{k}": v for k, v in rec_c["final"].items()}}
        print(f"[hier seed {seed}] containers: coarse strict {rec_c['final']['strict']:.3f} "
              f"forced {rec_c['final']['forced']:.3f} cov {rec_c['final']['coverage']:.3f} "
              f"| vetoes {vet_c} | flips {rec_c['growth_flips']}", flush=True)
        # ---- stage B: refinement, `refine_inc` superclasses per task ---
        children, reports, curve, refined = {}, [], [], []
        prev_fine = None
        fine_flips = 0
        order_r = [int(s) for s in order_c]           # refine in arrival order
        for i in range(0, 20, args.refine_inc):
            batch = order_r[i:i + args.refine_inc]
            for s in batch:
                if args.child == "proto":
                    child, rep = proto_child_for(cont, s, fine_of[s], Xtr, ytr, ytr_c, Xva, yva,
                                                 yva_c, CFG, seed, in_dim)
                elif args.child == "two-step":
                    child, rep = two_step_child_for(cont, s, fine_of[s], Xtr, ytr, ytr_c, Xva, yva,
                                                    yva_c, Xte, yte, CFG, seed, in_dim)
                elif args.child == "proto-two-step":
                    child, rep = proto_two_step_child_for(cont, s, fine_of[s], Xtr, ytr, ytr_c, Xva, yva,
                                                          yva_c, Xte, yte, CFG, seed, in_dim)
                else:
                    child, rep = child_for(cont, s, fine_of[s], Xtr, ytr, ytr_c, Xva, yva,
                                           yva_c, CFG, seed, in_dim)
                rep["name"] = names[s]
                reports.append(rep)
                if child is not None:
                    children[s] = child
                refined.append(s)
            sc, fine_now = hier_scores(cont, children, Xte, yte, yte_c, refined)
            if prev_fine is not None:
                # answers already given at the fine level must not change
                was = prev_fine != ABSTAIN
                fine_flips += int((was & (fine_now != prev_fine)).sum())
            prev_fine = fine_now
            sc["refined"] = list(refined)
            curve.append(sc)
            print(f"  refine {i // args.refine_inc + 1}/{20 // args.refine_inc}: "
                  f"fine strict {sc['fine_strict']:.3f} (on refined {sc.get('fine_strict_on_refined', 0):.3f}) "
                  f"forced {sc['fine_forced']:.3f} | answered fine {sc['answered_fine']:.2f} "
                  f"coarse-only {sc['answered_coarse_only']:.2f} abstain {sc['abstained']:.2f} "
                  f"| fine flips so far {fine_flips}", flush=True)
        final = curve[-1]
        n_children = len(children)
        rec = {"seed": seed, "containers": args.containers, "child": args.child,
               "within_vetoes_total": int(sum(r.get("within_vetoes", 0) for r in reports)),
               "within_growth_flips_total": int(sum(r.get("within_growth_flips", 0) for r in reports)),
               "container_members": {int(s): fine_of[s] for s in range(20)},
               "stageA": stageA, "children": reports,
               "n_children": n_children, "children_params": int(sum(c.n_params() for c in children.values())),
               "fine_flips": fine_flips, "curve": curve, "final": final,
               "elapsed_s": time.time() - t0}
        records.append(rec)
        print(f"[hier seed {seed}] ({time.time()-t0:.0f}s) FINAL fine strict {final['fine_strict']:.3f} "
              f"forced {final['fine_forced']:.3f} | coarse strict {final['coarse_strict']:.3f} "
              f"| graded: fine {final['answered_fine']:.2f} coarse-only {final['answered_coarse_only']:.2f} "
              f"(coarse acc {final['coarse_only_coarse_acc']:.2f}) abstain {final['abstained']:.2f} "
              f"| children {n_children}/20 | fine flips {fine_flips}", flush=True)
        with open(os.path.join(RES, out_name), "w") as f:
            json.dump(records, f, indent=1, default=float)
    print("wrote", os.path.join(RES, out_name))


if __name__ == "__main__":
    main()
