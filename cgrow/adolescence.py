"""Adolescence: a one-shot, verified metamorphosis of the organism.

Development builds the organism as frozen parts under a no-silent-drift
invariant; its price is frozen error. Adolescence attacks that price with the
one resource the organism owns outright: every organ's contract is a data
generator. The recipe:

  1. SAMPLE labeled data from every organ's proximity contract (no raw data).
  2. TRAIN one consolidated network jointly on the pooled synthetic data --
     the joint problem the cascade never got to solve, including the
     cross-organ boundaries where frozen errors live. On the spatial sheet
     this is a massive recombination of local wiring: units re-placed by the
     wiring economy, lateral connections re-learned from scratch under the
     hinge cost, then cost-weighted pruned.
  3. VERIFY on a synthetic holdout (per-class) and on the last task's real
     validation data; reject the metamorphosis if verification fails.
  4. SWAP: the consolidated organ answers first; the old cascade remains as
     fallback behind it (coverage cannot drop). Freezing then resumes.

Adolescence is the single sanctioned exception to the retention invariant:
answers may change, each change is logged and counted (to-right / to-wrong on
real test data in the benchmark), and the event happens only if verified.
"""
from __future__ import annotations

import numpy as np
import torch

from .organ import Organ
from .organism import Organism, ABSTAIN


def sample_pool(org: Organism, n_per_organ: int, rng: np.random.Generator):
    """Labeled synthetic data from every organ's contract."""
    Xs, ys = [], []
    for o in org.organs:
        X, y = o.prox.sample(n_per_organ, rng)
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def consolidate_mlp(org: Organism, in_dim: int, cfg: dict, seed: int,
                    Xva_last=None, yva_last=None, n_per_organ: int = 4000,
                    prox_policy: str = "union"):
    """Metamorphosis for the MLP-ladder organism. Returns (new_org, report);
    new_org is the unchanged input organism if verification rejects.

    prox_policy decides whether the CONTRACT's assumption side changes:
      "union"  inherit the old territories verbatim (calibrated on real data)
      "refit"  renegotiate: fresh territories fitted on the synthetic pool --
               the naive contract change, expected to go vacuous (measured)
    The guarantee side (anchors, margins, thresholds) is renegotiated in both."""
    rng = np.random.default_rng(seed * 31 + 5)
    X_syn, y_syn = sample_pool(org, n_per_organ, rng)
    perm = rng.permutation(len(X_syn))
    n_hold = int(0.2 * len(X_syn))
    ho, tr = perm[:n_hold], perm[n_hold:]
    classes = sorted(set(int(c) for o in org.organs for c in o.class_ids))

    C = Organ(in_dim, cfg["feat"], cfg["hidden"], cap=cfg["cap"],
              class_ids=classes, seed=seed * 977 + 13,
              q_cov_in=cfg.get("q_cov_in", 0.99), prox=cfg.get("prox", "pca"))
    C.train_proto(X_syn[tr], y_syn[tr], epochs=cfg.get("adol_epochs", 30))
    if prox_policy == "union":
        # the consolidated organ's territory is the UNION of the parts'
        # territories, each calibrated on real data at its organ's birth --
        # merged into a first-class prox so the consolidated organ stays
        # veto-capable (tightenable against classes it has not yet seen)
        C.prox = _inherit_territory(org)
    # "refit": keep the prox train_proto fitted on the synthetic pool
    solved = C.solve_thresholds(X_syn[ho], y_syn[ho],
                                p_target=cfg.get("p_target", 0.95), require=True)
    report = {"accepted": False, "classes": classes,
              "n_synth": int(len(tr)), "params_C": C.n_params()}
    if solved is None:
        report["why"] = "no threshold meets target on synthetic holdout"
        return org, report
    C.freeze(X_syn[ho], y_syn[ho])
    report["syn_holdout_cov"], report["syn_holdout_prec"] = solved

    ok, report2 = _gate_synthetic(C, X_syn[ho], y_syn[ho], classes, cfg,
                                  ref_org=org)
    report.update(report2)
    if not ok:
        return org, report

    # real sanity on the last task's validation data (the only real data left)
    if Xva_last is not None:
        new = Organism()
        new.organs = [C] + list(org.organs)
        old_s = org.scores(Xva_last, yva_last)["strict"]
        new_s = new.scores(Xva_last, yva_last)["strict"]
        report["real_val_last_task"] = {"before": old_s, "after": new_s}
        if new_s < old_s - cfg.get("adol_val_slack", 0.05):
            report["why"] = "regressed on last task's real val"
            return org, report

    new = Organism()
    new.organs = [C] + list(org.organs)   # consolidated first, cascade as fallback
    new.log = list(org.log) + [{"event": "adolescence", **{k: v for k, v in
                                report.items() if k != "per_class_syn"}}]
    report["accepted"] = True
    return new, report


def _strict_per_class(pred, accepted, y, classes):
    """Strict per-class accuracy; NaN for a class absent from y (no evidence).
    Callers aggregate with the nan-aware helpers below so that an unobserved
    class neither counts as 0 nor poisons min/mean (2026-09-07 GPU-box fix:
    with plain min/mean a single NaN made every gate comparison False, i.e.
    silently accepted, on datasets where a class has too few samples to
    appear in the synthetic holdout)."""
    return np.array([float((accepted & (pred == c))[y == c].mean())
                     if (y == c).any() else np.nan for c in classes])


def _agg(pc):
    """(min, mean, n_observed) over the classes that have holdout evidence."""
    obs = pc[~np.isnan(pc)]
    if len(obs) == 0:
        return float("nan"), float("nan"), 0
    return float(obs.min()), float(obs.mean()), int(len(obs))


def _inherit_territory(org: Organism):
    """Merge the parts' territories into one veto-capable prox where the family
    allows it; fall back to the opaque union otherwise."""
    from .proximity import UnionProx, merge_pca_family
    try:
        return merge_pca_family([o.prox for o in org.organs])
    except AssertionError:
        return UnionProx([o.prox for o in org.organs])


def _gate_synthetic(C, X_ho, y_ho, classes, cfg, ref_org=None):
    """The synthetic-holdout gate, repaired twice by the CIFAR diagnosis.

    (1) Usefulness: overall holdout coverage must clear a floor -- a vacuously
    abstaining consolidation is 'safe' but pointless (the renegotiation lesson).
    (2) RELATIVE quality: with a reference organism, the consolidated organ must
    match it per class on the same holdout (mean within adol_mean_slack, worst
    class within adol_floor_slack). An absolute floor on the MINIMUM over C
    classes implicitly hardens with class count and rejects consolidations that
    are strictly better than what they replace; the relative gate compares like
    with like. Without a reference, the absolute floor applies as before."""
    d1, m, pred, d_in, vet, _ = C.evaluate(X_ho)
    acc_mask = C.accept(d1, m, d_in, vet)
    rep = {"syn_holdout_coverage": float(acc_mask.mean())}
    if rep["syn_holdout_coverage"] < cfg.get("adol_min_cov", 0.3):
        rep["why"] = f"vacuous: holdout coverage {rep['syn_holdout_coverage']:.3f}"
        return False, rep
    pc_C = _strict_per_class(pred, acc_mask, y_ho, classes)
    c_min, c_mean, n_obs = _agg(pc_C)
    rep["class_floor"] = c_min
    rep["class_mean"] = c_mean
    rep["classes_observed_in_holdout"] = n_obs
    rep["classes_total"] = len(classes)
    if n_obs == 0:
        rep["why"] = "no class of C appears in the synthetic holdout"
        return False, rep
    if ref_org is not None:
        r = ref_org.predict(X_ho)
        pc_ref = _strict_per_class(r["pred"], r["pred"] != ABSTAIN, y_ho, classes)
        r_min, r_mean, _ = _agg(pc_ref)
        rep["ref_class_floor"] = r_min
        rep["ref_class_mean"] = r_mean
        if c_mean < r_mean - cfg.get("adol_mean_slack", 0.02):
            rep["why"] = (f"mean regressed: {c_mean:.3f} vs "
                          f"ref {r_mean:.3f}")
            return False, rep
        if c_min < r_min - cfg.get("adol_floor_slack", 0.05):
            rep["why"] = (f"worst class regressed: {c_min:.3f} vs "
                          f"ref {r_min:.3f}")
            return False, rep
        return True, rep
    if rep["class_floor"] < cfg.get("adol_class_floor", 0.75):
        rep["why"] = f"class floor violated: {rep['class_floor']:.3f}"
        return False, rep
    return True, rep


class ProtoOrgan(Organ):
    """A consolidated organ with NO trained trunk: anchors are the L2-normalized
    class means of the pooled contract samples (closed form -- solve, never
    train), features are L2-normalized inputs, territory is the union of the
    parts' proximity contracts. The regime-appropriate consolidation when the
    feature space is already good and the class count is high."""

    def __init__(self, in_dim: int, class_ids, anchors: np.ndarray,
                 min_norm: float, seed: int = 0):
        self.class_ids = np.asarray(sorted(class_ids), dtype=np.int64)
        self.rho, self.tau, self.rho_in = np.inf, 0.0, np.inf
        self.L = None
        self.frozen = False
        self.stats = None
        self.seed = seed
        self.k_centroids = 5
        self.q_cov_in = 0.99
        self.prox_name = "union"
        self.prox = None
        self.vetoes = []
        self._min_norm = float(min_norm)
        self.anchors = torch.nn.Parameter(
            torch.as_tensor(anchors, dtype=torch.float32))

    def _features(self, X: np.ndarray) -> torch.Tensor:
        Xt = torch.as_tensor(X, dtype=torch.float32)
        return Xt / (Xt.norm(dim=1, keepdim=True) + 1e-9)

    def _lipschitz_exact(self) -> float:
        # x -> x/||x|| has Jacobian norm <= 2/||x||; bounded by the pool's
        # smallest norm (conservative, reported as a bound)
        return 2.0 / max(self._min_norm, 1e-6)

    def n_params(self) -> int:
        return self.anchors.numel()


def consolidate_protos(org: Organism, in_dim: int, cfg: dict, seed: int,
                       Xva_last=None, yva_last=None, n_per_organ: int = 4000,
                       force: bool = False, Xva=None, yva=None):
    """Metamorphosis into the prototype topology: class anchors computed in
    closed form from the pooled contract samples; territory inherited (union);
    thresholds solved on the synthetic holdout; the same verify-then-swap gate.

    Xva / yva are the WHOLE held-out validation split. When supplied (and
    cfg["adol_escrow"] is not False) the candidate additionally faces the real
    -data escrow the arbiter gate uses: one-sided McNemar over the whole split,
    Bonferroni per-class flags, 0.25 hard floor, consolidated organism against
    the grown one, over every class seen so far. Without them only the
    synthetic holdout and the last task's validation split are available, which
    is the gate as it stood before 2026-09-09 and is what the report records.

    force=True returns the candidate organism even when the gate REFUSES it,
    with report["accepted"] False and report["forced"] True. The refused C must
    never be trusted as an answerer -- it is exposed only so that experiments
    can use it as a PRICE REFERENCE, a strictly weaker use (see exp_market.py
    --force-c). A refusal that fails to solve thresholds cannot be forced: that
    C has no contract at all."""
    import torch as _t
    rng = np.random.default_rng(seed * 31 + 5)
    X_syn, y_syn = sample_pool(org, n_per_organ, rng)
    perm = rng.permutation(len(X_syn))
    n_hold = int(0.2 * len(X_syn))
    ho, tr = perm[:n_hold], perm[n_hold:]
    classes = sorted(set(int(c) for o in org.organs for c in o.class_ids))

    # 2026-09-07 GPU-box fix. A class can have NO synthetic sample: either the
    # 80/20 split left none in `tr` (rare classes), or no organ's territory
    # covers it at all (PCAProx.fit skips classes with < 3 samples, so
    # prox.sample never emits them; ImageNet-A has classes with 2 train
    # images). The old code took the mean of an empty slice (NaN anchor,
    # RuntimeWarning) and carried the NaN into the organ. Now: fall back to
    # the class's samples in the whole pool; if it has none anywhere, C does
    # not represent that class (the cascade below C still answers it) and the
    # report says so.
    Xn_all = X_syn / (np.linalg.norm(X_syn, axis=1, keepdims=True) + 1e-9)
    in_tr = np.zeros(len(X_syn), bool); in_tr[tr] = True
    kept, dropped, from_pool, anchors = [], [], [], []
    for c in classes:
        m_tr = in_tr & (y_syn == c)
        m_all = y_syn == c
        if m_tr.any():
            anchors.append(Xn_all[m_tr].mean(axis=0)); kept.append(c)
        elif m_all.any():
            anchors.append(Xn_all[m_all].mean(axis=0)); kept.append(c)
            from_pool.append(c)
        else:
            dropped.append(c)
    classes = kept
    anchors = np.stack(anchors)
    anchors /= (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    C = ProtoOrgan(in_dim, classes, anchors,
                   min_norm=float(np.linalg.norm(X_syn[tr], axis=1).min()),
                   seed=seed)
    C.prox = _inherit_territory(org)
    C.prox_name = C.prox.name
    solved = C.solve_thresholds(X_syn[ho], y_syn[ho],
                                p_target=cfg.get("p_target", 0.9), require=True)
    report = {"accepted": False, "classes": classes, "mode": "protos",
              "params_C": C.n_params(),
              "classes_dropped_no_pool": dropped,
              "classes_anchored_from_holdout": from_pool}
    if solved is None:
        report["why"] = "no threshold meets target on synthetic holdout"
        return org, report
    C.freeze(X_syn[ho], y_syn[ho])
    report["syn_holdout_cov"], report["syn_holdout_prec"] = solved
    ok, rep2 = _gate_synthetic(C, X_syn[ho], y_syn[ho], classes, cfg,
                               ref_org=org)
    report.update(rep2)

    def _forced():
        report["forced"] = True
        f = Organism(); f.organs = [C] + list(org.organs)
        return f, report

    if not ok:
        return _forced() if force else (org, report)
    if Xva_last is not None:
        new = Organism(); new.organs = [C] + list(org.organs)
        old_s = org.scores(Xva_last, yva_last)["strict"]
        new_s = new.scores(Xva_last, yva_last)["strict"]
        report["real_val_last_task"] = {"before": old_s, "after": new_s}
        if new_s < old_s - cfg.get("adol_val_slack", 0.05):
            report["why"] = "regressed on last task's real val"
            report["refused_by"] = "last_task_val"
            return _forced() if force else (org, report)

    # ---- the full real-data escrow (2026-09-09) -------------------------- #
    # The last-task check above sees one task's validation data; a
    # consolidation that damages EARLIER classes on real data passed it, with
    # only synthetic evidence standing between that and acceptance. The escrow
    # closes that: the same test the arbiter gate uses, over the whole
    # validation split and every class seen so far.
    if Xva is not None and cfg.get("adol_escrow", True):
        from .gates import escrow_check, escrow_sample
        cand = Organism(); cand.organs = [C] + list(org.organs)
        Xe, ye = escrow_sample(Xva, yva, cfg.get("adol_escrow_per_class"), seed)
        all_classes = np.asarray(sorted(set(int(c) for o in org.organs
                                            for c in o.class_ids)))
        esc = escrow_check(cand.predict(Xe)["pred"], org.predict(Xe)["pred"],
                           np.asarray(ye), all_classes,
                           hard_floor_min_lost=cfg.get("adol_hard_floor_min_lost", 5))
        report["escrow"] = {k: esc[k] for k in
                            ("n_classes", "n_samples", "mcnemar_z",
                             "overall_new", "overall_ref", "flagged_classes",
                             "hard_floor_classes", "near_floor_classes",
                             "hard_floor_min_lost", "worst_class",
                             "worst_class_z", "passed")}
        if not esc["passed"]:
            why = []
            if esc["mcnemar_z"] < -1.645:
                why.append(f"McNemar z {esc['mcnemar_z']:.2f}")
            if esc["flagged_classes"]:
                why.append(f"{len(esc['flagged_classes'])} class(es) flagged")
            if esc["hard_floor_classes"]:
                why.append(f"{len(esc['hard_floor_classes'])} class(es) below the hard floor")
            report["why"] = "escrow refused: " + ", ".join(why)
            report["refused_by"] = "escrow"
            return _forced() if force else (org, report)
    else:
        report["escrow"] = None      # not available: gate as it stood pre-2026-09-09
    new = Organism()
    new.organs = [C] + list(org.organs)
    report["accepted"] = True
    return new, report


def consolidate_sheet(org: Organism, in_dim: int, T: int, cfg: dict, seed: int,
                      n_units: int = 80, n_per_organ: int = 3000,
                      Xva_last=None, yva_last=None, prox_policy: str = "union"):
    """Metamorphosis on the spatial sheet: one consolidated organ, units
    re-placed by the wiring economy, local connections re-learned from scratch
    under the hinge cost, cost-weighted pruned. Same verify-then-swap gate."""
    from .sheet_organ import SheetOrgan, place_center
    rng = np.random.default_rng(seed * 31 + 5)
    X_syn, y_syn = sample_pool(org, n_per_organ, rng)
    perm = rng.permutation(len(X_syn))
    n_hold = int(0.2 * len(X_syn))
    ho, tr = perm[:n_hold], perm[n_hold:]
    classes = sorted(set(int(c) for o in org.organs for c in o.class_ids))

    C = SheetOrgan(in_dim, T, n_units, classes, center=place_center(None),
                   prev_organs=None, seed=seed * 977 + 13, listen=False,
                   q_cov_in=cfg.get("q_cov_in", 0.99), static=cfg["static"],
                   spread=0.16)
    C.train_proto(X_syn[tr], y_syn[tr], epochs=cfg.get("adol_epochs", 40))
    if prox_policy == "union":
        C.prox = _inherit_territory(org)
    solved = C.solve_thresholds(X_syn[ho], y_syn[ho],
                                p_target=cfg.get("p_target", 0.95), require=True)
    report = {"accepted": False, "classes": classes, "params_C": C.n_params()}
    if solved is None:
        report["why"] = "no threshold meets target on synthetic holdout"
        return org, report
    C.freeze(X_syn[ho], y_syn[ho])
    report["syn_holdout_cov"], report["syn_holdout_prec"] = solved
    if Xva_last is not None:
        new = Organism(); new.organs = [C] + list(org.organs)
        old_s = org.scores(Xva_last, yva_last)["strict"]
        new_s = new.scores(Xva_last, yva_last)["strict"]
        report["real_val_last_task"] = {"before": old_s, "after": new_s}
        if new_s < old_s - cfg.get("adol_val_slack", 0.05):
            report["why"] = "regressed on last task's real val"
            return org, report
    new = Organism()
    new.organs = [C] + list(org.organs)
    report["accepted"] = True
    return new, report


def metamorphosis_audit(before: Organism, after: Organism, Xte, yte):
    """Real-test accounting of what adolescence changed: answer flips and their
    direction, coverage movement, and who answers now."""
    b = before.predict(Xte)
    a = after.predict(Xte)
    was = b["pred"] != ABSTAIN
    now = a["pred"] != ABSTAIN
    flips = was & now & (a["pred"] != b["pred"])
    return {
        "flips": int(flips.sum()),
        "flips_to_right": int((flips & (a["pred"] == yte)).sum()),
        "flips_to_wrong": int((flips & (b["pred"] == yte)).sum()),
        "gained_answers": int((~was & now).sum()),
        "lost_answers": int((was & ~now).sum()),
        "share_answered_by_C": float((a["organ"] == 0).mean()),
    }
