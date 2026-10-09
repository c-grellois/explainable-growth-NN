"""The repaired acceptance gate.

Two changes slipped through the lifecycle's own gates and were reported as a
blind spot (REPORT.md B-ensemble and B-negotiate):

  * the free arbiter, which buys 94% of its CIFAR-ViT gain by answering OUTSIDE
    every certified territory -- accepted 3/3 by the phase-4 gate;
  * the negotiation oracle arm, quietly -0.0045 on real test -- accepted 3/3.

Both were accepted because the gate scores per-class strict accuracy on
CONTRACT SAMPLES plus the last task's real validation, and neither signal can
see either failure. This module adds the two missing checks and keeps the old
ones, so every verdict names which sub-check fired.

  A. territory  -- data-free and exact. Every answer must come from an organ
     that ACCEPTED the input. A candidate that answers where no contract speaks
     is not covered by the certificate, whatever it does to accuracy. Catches
     the arbiter; costs nothing; needs no data.
  B. escrow     -- per-class floors on a small REAL held-out escrow (N samples
     per class, used only by gates, never for training). This is the apoptosis
     v4 remedy, which turned an 8-point silent leak into 0 deaths / 0 leak: the
     damage that contract-sample MEANS cannot see is per-class and real.
  C. contract   -- the ORIGINAL check (per-class on the synthetic holdout plus
     last-task real validation), retained so the report can show what it misses.

A candidate passes only if every enabled sub-check passes. `certified` is
reported separately from `accepted`: a change may be beneficial and uncertified
(the free arbiter), which the old gate could not express at all.
"""
from __future__ import annotations

import numpy as np

from .organism import ABSTAIN


def _per_class(pred, y, classes):
    """Strict per-class accuracy (abstention counts as error)."""
    out = np.zeros(len(classes))
    for i, c in enumerate(classes):
        m = y == c
        out[i] = float((pred[m] == c).mean()) if m.any() else 1.0
    return out


def territory_check(answered, inside):
    """A. Data-free: every answer must sit in an accepting organ's territory."""
    n = int(answered.sum())
    outside = int((answered & ~inside).sum())
    return {"name": "territory", "answers": n, "outside": outside,
            "uncertified_fraction": (outside / n) if n else 0.0,
            "passed": outside == 0}


def _mcnemar(pred_new, pred_ref, y):
    """Paired test on the SAME samples: b = ref right & new wrong, c = the
    reverse. Only discordant pairs carry information, which is why this has far
    more power than comparing two accuracies -- the concordant majority cancels.
    Returns (b, c, z); z < 0 means the candidate is worse."""
    r_ok, n_ok = pred_ref == y, pred_new == y
    b = int((r_ok & ~n_ok).sum())
    c = int((~r_ok & n_ok).sum())
    z = (c - b) / np.sqrt(b + c) if (b + c) else 0.0
    return b, c, float(z)


def escrow_check(pred_new, pred_ref, y, classes, z_crit=1.645,
                 hard_floor=0.25, min_per_class=10, hard_floor_min_lost=5):
    """B. REAL held-out escrow (gates only, never training), tested properly.

    Two earlier designs failed and both failures are instructive:

      * comparing two per-class accuracies with a fixed 0.05 floor across 100
        classes is a MULTIPLE-COMPARISONS trap -- at ~60 samples per class the
        minimum over 100 noisy classes falls below any small fixed slack almost
        surely, so it refused a change that gained 3 points overall;
      * drawing the escrow from TRAIN flatters every candidate, because the
        organs fit train better than test: it scored a genuinely -0.0048 change
        at +0.0003.

    So: escrow comes from the held-out VALIDATION split, the overall verdict is
    a one-sided McNemar test on discordant pairs, and a class is flagged only
    when its own paired test survives a Bonferroni correction -- plus a hard
    floor for catastrophic single-class collapse, which is the apoptosis
    failure mode and must stay catchable regardless of significance.

    A third failure, found on 2026-09-09 when this escrow was extended to the
    adolescence gate: the hard floor was a FRACTION with no count condition, so
    on a benchmark with many small classes it fires on noise. ImageNet-R has
    ~120 escrow classes with a median of 14 validation samples, and one class
    going from 7/10 to 4/10 -- three samples -- breached a 0.25 floor and
    refused a consolidation that improved overall validation accuracy from
    0.289 to 0.456 (McNemar z = +18.1) and improved 94 of 119 classes. The
    floor now also requires the loss to amount to at least `hard_floor_min_lost`
    samples. Five is deliberately permissive: it only excludes swings of one to
    four samples, which cannot survive the multiplicity correction the flagged
    test applies anyway (with ~120 classes that threshold is z ~ 4.7, needing
    ~22 net losses), while the collapse the floor exists for -- a class falling
    from ~0.9 to 0 on 20+ samples -- loses 18 or more and is still caught. The
    floor had never fired in the 41 recorded arbiter escrow decisions, so this
    condition cannot alter any verdict already on record.
    """
    keep = [c for c in classes if (y == c).sum() >= min_per_class]
    b, c, z = _mcnemar(pred_new, pred_ref, y)
    overall_ok = z >= -z_crit

    zc = z_crit + np.sqrt(2.0 * np.log(max(len(keep), 1)))   # Bonferroni-ish
    flagged, worst_c, worst_z = [], None, 0.0
    hard, near_floor = [], []
    for cl in keep:
        m = y == cl
        bb, cc, zz = _mcnemar(pred_new[m], pred_ref[m], y[m])
        if zz < worst_z:
            worst_z, worst_c = zz, int(cl)
        if zz < -zc:
            flagged.append(int(cl))
        p_ref = float((pred_ref[m] == cl).mean())
        p_new = float((pred_new[m] == cl).mean())
        # a floor breach must be both a large FRACTION and enough SAMPLES to
        # tell collapse from noise (see the docstring)
        n_lost = int(round((p_ref - p_new) * int(m.sum())))
        if p_ref - p_new > hard_floor and n_lost >= hard_floor_min_lost:
            hard.append(int(cl))
        elif p_ref - p_new > hard_floor:
            near_floor.append({"class": int(cl), "n": int(m.sum()),
                               "p_ref": p_ref, "p_new": p_new, "n_lost": n_lost})

    return {"name": "escrow", "n_classes": len(keep), "n_samples": int(len(y)),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_z": z,
            "overall_new": float((pred_new == y).mean()),
            "overall_ref": float((pred_ref == y).mean()),
            "flagged_classes": flagged, "hard_floor_classes": hard,
            "near_floor_classes": near_floor,
            "hard_floor_min_lost": int(hard_floor_min_lost),
            "worst_class": worst_c, "worst_class_z": worst_z,
            "z_crit_per_class": float(zc),
            "passed": bool(overall_ok and not flagged and not hard)}


def contract_check(pred_new, pred_ref, y_ho, classes, val_new, val_ref,
                   mean_slack=0.02, floor_slack=0.05, val_slack=0.05):
    """C. The ORIGINAL gate: per-class on contract samples + last-task val."""
    pc_new = _per_class(pred_new, y_ho, classes)
    pc_ref = _per_class(pred_ref, y_ho, classes)
    ok = (pc_new.mean() >= pc_ref.mean() - mean_slack
          and pc_new.min() >= pc_ref.min() - floor_slack
          and val_new >= val_ref - val_slack)
    return {"name": "contract", "mean_new": float(pc_new.mean()),
            "mean_ref": float(pc_ref.mean()), "floor_new": float(pc_new.min()),
            "floor_ref": float(pc_ref.min()), "val_new": float(val_new),
            "val_ref": float(val_ref), "passed": bool(ok)}


def certify(checks):
    """Combine sub-check dicts into one verdict that names what fired."""
    failed = [c["name"] for c in checks if not c["passed"]]
    terr = next((c for c in checks if c["name"] == "territory"), None)
    return {"accepted": not failed, "failed": failed,
            "certified": bool(terr["passed"]) if terr else None,
            "uncertified_fraction": terr["uncertified_fraction"] if terr else None,
            "checks": {c["name"]: c for c in checks}}


def escrow_sample(Xva, yva, per_class=None, seed=0):
    """The escrow: REAL held-out VALIDATION data, gates only, never training.

    Two bugs lived here in turn. Drawing from TRAIN flattered every candidate,
    because organs fit train better than test. Then SUBSAMPLING validation to
    N per class silently discarded most of it on datasets with few classes --
    on wall-robot (4 classes) a 60/class cap kept 214 of 613 samples and
    dropped a real -0.116 regression from z = -6.86 to z = -1.57, i.e. from
    decisively refused to accepted. Validation is held out by construction, so
    there is no reason to spend less of it than exists.

    per_class=None (the default) uses the WHOLE validation split. Pass an int
    only to study how the gate degrades with a smaller escrow."""
    if per_class is None:
        return np.asarray(Xva), np.asarray(yva)
    rng = np.random.default_rng(seed * 71 + 3)
    Xe, ye = [], []
    for c in np.unique(yva):
        idx = np.flatnonzero(yva == c)
        take = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        Xe.append(Xva[take]); ye.append(yva[take])
    return np.concatenate(Xe), np.concatenate(ye)


def detectable_drop(n_escrow, discordant_rate=0.05, z_crit=1.645):
    """Smallest OVERALL regression a McNemar escrow of this size can resolve.

    With d = discordant_rate * n_escrow informative pairs, a net imbalance is
    detectable at z_crit when (c-b) >= z_crit*sqrt(d); dividing by n_escrow
    gives the accuracy drop that corresponds to. Comparing two accuracies
    instead would need far more samples -- the concordant majority is noise.
    """
    d = max(discordant_rate * n_escrow, 1.0)
    return float(z_crit * np.sqrt(d) / n_escrow)


class DriftLedger:
    """A LIFETIME budget on accepted escrow drift.

    The gate's honest residual is that regressions below its stated resolution
    pass. Over a long life of many gated events -- nightly sleep cycles are the
    extreme case -- such sub-resolution slips can COMPOUND, each individually
    invisible. The ledger closes that loophole: every accepted change records
    its signed escrow delta (c - b) / n, and a candidate is refused when the
    cumulative delta would fall below -budget, however well it passes its own
    per-event test. This is the second-order SLA (exp_meta) applied to the
    escrow: the organism promises its lifetime accepted drift stays bounded.
    """

    def __init__(self, budget: float = 0.02):
        self.budget = float(budget)
        self.events = []

    def delta(self, b: int, c: int, n: int) -> float:
        return (c - b) / n if n else 0.0

    def cumulative(self) -> float:
        return float(sum(e["delta"] for e in self.events))

    def would_breach(self, b: int, c: int, n: int) -> bool:
        return self.cumulative() + self.delta(b, c, n) < -self.budget

    def record(self, name: str, b: int, c: int, n: int) -> float:
        d = self.delta(b, c, n)
        self.events.append({"event": name, "b": int(b), "c": int(c),
                            "n": int(n), "delta": d,
                            "cumulative": self.cumulative() + d})
        return d

    def summary(self) -> dict:
        return {"budget": self.budget, "events": list(self.events),
                "cumulative": self.cumulative(),
                "n_events": len(self.events)}


def escrow_split(Xva, yva, seed: int = 0):
    """Deterministic, class-stratified halves of the validation split.

    Half A feeds the lifecycle's own gates (adolescence's last-task check, the
    original rule's validation check); half B is the escrow. Used to test
    whether the escrow's reuse of validation data -- disclosed as benign in the
    report -- actually changes any verdict. Returns (idx_A, idx_B)."""
    rng = np.random.default_rng(seed * 131 + 17)
    a, b = [], []
    for cl in np.unique(yva):
        idx = np.flatnonzero(yva == cl)
        rng.shuffle(idx)
        half = len(idx) // 2
        a.extend(idx[:half].tolist()); b.extend(idx[half:].tolist())
    return np.asarray(sorted(a)), np.asarray(sorted(b))
