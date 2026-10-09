"""The organism: a first-accept cascade of frozen organs, grown by abstention.

Retention theorem (checked, not trusted): an input answered by the organism keeps
its answer under all future growth, because routing is first-accept in birth order
and every organ is frozen at birth. Tightening may move an input from answered to
abstained (never flip it); growth may move abstained to answered. `retention_check`
verifies this bitwise against stored snapshots on a fixed reference set.
"""
from __future__ import annotations

import numpy as np

from .organ import Organ

ABSTAIN = -1


class Organism:
    def __init__(self):
        self.organs: list[Organ] = []
        self.log: list[dict] = []
        self._snapshots: list[dict] = []

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray):
        """First-accept cascade. Returns dict with:
        pred      : global class or ABSTAIN
        organ     : index of the answering organ (or -1)
        forced    : nearest-anchor fallback prediction for abstained inputs
        radius    : certified input-space robust radius (0 where abstained)
        """
        n = len(X)
        pred = np.full(n, ABSTAIN, dtype=np.int64)
        organ = np.full(n, -1, dtype=np.int64)
        radius = np.zeros(n)
        forced = np.full(n, ABSTAIN, dtype=np.int64)
        best_norm_d = np.full(n, np.inf)
        open_mask = np.ones(n, dtype=bool)
        for i, o in enumerate(self.organs):
            d1, m, p, d_in, vet, veto_r = o.evaluate(X)
            acc = o.accept(d1, m, d_in, vet)
            take = open_mask & acc
            pred[take] = p[take]
            organ[take] = i
            radius[take] = o.robust_radius(d1, m, d_in, vet, veto_r)[take]
            open_mask &= ~acc
            norm_d = d1 / (o.rho if np.isfinite(o.rho) else np.median(d1) + 1e-12)
            better = norm_d < best_norm_d
            forced[better] = p[better]
            best_norm_d[better] = norm_d[better]
        return {"pred": pred, "organ": organ, "radius": radius,
                "forced": np.where(pred != ABSTAIN, pred, forced)}

    def declined_mask(self, X) -> np.ndarray:
        return self.predict(X)["pred"] == ABSTAIN

    def scores(self, X, y) -> dict:
        r = self.predict(X)
        answered = r["pred"] != ABSTAIN
        cov = float(answered.mean())
        prec = float((r["pred"][answered] == y[answered]).mean()) if answered.any() else 0.0
        return {"strict": float((r["pred"] == y).mean()),          # abstain = error
                "forced": float((r["forced"] == y).mean()),
                "coverage": cov, "precision": prec,
                "mean_radius": float(r["radius"][answered].mean()) if answered.any() else 0.0}

    def n_params(self) -> int:
        return sum(o.n_params() for o in self.organs)

    # ------------------------------------------------------------------ #
    # growth: verify, then accept
    # ------------------------------------------------------------------ #

    def try_grow(self, candidate: Organ, Xval, yval, p_target=0.9,
                 gate: bool = True, label: str = "") -> bool:
        """Solve the candidate's thresholds on the val subset the organism
        currently declines; append only if it answers net-positively there.
        p_target may be a float or a declining ladder of floats: the candidate
        takes the highest-precision slice it can certify."""
        declined = self.declined_mask(Xval) if self.organs else np.ones(len(Xval), bool)
        Xd, yd = Xval[declined], yval[declined]
        if len(Xd) == 0:
            self.log.append({"event": "reject", "label": label, "why": "nothing declined"})
            return False
        ladder = [p_target] if np.isscalar(p_target) else list(p_target)
        solved = None
        for p in ladder:
            solved = candidate.solve_thresholds(Xd, yd, p_target=p, require=gate)
            if solved is not None:
                break
        if solved is None:
            self.log.append({"event": "reject", "label": label, "why": "no threshold meets target"})
            return False
        candidate.freeze(Xd, yd)
        d1, m, p, d_in, vet, _ = candidate.evaluate(Xd)
        acc = candidate.accept(d1, m, d_in, vet)
        net = int((p[acc] == yd[acc]).sum()) - int((p[acc] != yd[acc]).sum())
        if gate and net <= 0:
            self.log.append({"event": "reject", "label": label, "why": f"net gain {net}"})
            return False
        self.organs.append(candidate)
        self.log.append({"event": "grow", "label": label, "net": net,
                         "cov_on_declined": float(acc.mean()),
                         "prec_on_declined": float((p[acc] == yd[acc]).mean()) if acc.any() else 0.0,
                         "rho": candidate.rho, "tau": candidate.tau, "L": candidate.L})
        return True

    # ------------------------------------------------------------------ #
    # tightening against measured false accepts (continual setting)
    # ------------------------------------------------------------------ #

    def tighten_against(self, X_foreign: np.ndarray, self_budget: float = 0.02,
                        veto: bool = True, veto_theta_q: float = 0.97,
                        min_veto_neg: int = 20):
        """New data arrives whose classes no existing organ serves: every accept
        on it is a false accept. Two tightening stages per organ, both monotone:
        (1) scalar (rho, tau, rho_in) under a self-coverage budget; (2) LEARNED --
        a subspace veto trained on contract-sampled positives vs the residual
        falsely-accepted samples (see Organ.add_veto). Returns per-organ
        (n_excluded_scalar, self_loss, n_vetoes_added)."""
        out = []
        for i, o in enumerate(self.organs):
            excluded, loss = 0, 0.0
            if self_budget > 0 or veto:
                d1, m, _, d_in, vet, _ = o.evaluate(X_foreign)
            if self_budget > 0:
                excluded, loss = o.solve_tightening(d1, m, d_in,
                                                    self_budget=self_budget)
            n_veto = 0
            if veto:
                residual = o.accept(d1, m, d_in, vet)
                if residual.sum() >= min_veto_neg:
                    n_veto = o.add_veto(X_foreign[residual],
                                        theta_q=veto_theta_q, seed=1000 + i)
            out.append((excluded, loss, n_veto))
            self.log.append({"event": "tighten", "organ": i,
                             "excluded": excluded, "self_loss": loss,
                             "vetoes_added": n_veto})
        return out

    # ------------------------------------------------------------------ #
    # retention: the theorem, checked bitwise
    # ------------------------------------------------------------------ #

    def retention_snapshot(self, Xref: np.ndarray, tag: str = ""):
        r = self.predict(Xref)
        self._snapshots.append({"tag": tag, "answered": r["pred"] != ABSTAIN,
                                "pred": r["pred"].copy()})

    def retention_check(self, Xref: np.ndarray) -> dict:
        """Against every stored snapshot: no previously-answered input may have a
        DIFFERENT answer now (moving to abstained is allowed only via tightening;
        flips are never allowed). Returns the count of violations (must be 0)."""
        r = self.predict(Xref)
        flips = 0
        dropped = 0
        for snap in self._snapshots:
            was = snap["answered"]
            now_answered = r["pred"] != ABSTAIN
            flips += int((was & now_answered & (r["pred"] != snap["pred"])).sum())
            dropped += int((was & ~now_answered).sum())
        return {"flips": flips, "dropped_to_abstain": dropped,
                "n_snapshots": len(self._snapshots)}
