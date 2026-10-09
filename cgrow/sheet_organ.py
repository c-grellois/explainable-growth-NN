"""The organism on ASDbrain's spatial substrate.

A SheetOrgan is a spatially localized recurrent population on the shared 2-D
sheet, faithful to the parent substrate's commitments: units have positions in
[0,1]^2, dynamics h_{t+1} = tanh(W h_t + U x_t + b) for T steps, input enters at
a fixed entry site and input wiring is priced by distance to it, lateral wiring
is priced by the hinge cost c(d) = d + beta * max(0, d - d0)^2, and edges are
pruned to a budget by the cost-weighted score |W| / c(d).

The organism-level additions:

  * DIRECTIONAL growth. A newborn organ may wire FROM earlier organs' frozen
    units (listening: rows of the newborn only), but no edge ever points into a
    frozen row. Old organs' dynamics are therefore bit-identical under all
    future growth -- the retention theorem on a shared recurrent sheet.
  * The organ's contract lives on its own units' mean state over time; the
    proximity contract and vetoes operate on the flattened input and are
    inherited unchanged from Organ (the feature map is the only override).
  * W_self is spectrally capped below 1 (projection after each step), keeping
    the echo-state property and a finite input->state Lipschitz bound
    sigma(U) * sum_t sigma(W)^t.

Placement: an organ's center is chosen on a grid by minimizing input-wiring
cost to the entry site plus a crowding penalty against existing units -- the
wiring economy decides where development goes, as in the parent paper.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .organ import Organ, DTYPE

ENTRY = torch.tensor([0.08, 0.5])       # input entry site on the sheet
D0, BETA = 0.2, 5.0                      # ASDbrain's hinge cost parameters


def hinge_cost(d: torch.Tensor) -> torch.Tensor:
    return d + BETA * torch.clamp(d - D0, min=0.0) ** 2


def place_center(existing_pos: torch.Tensor | None, grid: int = 21,
                 crowd_sigma: float = 0.12, crowd_w: float = 3.0) -> torch.Tensor:
    """Center for a new organ: cheapest input wiring, penalized for crowding."""
    ax = torch.linspace(0.08, 0.92, grid)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    cand = torch.stack([gx.ravel(), gy.ravel()], dim=1)
    cost = hinge_cost(torch.linalg.norm(cand - ENTRY, dim=1))
    if existing_pos is not None and len(existing_pos):
        d = torch.cdist(cand, existing_pos)
        cost = cost + crowd_w * torch.exp(-0.5 * (d / crowd_sigma) ** 2).sum(dim=1)
    return cand[int(cost.argmin())]


class SheetOrgan(Organ):
    """A contracted organ made of spatially embedded recurrent units."""

    def __init__(self, in_dim: int, T: int, n_units: int, class_ids,
                 center: torch.Tensor, prev_organs: list | None = None,
                 seed: int = 0, g: float = 0.9, w_cap: float = 0.95,
                 spread: float = 0.07, listen: bool = True,
                 q_cov_in: float = 0.995, prox: str = "pca",
                 lam: float = 1.5e-3, lam_u: float = 1.5e-3,
                 static: bool = False):
        # deliberately NOT calling Organ.__init__ -- no MLP trunk; we recreate
        # the contract-side attributes it would have set
        self.class_ids = np.asarray(sorted(class_ids), dtype=np.int64)
        self.rho, self.tau, self.rho_in = np.inf, 0.0, np.inf
        self.L = None
        self.frozen = False
        self.stats = None
        self.seed = seed
        self.k_centroids = 5
        self.q_cov_in = q_cov_in
        self.prox_name = prox
        self.prox = None
        self.vetoes = []

        self.in_dim, self.T = int(in_dim), int(T)
        self.static = bool(static)   # static stream: one pattern drives all T steps
        self.n = int(n_units)
        self.listen = bool(listen)
        self.lam, self.lam_u = float(lam), float(lam_u)
        self.prev = list(prev_organs or []) if listen else []
        g_t = torch.Generator().manual_seed(seed)
        self.pos = (center + spread * torch.randn(self.n, 2, generator=g_t)
                    ).clamp(0.02, 0.98)
        self.center = center.clone()
        # geometry-derived costs, fixed at birth
        self.cost_self = hinge_cost(torch.cdist(self.pos, self.pos))
        self.cost_u = hinge_cost(torch.linalg.norm(self.pos - ENTRY, dim=1)
                                 ).unsqueeze(1)
        n_prev = sum(o.n for o in self.prev)
        if n_prev:
            prev_pos = torch.cat([o.pos for o in self.prev])
            self.cost_prev = hinge_cost(torch.cdist(self.pos, prev_pos))
        # parameters (rows belong to THIS organ only -- directionality)
        self.W_self = nn.Parameter(
            torch.randn(self.n, self.n, generator=g_t) * (g / math.sqrt(self.n)))
        self.W_prev = (nn.Parameter(torch.zeros(self.n, n_prev))
                       if n_prev else None)
        self.U = nn.Parameter(
            torch.randn(self.n, in_dim, generator=g_t) / math.sqrt(in_dim))
        self.b = nn.Parameter(torch.zeros(self.n))
        self.anchors = nn.Parameter(
            0.1 * torch.randn(len(self.class_ids), self.n, generator=g_t))
        self.w_cap = float(w_cap)
        self._u_pi = torch.randn(self.n, generator=g_t)   # power-iteration state
        self.edge_mask_self = torch.ones(self.n, self.n, dtype=torch.bool)
        self.edge_mask_prev = (torch.ones(self.n, n_prev, dtype=torch.bool)
                               if n_prev else None)

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #

    def _run(self, X3: torch.Tensor, prev_states: list[torch.Tensor]):
        """States of this organ's units. X3 is (B, T, d_in); prev_states is a
        list of (B, T, n_j) trajectories of earlier organs (frozen)."""
        B = X3.shape[0]
        h = torch.zeros(B, self.n)
        Ws = self.W_self * self.edge_mask_self
        prev = torch.cat(prev_states, dim=2) if (self.prev and prev_states) else None
        states = []
        for t in range(self.T):
            drive = X3[:, t] @ self.U.t() + self.b
            if prev is not None and self.W_prev is not None:
                drive = drive + prev[:, t] @ (self.W_prev * self.edge_mask_prev).t()
            h = torch.tanh(h @ Ws.t() + drive)
            states.append(h)
        S = torch.stack(states, dim=1)                    # (B, T, n)
        return S, S.mean(dim=1)                           # trajectory, features

    def _chain_states(self, X3: torch.Tensor):
        """Recompute earlier organs' (frozen) trajectories, in birth order."""
        outs = []
        for o in self.prev:
            with torch.no_grad():
                S, _ = o._run(X3, [outs[i] for i in range(len(o.prev))])
            outs.append(S)
        return outs

    def _as_3d(self, X: np.ndarray) -> torch.Tensor:
        Xt = torch.as_tensor(X, dtype=DTYPE)
        if self.static:
            return Xt.unsqueeze(1).expand(-1, self.T, -1)
        return Xt.reshape(len(X), self.T, self.in_dim)

    def _features(self, X: np.ndarray) -> torch.Tensor:
        X3 = self._as_3d(X)
        with torch.no_grad():
            _, feat = self._run(X3, self._chain_states(X3))
        return feat

    def _lipschitz_exact(self) -> float:
        """Input->feature bound: sigma(U) * sum_t sigma(W_self)^t (listening
        paths only add frozen-organ terms; reported bound covers self path)."""
        sU = float(torch.linalg.svdvals(self.U.detach().double())[0])
        sW = min(float(torch.linalg.svdvals(
            (self.W_self * self.edge_mask_self).detach().double())[0]), self.w_cap)
        return sU * sum(sW ** t for t in range(self.T))

    @torch.no_grad()
    def _project(self, n_iter: int = 2):
        W = self.W_self.data
        u = self._u_pi
        for _ in range(n_iter):
            v = W.t() @ u
            v = v / (v.norm() + 1e-12)
            u = W @ v
            u = u / (u.norm() + 1e-12)
        self._u_pi = u
        sigma = float(u @ (W @ v))
        if sigma > self.w_cap:
            W.mul_(self.w_cap / sigma)

    # ------------------------------------------------------------------ #
    # training (prototypical + wiring economy), then cost-weighted pruning
    # ------------------------------------------------------------------ #

    def parameters(self):
        ps = [self.W_self, self.U, self.b, self.anchors]
        if self.W_prev is not None:
            ps.append(self.W_prev)
        return ps

    def train_proto(self, X: np.ndarray, y: np.ndarray, epochs: int = 40,
                    lr: float = 5e-3, batch: int = 256, verbose: bool = False):
        assert not self.frozen
        keep = np.isin(y, self.class_ids)
        X, y = X[keep], y[keep]
        local = {c: i for i, c in enumerate(self.class_ids)}
        yl = torch.as_tensor(np.array([local[c] for c in y], dtype=np.int64))
        X3 = self._as_3d(X)
        prev_states = self._chain_states(X3)              # frozen, computed once
        with torch.no_grad():
            _, z0 = self._run(X3, prev_states)
            for i in range(len(self.class_ids)):
                sel = yl == i
                if sel.any():
                    self.anchors.data[i] = z0[sel].mean(0)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        g = torch.Generator().manual_seed(self.seed + 99)
        n = len(X3)
        for ep in range(epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                _, feat = self._run(X3[idx], [p[idx] for p in prev_states])
                logits = -torch.cdist(feat, self.anchors) ** 2
                loss = nn.functional.cross_entropy(logits, yl[idx])
                wire = (self.W_self.abs() * self.cost_self).sum() \
                    + (self.U.abs() * self.cost_u).sum() * (self.lam_u / self.lam)
                if self.W_prev is not None:
                    wire = wire + (self.W_prev.abs() * self.cost_prev).sum()
                loss = loss + self.lam * wire
                opt.zero_grad(); loss.backward(); opt.step()
                self._project()
        self.prune(frac_keep=0.35)
        from .proximity import make_prox
        self.prox = make_prox(self.prox_name, seed=self.seed).fit(
            X.reshape(len(X), -1), y)

    @torch.no_grad()
    def prune(self, frac_keep: float = 0.35):
        """Cost-weighted pruning (|W| / c(d)), ASDbrain's stage-3 rule, applied
        per organ before freeze -- pruning is functionally free in the parent;
        thresholds are solved after it, so nothing is silently lost."""
        def keep_mask(W, cost, frac):
            score = (W.abs() / cost.clamp(min=1e-6)).ravel()
            k = max(int(frac * score.numel()), 1)
            thr = score.kthvalue(score.numel() - k + 1).values
            return (score >= thr).reshape(W.shape)
        self.edge_mask_self &= keep_mask(self.W_self.data, self.cost_self, frac_keep)
        self.W_self.data *= self.edge_mask_self
        if self.W_prev is not None:
            self.edge_mask_prev &= keep_mask(self.W_prev.data, self.cost_prev, frac_keep)
            self.W_prev.data *= self.edge_mask_prev

    # ------------------------------------------------------------------ #
    # anatomy accounting
    # ------------------------------------------------------------------ #

    def edges(self):
        """(|w|, length) arrays of this organ's live edges, self + listening."""
        w, d = [], []
        m = self.edge_mask_self
        w.append(self.W_self.data.abs()[m].ravel())
        d.append(torch.cdist(self.pos, self.pos)[m].ravel())
        if self.W_prev is not None:
            prev_pos = torch.cat([o.pos for o in self.prev])
            mp = self.edge_mask_prev
            w.append(self.W_prev.data.abs()[mp].ravel())
            d.append(torch.cdist(self.pos, prev_pos)[mp].ravel())
        return torch.cat(w), torch.cat(d)

    def n_params(self) -> int:
        n = self.W_self.numel() + self.U.numel() + self.b.numel() + self.anchors.numel()
        if self.W_prev is not None:
            n += self.W_prev.numel()
        return n
