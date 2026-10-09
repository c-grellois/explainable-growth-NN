"""An organ: a spectrally-capped prototypical trunk behind a ball contract.

The contract is (anchors, rho, tau, L):
  accept x  iff  d1(x) <= rho  and  (d2(x) - d1(x))/2 > tau
  predict the nearest anchor's class; the prediction is invariant to any input
  perturbation of norm < ((d2-d1)/2 - tau)/L, with L the trunk's (exact, post-hoc)
  Lipschitz constant -- certification is arithmetic, no bound propagation.

Methodological inheritances: the certificate's free parameters (rho, tau) are
SOLVED against held-out data, never penalized in the loss (contracts note §7);
after freeze() nothing about the organ may change except monotone tightening
(rho down, tau up), which can only shrink the accepted set, never flip an answer.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

DTYPE = torch.float32


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal calibration: the k-th order statistic with
    k = ceil((n+1)(1-alpha)). Under exchangeability of calibration and test
    samples, a fresh sample's score is <= this threshold with probability
    >= 1-alpha, finite-sample and distribution-free. This replaces plain
    quantiles everywhere a threshold buys a coverage guarantee (territory
    radii; veto pass rates, where the guarantee is w.r.t. the contract's
    asserted sampling model)."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float(s[-1])          # alpha too small for n: report max
    return float(s[k - 1])


class SpectralMLP(nn.Module):
    """MLP whose layer spectral norms are projected to <= cap after each step."""

    def __init__(self, dims, cap: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1], dtype=DTYPE)
            with torch.no_grad():
                w = torch.randn(dims[i + 1], dims[i], generator=g, dtype=DTYPE)
                lin.weight.copy_(w / max(1.0, float(torch.linalg.matrix_norm(w, 2)) / cap))
                lin.bias.zero_()
            self.layers.append(lin)
        self.cap = float(cap)
        # persistent power-iteration vectors, one per layer
        self._u = [torch.randn(d_out, generator=g, dtype=DTYPE)
                   for d_out in dims[1:]]

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

    @torch.no_grad()
    def project(self, n_iter: int = 2):
        """Clamp each layer's spectral norm to cap (approx. power iteration)."""
        for k, lin in enumerate(self.layers):
            W = lin.weight
            u = self._u[k]
            for _ in range(n_iter):
                v = W.t() @ u
                v = v / (v.norm() + 1e-12)
                u = W @ v
                u = u / (u.norm() + 1e-12)
            self._u[k] = u
            sigma = float(u @ (W @ v))
            if sigma > self.cap:
                W.mul_(self.cap / sigma)

    @torch.no_grad()
    def lipschitz_exact(self) -> float:
        """Exact product of layer spectral norms (ReLU is 1-Lipschitz)."""
        L = 1.0
        for lin in self.layers:
            L *= float(torch.linalg.svdvals(lin.weight.double())[0])
        return L


class Organ:
    def __init__(self, in_dim: int, feat_dim: int, hidden, cap: float,
                 class_ids, seed: int = 0, k_centroids: int = 5,
                 q_cov_in: float = 0.995, prox: str = "kmeans"):
        self.k_centroids = k_centroids
        self.q_cov_in = q_cov_in
        self.prox_name = prox
        self.prox = None         # fitted proximity geometry (see proximity.py)
        self.vetoes = []         # learned tightening gates: (class_idx, net, theta)
        self.trunk = SpectralMLP([in_dim, *hidden, feat_dim], cap=cap, seed=seed)
        self.class_ids = np.asarray(sorted(class_ids), dtype=np.int64)
        g = torch.Generator().manual_seed(seed + 7)
        self.anchors = nn.Parameter(
            0.1 * torch.randn(len(self.class_ids), feat_dim, generator=g, dtype=DTYPE))
        self.rho = np.inf
        self.tau = 0.0
        self.rho_in = np.inf     # input-space proximity cap (OOD defense)
        self.L = None            # exact Lipschitz constant, set at freeze
        self.frozen = False
        self.stats = None        # birth-val (d1, margin, d_in, correct) for tightening
        self.seed = seed

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #

    def _features(self, X: np.ndarray) -> torch.Tensor:
        """Feature map behind the contract. Subclasses (e.g. the spatial sheet
        organ) override this; everything else -- proximity, thresholds, vetoes,
        tightening -- is geometry-agnostic and inherited."""
        return self.trunk(torch.as_tensor(X, dtype=DTYPE))

    def _lipschitz_exact(self) -> float:
        return self.trunk.lipschitz_exact()

    @torch.no_grad()
    def evaluate(self, X: np.ndarray):
        """Return (d1, margin, pred, d_in): nearest-anchor distance and half-gap
        to the runner-up in feature space, predicted GLOBAL class id, and
        nearest-centroid distance in INPUT space (the OOD-sensitive coordinate)."""
        Xt = torch.as_tensor(X, dtype=DTYPE)
        z = self._features(X)
        d = torch.cdist(z, self.anchors.detach())
        if d.shape[1] == 1:
            d1 = d[:, 0]
            margin = torch.full_like(d1, np.inf)
            idx = torch.zeros(len(d1), dtype=torch.long)
        else:
            vals, order = torch.topk(d, 2, dim=1, largest=False)
            d1, d2 = vals[:, 0], vals[:, 1]
            margin = (d2 - d1) / 2.0
            idx = order[:, 0]
        pred = self.class_ids[idx.numpy()]
        parts = (self.prox.project(X)
                 if self.vetoes and getattr(self.prox, "name", None) in ("pca", "mix")
                 else None)
        if self.prox is not None:
            d_in = self.prox.score(X) if parts is None else self.prox.score(X, parts=parts)
        else:
            d_in = np.zeros(len(Xt))
        vetoed, veto_r = self._veto_judge(X, parts=parts)
        return (d1.numpy().astype(np.float64), margin.numpy().astype(np.float64),
                pred, np.asarray(d_in, dtype=np.float64), vetoed, veto_r)

    def _veto_judge(self, X: np.ndarray, parts: list | None = None):
        """Apply the learned tightening gates. Each veto lives in one class
        territory's whitened PCA coordinates; a sample is judged by the veto of
        its NEAREST territory. Returns (vetoed mask, certified veto slack as an
        input-space radius; inf where no veto applies). `parts` is the prox's
        precomputed project(X), if the caller already has it."""
        n = len(X)
        vetoed = np.zeros(n, bool)
        veto_r = np.full(n, np.inf)
        if (not self.vetoes or self.prox is None
                or self.prox.name not in ("pca", "mix")):
            return vetoed, veto_r
        projs, ci = self._proj_nearest(X, parts=parts)
        for class_idx, net, theta, L_in in self.vetoes:
            sel = ci == class_idx
            if not sel.any():
                continue
            with torch.no_grad():
                p = torch.sigmoid(net(torch.as_tensor(projs[class_idx][sel],
                                                      dtype=DTYPE))
                                  ).numpy().ravel()
            vetoed[sel] |= p > theta
            veto_r[sel] = np.minimum(veto_r[sel],
                                     np.maximum(theta - p, 0.0) / L_in)
        return vetoed, veto_r

    def _proj_nearest(self, X: np.ndarray, parts: list | None = None):
        """Whitened PCA coordinates of every sample in EVERY class territory
        (PCAProx only), plus the index of each sample's nearest territory.

        Returns (projs, ci): projs[j] has shape (n, q_j) and ci[i] is the
        territory whose coordinates judge sample i, so a consumer wanting the
        nearest-territory coordinates of the samples routed to territory j
        takes projs[j][ci == j]. Territories may have different widths q_j
        (PCAProx.fit caps q at n_c - 1 for a class with few samples), so the
        rows must NOT be stacked into one (n, q) array. 2026-09-07 GPU-box
        fix: the previous np.stack over mixed widths raised ValueError on the
        PTM suite (see PTM_RESULTS.md §3); for equal widths the per-territory
        arrays hold exactly the values the stack used to hold. `parts` is as in
        _veto_judge."""
        pr = self.prox
        parts = pr.project(X) if parts is None else parts
        best = np.full(len(X), np.inf)
        ci = np.zeros(len(X), dtype=int)
        projs = []
        for j, ((p, res), lam, rq, wq) in enumerate(zip(parts, pr.lam,
                                                        pr.r_q, pr.w_q)):
            pw = p / np.sqrt(lam)
            if pr.name == "mix":
                white = pr._w_score(pw, pr.gmm[j])
            else:
                white = np.sqrt((pw ** 2).sum(axis=1))
            s = np.maximum(res / rq, white / wq)
            take = s < best
            best[take] = s[take]; ci[take] = j
            projs.append(pw)
        return projs, ci

    def add_veto(self, X_fa: np.ndarray, theta_q: float = 0.97,
                 min_neg: int = 10, seed: int = 0,
                 X_self: np.ndarray | None = None, max_self_loss: float = 0.03):
        """Learned tightening: per class territory, train a small capped net on
        exact-spec synthetic positives (z ~ N(0, I) in whitened subspace coords)
        vs the (real or contract-sampled) negatives routed to that territory.
        Purely conjunctive -- the accepted set can only shrink. When real own
        data X_self is available (birth vetoes), each veto is VERIFIED against
        it and discarded if it costs more than max_self_loss of own coverage.
        Returns vetoes added."""
        if (self.prox is None or self.prox.name not in ("pca", "mix")
                or len(X_fa) < min_neg):
            return 0
        projs, ci = self._proj_nearest(X_fa)
        projs_self = ci_self = None
        if X_self is not None and len(X_self):
            projs_self, ci_self = self._proj_nearest(X_self)
        rng = np.random.default_rng(seed)
        added = 0
        for j in range(len(self.prox.classes)):
            neg = projs[j][ci == j]
            if len(neg) < min_neg:
                continue
            q = neg.shape[1]
            Z_syn = self.prox.sample_whitened(j, 3000, rng)
            net = SpectralMLP([q, 128, 1], cap=4.0, seed=seed + 7 * j)
            Xall = torch.as_tensor(np.concatenate([Z_syn, neg]), dtype=DTYPE)
            lab = torch.cat([torch.zeros(len(Z_syn)), torch.ones(len(neg))])
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            g = torch.Generator().manual_seed(seed + j)
            for _ in range(12):
                perm = torch.randperm(len(Xall), generator=g)
                for s in range(0, len(Xall), 256):
                    idx = perm[s:s + 256]
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        net(Xall[idx]).squeeze(1), lab[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
                    net.project()
            with torch.no_grad():
                p_syn = torch.sigmoid(net(torch.as_tensor(Z_syn, dtype=DTYPE))
                                      ).numpy().ravel()
            # conformal w.r.t. the spec's model: a fresh exact-spec sample
            # passes this veto with probability >= theta_q
            theta = conformal_quantile(p_syn, 1.0 - theta_q)
            if projs_self is not None:
                sel = ci_self == j
                if sel.any():
                    with torch.no_grad():
                        p_real = torch.sigmoid(net(torch.as_tensor(
                            projs_self[j][sel], dtype=DTYPE))).numpy().ravel()
                    if (p_real > theta).mean() > max_self_loss:
                        continue        # verified too costly on real own data
            # input -> whitened coords is linear with spectral norm 1/sqrt(lam_min);
            # sigmoid slope <= 1/4: certified input-space Lipschitz of the score
            L_in = (net.lipschitz_exact() * 0.25
                    / np.sqrt(float(self.prox.lam[j].min())))
            self.vetoes.append((j, net, theta, L_in))
            added += 1
        return added

    def accept(self, d1, margin, d_in, vetoed=None) -> np.ndarray:
        base = (d1 <= self.rho) & (margin > self.tau) & (d_in <= self.rho_in)
        return base if vetoed is None else base & ~vetoed

    def robust_radius(self, d1, margin, d_in, vetoed=None, veto_r=None):
        """Certified input-space radius for accepted points (0 elsewhere):
        min of the feature-margin slack (scaled by 1/L), the proximity slack
        (scaled by the proximity geometry's Lipschitz bound), and each
        applicable veto's slack."""
        L_prox = self.prox.lipschitz() if self.prox is not None else 1.0
        r = np.minimum(np.maximum(margin - self.tau, 0.0) / self.L,
                       np.maximum(self.rho_in - d_in, 0.0) / L_prox)
        if veto_r is not None:
            r = np.minimum(r, veto_r)
        return np.where(self.accept(d1, margin, d_in, vetoed), r, 0.0)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def train_proto(self, X: np.ndarray, y: np.ndarray, epochs: int = 40,
                    lr: float = 3e-3, batch: int = 256, verbose: bool = False,
                    prune: float = 0.0, prune_rounds: int = 1,
                    prune_epochs: int | None = None):
        """Prototypical training: CE over -||phi(x) - a_c||^2, spectral projection
        after every step. y carries GLOBAL class ids restricted to self.class_ids.

        `prune > 0` adds magnitude pruning *before the organ is frozen*, in the
        shape Han et al. (2015) use: train, drop the smallest-|w| fraction of
        each layer, retrain the survivors with the dropped weights pinned at
        zero. It happens here, and only here, because an organ's contract is
        solved against whatever trunk it ends up with: prune before
        `solve_thresholds`/`freeze` and every guarantee still describes the
        network that is actually shipped. Pruning a *frozen* organ would
        invalidate its contract and is not offered. `prune = 0` is a no-op and
        leaves the unpruned path bit-for-bit unchanged."""
        assert not self.frozen
        keep = np.isin(y, self.class_ids)      # an organ trains only on its classes
        X, y = X[keep], y[keep]
        local = {c: i for i, c in enumerate(self.class_ids)}
        yl = np.array([local[c] for c in y], dtype=np.int64)
        Xt = torch.as_tensor(X, dtype=DTYPE)
        yt = torch.as_tensor(yl)
        with torch.no_grad():   # anchor init: class means of initial features
            z0 = self.trunk(Xt)
            for i in range(len(self.class_ids)):
                sel = yt == i
                if sel.any():
                    self.anchors.data[i] = z0[sel].mean(0)
        opt = torch.optim.Adam([*self.trunk.parameters(), self.anchors], lr=lr)
        g = torch.Generator().manual_seed(self.seed + 99)
        n = len(Xt)

        def _epochs(n_ep, mask=None):
            for ep in range(n_ep):
                perm = torch.randperm(n, generator=g)
                tot = 0.0
                for s in range(0, n, batch):
                    idx = perm[s:s + batch]
                    z = self.trunk(Xt[idx])
                    logits = -torch.cdist(z, self.anchors) ** 2
                    loss = nn.functional.cross_entropy(logits, yt[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
                    self.trunk.project()
                    if mask is not None:          # dropped weights stay dropped
                        with torch.no_grad():
                            for lin, m in zip(self.trunk.layers, mask):
                                lin.weight.mul_(m)
                    tot += float(loss.detach()) * len(idx)
                if verbose and (ep % 10 == 0 or ep == n_ep - 1):
                    print(f"    epoch {ep:3d}  loss {tot / n:.4f}")

        _epochs(epochs)
        if prune and prune > 0:
            # per-layer magnitude pruning, uniform rate, one mask per round;
            # the spectral projection is multiplicative and preserves zeros
            keep_frac = (1.0 - float(prune)) ** (1.0 / max(prune_rounds, 1))
            mask = [torch.ones_like(lin.weight) for lin in self.trunk.layers]
            for _ in range(max(prune_rounds, 1)):
                with torch.no_grad():
                    for k, lin in enumerate(self.trunk.layers):
                        live = mask[k].bool()
                        n_live = int(live.sum())
                        n_keep = max(1, int(round(n_live * keep_frac)))
                        w = lin.weight.abs()[live]
                        thr = torch.topk(w, n_keep, largest=True).values.min()
                        mask[k] = (lin.weight.abs() >= thr).to(lin.weight.dtype) * mask[k]
                        lin.weight.mul_(mask[k])
                _epochs(epochs if prune_epochs is None else prune_epochs, mask)
            self.trunk_mask = [m.clone() for m in mask]
        from .proximity import make_prox
        kw = {"k_per_class": self.k_centroids} if self.prox_name == "kmeans" else {}
        self.prox = make_prox(self.prox_name, seed=self.seed, **kw).fit(X, y)

    # ------------------------------------------------------------------ #
    # the contract's free parameters: solved, never penalized
    # ------------------------------------------------------------------ #

    def solve_thresholds(self, Xval, yval, p_target: float = 0.95,
                         n_grid: int = 14, require: bool = True,
                         q_cov: float = 0.995):
        """Proximity radii (rho, rho_in) are solved as 'cover my own correct val,
        nothing more' -- the q_cov quantile of own correct distances. The margin
        threshold tau is then chosen to maximize coverage subject to precision
        >= p_target. Precision cannot see future negatives; proximity is the
        defense against them, so it must not be maximized against precision."""
        d1, m, pred, d_in, _, _ = self.evaluate(Xval)
        correct = pred == yval
        if correct.sum() == 0:
            return None
        # conformal: P(fresh own-correct sample accepted by these radii) is
        # >= q_cov / q_cov_in, finite-sample, under exchangeability
        self.rho = conformal_quantile(d1[correct], 1.0 - q_cov)
        self.rho_in = conformal_quantile(d_in[correct], 1.0 - self.q_cov_in)
        finite_m = m[np.isfinite(m)]
        tau_grid = (np.quantile(finite_m, np.linspace(0.0, 0.95, n_grid))
                    if len(finite_m) else np.array([0.0]))
        best, fallback = None, None
        for tau in tau_grid:
            acc_mask = (d1 <= self.rho) & (m > tau) & (d_in <= self.rho_in)
            if acc_mask.sum() == 0:
                continue
            cov = acc_mask.mean()
            prec = correct[acc_mask].mean()
            if prec >= p_target and (best is None or cov > best[0]):
                best = (cov, prec, float(tau))
            if fallback is None or (prec, cov) > (fallback[1], fallback[0]):
                fallback = (cov, prec, float(tau))
        if best is None:
            if require or fallback is None:
                return None
            best = fallback
        self.tau = best[2]
        return best[0], best[1]

    def freeze(self, Xval, yval):
        """Exact Lipschitz constant; store birth-val sufficient statistics
        (d1, margin, d_in, correct) -- the organ's license for future tightening."""
        self.L = self._lipschitz_exact()
        d1, m, pred, d_in, _, _ = self.evaluate(Xval)
        self.stats = {"d1": d1, "margin": m, "d_in": d_in, "correct": pred == yval}
        self.frozen = True

    # ------------------------------------------------------------------ #
    # monotone tightening: shrink the accepted set, never flip an answer
    # ------------------------------------------------------------------ #

    def tighten(self, new_rho: float, new_tau: float, new_rho_in: float):
        assert self.frozen
        assert (new_rho <= self.rho + 1e-12 and new_tau >= self.tau - 1e-12
                and new_rho_in <= self.rho_in + 1e-12), "tightening must be monotone"
        self.rho = min(new_rho, self.rho)
        self.tau = max(new_tau, self.tau)
        self.rho_in = min(new_rho_in, self.rho_in)

    def solve_tightening(self, d1_fa, m_fa, din_fa, self_budget: float = 0.02,
                         n_grid: int = 8):
        """Given false-accept stats measured on foreign data, choose the monotone
        (rho', tau', rho_in') excluding the most false accepts subject to an
        estimated self-coverage loss <= self_budget (from stored birth stats)."""
        s = self.stats
        own_acc = self.accept(s["d1"], s["margin"], s["d_in"])
        n_own = max(int(own_acc.sum()), 1)
        fa_acc = self.accept(d1_fa, m_fa, din_fa)
        if fa_acc.sum() == 0:
            return 0, 0.0

        def grid_down(vals, current):
            g = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, n_grid)))
            return [*g[g <= current], current]

        def grid_up(vals, current):
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                return [current]
            g = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, n_grid)))
            return [*g[g >= current], current]

        rho_grid = grid_down(d1_fa[fa_acc], self.rho)
        rin_grid = grid_down(din_fa[fa_acc], self.rho_in)
        tau_grid = grid_up(m_fa[fa_acc], self.tau)
        best = (0, self.rho, self.tau, self.rho_in, 0.0)
        for rho in rho_grid:
            for tau in tau_grid:
                for rin in rin_grid:
                    keep_own = ((s["d1"] <= rho) & (s["margin"] > tau)
                                & (s["d_in"] <= rin) & own_acc)
                    self_loss = 1.0 - keep_own.sum() / n_own
                    if self_loss > self_budget:
                        continue
                    excl = int((fa_acc & ~((d1_fa <= rho) & (m_fa > tau)
                                           & (din_fa <= rin))).sum())
                    if excl > best[0] or (excl == best[0] and self_loss < best[4]):
                        best = (excl, float(rho), float(tau), float(rin), self_loss)
        if best[0] > 0:
            self.tighten(best[1], best[2], best[3])
        return best[0], best[4]

    def n_params(self) -> int:
        return (sum(p.numel() for p in self.trunk.parameters())
                + self.anchors.numel())
