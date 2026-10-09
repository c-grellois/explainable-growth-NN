"""Pluggable proximity geometries for the organ's OOD contract.

The proximity coordinate d_in answers "is this input inside the organ's known
territory?" — the OOD-sensitive half of the contract, and the measured bottleneck
of class-incremental performance (false accepts on classes an organ never saw).

One interface, three geometries:

  KMeansProx  raw input-space centroids with per-cluster radii (the v1 baseline)
  PCAProx     per-class PCA: subspace residual + whitened in-subspace distance,
              fitted in closed form -- no gradient training, no penalty terms
  AEProx      spectrally-capped autoencoder; d_in = reconstruction error

All scores are dimensionless, calibrated so the organ's own data sits near <= 1.
Each geometry publishes a Lipschitz bound (|d_in(x') - d_in(x)| <= L ||x' - x||)
so the acceptance rule stays a certificate, and a sample() method so the organ's
spec remains a data generator (the replacement experiment's requirement).

No geometry stores raw samples: centroids, means, principal axes, eigenvalues,
network weights, and calibration scalars only.
"""
from __future__ import annotations

import numpy as np
import torch

from .organ import SpectralMLP, DTYPE

CAL_Q = 0.9      # own-data quantile that defines score = 1


class KMeansProx:
    """v1 geometry: min over centroids of ||x - c_j|| / r_j (radii shrunk)."""

    name = "kmeans"

    def __init__(self, k_per_class: int = 5, seed: int = 0):
        self.k = k_per_class
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.cluster import KMeans
        cents, radii, labels = [], [], []
        for c in np.unique(y):
            Xc = X[y == c]
            k = min(self.k, len(Xc))
            if k <= 1:
                ctr = Xc.mean(axis=0, keepdims=True)
                cents.append(ctr)
                radii.append([np.linalg.norm(Xc - ctr, axis=1).mean() + 1e-6])
                labels.append([c])
                continue
            km = KMeans(n_clusters=k, n_init=3, random_state=self.seed).fit(Xc)
            cents.append(km.cluster_centers_)
            d = np.linalg.norm(Xc - km.cluster_centers_[km.labels_], axis=1)
            radii.append([max(np.quantile(d[km.labels_ == j], CAL_Q), 1e-6)
                          if (km.labels_ == j).any() else 1e-6 for j in range(k)])
            labels.append([c] * k)
        self.cents = np.concatenate(cents)
        r = np.concatenate(radii)
        self.radii = 0.5 * (r + r.mean())          # shrink noisy radii
        self.labels = np.concatenate(labels).astype(np.int64)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - self.cents[None], axis=2) / self.radii
        return d.min(axis=1)

    def lipschitz(self) -> float:
        return 1.0 / float(self.radii.min())

    def sample(self, n: int, rng: np.random.Generator, rho: float = 1.0):
        idx = rng.integers(0, len(self.cents), size=n)
        scale = np.maximum(rho * self.radii[idx] / np.sqrt(self.cents.shape[1]), 1e-3)
        X = self.cents[idx] + rng.normal(0, 1.0, (n, self.cents.shape[1])) * scale[:, None]
        return np.clip(X, 0.0, 1.0), self.labels[idx]

    def spec_size(self) -> int:
        return self.cents.size + self.radii.size


class PCAProx:
    """Per-class PCA territory: score_c = max(residual/r_q, whitened/w_q),
    d_in = min_c score_c. Fitted in closed form from class means and top-q
    principal axes; eigenvalues floored for a finite Lipschitz constant."""

    name = "pca"

    def __init__(self, n_components: int = 24, seed: int = 0):
        self.q = n_components
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes, self.mu, self.V, self.lam = [], [], [], []
        self.r_q, self.w_q = [], []
        for c in np.unique(y):
            Xc = X[y == c]
            if len(Xc) < 3:      # too few samples for a territory: skip the
                continue         # class (it simply has no territory yet)
            q = min(self.q, Xc.shape[1] - 1, len(Xc) - 1)
            mu = Xc.mean(axis=0)
            Z = Xc - mu
            # economy SVD: principal axes and variances
            _, S, Vt = np.linalg.svd(Z, full_matrices=False)
            lam = (S[:q] ** 2) / max(len(Xc) - 1, 1)
            lam = np.maximum(lam, max(1e-3 * lam[0], 1e-8))
            V = Vt[:q]
            proj = Z @ V.T
            resid = np.linalg.norm(Z - proj @ V, axis=1)
            white = np.sqrt(((proj ** 2) / lam).sum(axis=1))
            self.classes.append(c); self.mu.append(mu); self.V.append(V)
            self.lam.append(lam)
            self.r_q.append(max(np.quantile(resid, CAL_Q), 1e-6))
            self.w_q.append(max(np.quantile(white, CAL_Q), 1e-6))
        self.classes = np.asarray(self.classes, dtype=np.int64)
        return self

    def project(self, X: np.ndarray) -> list:
        """Per class: coordinates along the principal axes, and distance from
        that subspace. Shared by score() and an organ's veto routing."""
        parts = []
        for mu, V in zip(self.mu, self.V):
            Z = X - mu
            proj = Z @ V.T
            parts.append((proj, np.linalg.norm(Z - proj @ V, axis=1)))
        return parts

    def score(self, X: np.ndarray, parts: list | None = None) -> np.ndarray:
        parts = self.project(X) if parts is None else parts
        scores = np.full(len(X), np.inf)
        for (proj, res), lam, rq, wq in zip(parts, self.lam, self.r_q, self.w_q):
            resid = res / rq
            white = np.sqrt(((proj ** 2) / lam).sum(axis=1)) / wq
            scores = np.minimum(scores, np.maximum(resid, white))
        return scores

    def lipschitz(self) -> float:
        # residual term is 1-Lipschitz; whitened term is (1/sqrt(lam_min))-Lipschitz;
        # max and min of L-Lipschitz functions stay L-Lipschitz
        return max(max(1.0 / rq, 1.0 / (wq * np.sqrt(lam.min())))
                   for rq, wq, lam in zip(self.r_q, self.w_q, self.lam))

    def sample_whitened(self, class_idx: int, n: int,
                        rng: np.random.Generator) -> np.ndarray:
        """Exact-spec positives in whitened subspace coords (for vetoes):
        the single-gaussian model asserts N(0, I)."""
        return rng.normal(0, 1.0, (n, self.V[class_idx].shape[0]))

    def sample(self, n: int, rng: np.random.Generator, rho: float = 1.0):
        d = self.mu[0].shape[0]
        which = rng.integers(0, len(self.classes), size=n)
        X = np.empty((n, d))
        for i, w in enumerate(which):
            lam, V, mu, rq = self.lam[w], self.V[w], self.mu[w], self.r_q[w]
            z = rng.normal(0, np.sqrt(lam) * rho)
            res = rng.normal(0, rho * rq / np.sqrt(d), size=d)
            X[i] = mu + z @ V + res
        return np.clip(X, 0.0, 1.0), self.classes[which]

    def spec_size(self) -> int:
        return sum(m.size + v.size + l.size + 2
                   for m, v, l in zip(self.mu, self.V, self.lam))


class AEProx:
    """Spectrally-capped autoencoder territory: d_in = ||x - AE(x)|| / cal.
    Lipschitz bound (1 + L_enc * L_dec) / cal via the same cap machinery as the
    trunks. Latents are summarized per class (mean, sd) so sample() can decode."""

    name = "ae"

    def __init__(self, latent: int = 32, hidden: int = 256, cap: float = 4.0,
                 epochs: int = 15, seed: int = 0):
        self.latent, self.hidden, self.cap = latent, hidden, cap
        self.epochs, self.seed = epochs, seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        d = X.shape[1]
        self.enc = SpectralMLP([d, self.hidden, self.latent], cap=self.cap,
                               seed=self.seed)
        self.dec = SpectralMLP([self.latent, self.hidden, d], cap=self.cap,
                               seed=self.seed + 1)
        Xt = torch.as_tensor(X, dtype=DTYPE)
        opt = torch.optim.Adam([*self.enc.parameters(), *self.dec.parameters()],
                               lr=2e-3)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for s in range(0, len(Xt), 256):
                xb = Xt[perm[s:s + 256]]
                loss = torch.nn.functional.mse_loss(self.dec(self.enc(xb)), xb)
                opt.zero_grad(); loss.backward(); opt.step()
                self.enc.project(); self.dec.project()
        with torch.no_grad():
            rec = np.linalg.norm(self.dec(self.enc(Xt)).numpy() - X, axis=1)
            z = self.enc(Xt).numpy()
        self.cal = max(float(np.quantile(rec, CAL_Q)), 1e-6)
        self.classes = np.unique(y)
        self.z_mu = np.stack([z[y == c].mean(axis=0) for c in self.classes])
        self.z_sd = np.stack([z[y == c].std(axis=0) + 1e-6 for c in self.classes])
        self._L = (1.0 + self.enc.lipschitz_exact() * self.dec.lipschitz_exact()) / self.cal
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(X, dtype=DTYPE)
        rec = self.dec(self.enc(Xt)).numpy()
        return np.linalg.norm(rec - X, axis=1) / self.cal

    def lipschitz(self) -> float:
        return self._L

    @torch.no_grad()
    def sample(self, n: int, rng: np.random.Generator, rho: float = 1.0):
        which = rng.integers(0, len(self.classes), size=n)
        z = self.z_mu[which] + rng.normal(0, 1.0, (n, self.latent)) * self.z_sd[which]
        X = self.dec(torch.as_tensor(z, dtype=DTYPE)).numpy()
        return np.clip(X, 0.0, 1.0), self.classes[which]

    def spec_size(self) -> int:
        return (sum(p.numel() for p in self.enc.parameters())
                + sum(p.numel() for p in self.dec.parameters())
                + self.z_mu.size + self.z_sd.size + 1)


class MixProx(PCAProx):
    """Multimodal territory: the PCA basis and whitening of PCAProx, with a
    small diagonal Gaussian mixture fitted on the WHITENED in-subspace
    coordinates of each class. Same attributes as PCAProx (mu, V, lam, r_q,
    w_q) so vetoes and specs stay compatible; what changes is fidelity --
    score, sample, and the veto positives all follow the mixture instead of a
    single gaussian. Spec grows by k * (2q + 1) scalars per class."""

    name = "mix"

    def __init__(self, n_components: int = 24, k_mix: int = 5, seed: int = 0):
        super().__init__(n_components=n_components, seed=seed)
        self.k_mix = k_mix

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        from sklearn.mixture import GaussianMixture
        self.gmm = []
        for i, c in enumerate(self.classes):
            Z = (X[y == c] - self.mu[i]) @ self.V[i].T / np.sqrt(self.lam[i])
            k = min(self.k_mix, max(len(Z) // 25, 1))
            gm = GaussianMixture(n_components=k, covariance_type="diag",
                                 reg_covar=1e-3, random_state=self.seed,
                                 n_init=1).fit(Z)
            self.gmm.append(gm)
            # recalibrate the whitened score against the mixture
            w = self._w_score(Z, gm)
            self.w_q[i] = max(np.quantile(w, CAL_Q), 1e-6)
        return self

    @staticmethod
    def _w_score(Zw, gm):
        """Min over components of the per-dim-normalized distance."""
        d = np.stack([np.sqrt((((Zw - m) ** 2) / v).sum(axis=1))
                      for m, v in zip(gm.means_, gm.covariances_)])
        return d.min(axis=0)

    def score(self, X: np.ndarray, parts: list | None = None) -> np.ndarray:
        parts = self.project(X) if parts is None else parts
        scores = np.full(len(X), np.inf)
        for (proj, res), lam, rq, wq, gm in zip(parts, self.lam, self.r_q,
                                                self.w_q, self.gmm):
            resid = res / rq
            white = self._w_score(proj / np.sqrt(lam), gm) / wq
            scores = np.minimum(scores, np.maximum(resid, white))
        return scores

    def lipschitz(self) -> float:
        L = 0.0
        for rq, wq, lam, gm in zip(self.r_q, self.w_q, self.lam, self.gmm):
            sd_min = float(np.sqrt(gm.covariances_.min()))
            L = max(L, 1.0 / rq,
                    1.0 / (wq * sd_min * np.sqrt(float(lam.min()))))
        return L

    def sample_whitened(self, class_idx: int, n: int,
                        rng: np.random.Generator) -> np.ndarray:
        """Exact-spec positives in whitened subspace coords (for vetoes)."""
        gm = self.gmm[class_idx]
        comp = rng.choice(len(gm.weights_), size=n, p=gm.weights_)
        return (gm.means_[comp]
                + rng.normal(0, 1.0, (n, gm.means_.shape[1]))
                * np.sqrt(gm.covariances_[comp]))

    def sample(self, n: int, rng: np.random.Generator, rho: float = 1.0):
        d = self.mu[0].shape[0]
        which = rng.integers(0, len(self.classes), size=n)
        X = np.empty((n, d))
        for i, w in enumerate(which):
            zw = self.sample_whitened(w, 1, rng)[0] * rho
            z = zw * np.sqrt(self.lam[w])
            res = rng.normal(0, rho * self.r_q[w] / np.sqrt(d), size=d)
            X[i] = self.mu[w] + z @ self.V[w] + res
        return np.clip(X, 0.0, 1.0), self.classes[which]

    def spec_size(self) -> int:
        base = super().spec_size()
        return base + sum(g.means_.size + g.covariances_.size + g.weights_.size
                          for g in self.gmm)


class ComboProx:
    """Intersection of the kmeans and PCA territories: score = max of the two
    calibrated scores (accept requires being inside BOTH). Lipschitz = max."""

    name = "combo"

    def __init__(self, seed: int = 0):
        self.km = KMeansProx(k_per_class=25, seed=seed)
        self.pca = PCAProx(n_components=24, seed=seed)

    def fit(self, X, y):
        self.km.fit(X, y); self.pca.fit(X, y)
        return self

    def score(self, X):
        return np.maximum(self.km.score(X), self.pca.score(X))

    def lipschitz(self):
        return max(self.km.lipschitz(), self.pca.lipschitz())

    def sample(self, n, rng, rho: float = 1.0):
        return self.pca.sample(n, rng, rho)

    def spec_size(self):
        return self.km.spec_size() + self.pca.spec_size()


def _corrupt(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic outliers from own data: patch shuffle (if square), heavy noise,
    uniform mixup, feature shuffle -- outlier exposure without outliers."""
    n, d = X.shape
    out = X.copy()
    kind = rng.integers(0, 4, size=n)
    s = int(np.sqrt(d))
    square = s * s == d and s >= 8
    for i in range(n):
        k = kind[i]
        if k == 0 and square:                     # shuffle 4x4 grid of patches
            img = out[i].reshape(s, s)
            p = s // 4
            blocks = [img[r * p:(r + 1) * p, c * p:(c + 1) * p].copy()
                      for r in range(4) for c in range(4)]
            perm = rng.permutation(16)
            for j, b in enumerate(perm):
                r, c = divmod(j, 4)
                img[r * p:(r + 1) * p, c * p:(c + 1) * p] = blocks[b]
            out[i] = img.ravel()
        elif k == 0:
            out[i] = out[i][rng.permutation(d)]
        elif k == 1:                              # heavy gaussian noise
            out[i] = np.clip(out[i] + rng.normal(0, 0.3, d), 0, 1)
        elif k == 2:                              # mixup with uniform
            out[i] = 0.5 * out[i] + 0.5 * rng.uniform(0, 1, d)
        else:                                     # global feature shuffle
            out[i] = out[i][rng.permutation(d)]
    return out


class DiscProx:
    """Discriminative territory: a capped MLP trained own-vs-corrupted(own).
    Score = P(outlier), calibrated on own data. No foreign data is ever seen;
    whether synthetic corruption transfers to real foreign classes is exactly
    what the shoot-out measures."""

    name = "disc"

    def __init__(self, hidden: int = 128, cap: float = 4.0, epochs: int = 10,
                 seed: int = 0):
        self.hidden, self.cap, self.epochs, self.seed = hidden, cap, epochs, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        Xneg = _corrupt(X, rng)
        Xall = np.concatenate([X, Xneg])
        t = torch.as_tensor(Xall, dtype=DTYPE)
        lab = torch.cat([torch.zeros(len(X)), torch.ones(len(Xneg))])
        self.net = SpectralMLP([X.shape[1], self.hidden, 1], cap=self.cap,
                               seed=self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(t), generator=g)
            for s in range(0, len(t), 256):
                idx = perm[s:s + 256]
                logit = self.net(t[idx]).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, lab[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                self.net.project()
        with torch.no_grad():
            p_own = torch.sigmoid(self.net(torch.as_tensor(X, dtype=DTYPE))).numpy().ravel()
        self.cal = max(float(np.quantile(p_own, CAL_Q)), 1e-3)
        self.classes = np.unique(y)
        return self

    @torch.no_grad()
    def score(self, X):
        p = torch.sigmoid(self.net(torch.as_tensor(X, dtype=DTYPE))).numpy().ravel()
        return p / self.cal

    def lipschitz(self):
        return self.net.lipschitz_exact() * 0.25 / self.cal

    def sample(self, n, rng, rho: float = 1.0):
        raise NotImplementedError("disc territory is a boundary, not a generator")

    def spec_size(self):
        return sum(p.numel() for p in self.net.parameters()) + 1


class UnionProx:
    """Union of existing territories: score = min over members. Used by the
    consolidated (adolescent) organ, whose territory is by construction the
    union of the parts it inherits -- each member was calibrated on real data
    at its organ's birth, so no synthetic/real gap is introduced."""

    name = "union"

    def __init__(self, members):
        self.members = list(members)

    def fit(self, X, y):
        return self

    def score(self, X):
        s = self.members[0].score(X)
        for m in self.members[1:]:
            s = np.minimum(s, m.score(X))
        return s

    def lipschitz(self):
        return max(m.lipschitz() for m in self.members)

    def sample(self, n, rng, rho: float = 1.0):
        per = max(n // len(self.members), 1)
        Xs, ys = zip(*(m.sample(per, rng, rho) for m in self.members))
        return np.concatenate(Xs), np.concatenate(ys)

    def spec_size(self):
        return sum(m.spec_size() for m in self.members)


def merge_pca_family(members):
    """Merge PCAProx/MixProx territories by concatenation: the union of PCA
    territories IS a PCA territory with more classes. Unlike the opaque
    UnionProx wrapper, the merged prox is first-class -- it supports vetoes
    (whitened per-territory projection), exact-spec sampling, and tightening.
    Members must serve disjoint classes and share a family."""
    names = {m.name for m in members}
    assert names <= {"pca", "mix"} and len(names) == 1, names
    name = names.pop()
    out = (MixProx() if name == "mix" else PCAProx())
    out.classes = np.concatenate([m.classes for m in members])
    for attr in ["mu", "V", "lam", "r_q", "w_q"]:
        setattr(out, attr, [x for m in members for x in getattr(m, attr)])
    if name == "mix":
        out.gmm = [g for m in members for g in m.gmm]
    return out


PROX = {"kmeans": KMeansProx, "pca": PCAProx, "ae": AEProx,
        "combo": ComboProx, "disc": DiscProx, "mix": MixProx}


def make_prox(name: str, seed: int = 0, **kw):
    return PROX[name](seed=seed, **kw)
