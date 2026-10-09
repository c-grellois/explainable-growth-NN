"""Datasets for the contracted-growth benchmarks.

All real datasets load from the local openml / sklearn cache (verified present), so
everything runs offline. Features are min-max scaled on train statistics to [0,1]^d
(the contracts convention) unless noted.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# toy: multi-modal gaussian classes in 2-D
# --------------------------------------------------------------------------- #

def toy_modes(seed: int = 0, n_per_mode: int = 400, spread: float = 0.09):
    """Three classes, each with 2-3 well-separated modes, in [0,1]^2.

    Multi-modality is the point: a single prototypical organ with one anchor per
    class and a capped-Lipschitz trunk cannot fold distant modes onto one anchor,
    so its abstention concentrates on the uncovered modes -- exactly the need
    field growth is supposed to answer.
    """
    rng = np.random.default_rng(seed)
    # (cx, cy, class)
    modes = [
        (0.15, 0.20, 0), (0.85, 0.80, 0),                    # class 0: two far modes
        (0.20, 0.85, 1), (0.80, 0.15, 1), (0.50, 0.50, 1),   # class 1: three modes
        (0.50, 0.10, 2), (0.10, 0.55, 2),                    # class 2: two modes
    ]
    X, y, m = [], [], []
    for mi, (cx, cy, c) in enumerate(modes):
        X.append(rng.normal([cx, cy], spread, size=(n_per_mode, 2)))
        y.append(np.full(n_per_mode, c))
        m.append(np.full(n_per_mode, mi))
    X = np.clip(np.concatenate(X), 0.0, 1.0)
    y = np.concatenate(y).astype(np.int64)
    m = np.concatenate(m).astype(np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], m[perm], 3


# --------------------------------------------------------------------------- #
# real data (cached openml / sklearn)
# --------------------------------------------------------------------------- #

OPENML_SETS = ("mnist", "fashion", "kmnist", "letter", "satimage", "robot")


def load_real(name: str):
    """(X, y, n_classes). OpenML sets are cached as .npz in
    <sklearn data home>/cgrow_cache/; delete it if _load_real changes."""
    import os
    from sklearn.datasets import get_data_home

    if name not in OPENML_SETS:
        return _load_real(name)
    path = os.path.join(get_data_home(), "cgrow_cache", f"{name}.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            X, y = f["X"], f["y"]
        return X, y, int(y.max()) + 1
    X, y, n = _load_real(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=X, y=y)
    return X, y, n


def _load_real(name: str):
    from sklearn.datasets import fetch_openml, load_digits

    if name == "digits":
        d = load_digits()
        X, y = d.data.astype(np.float64), d.target.astype(np.int64)
    elif name == "mnist":
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X, y = d.data.astype(np.float64), d.target.astype(np.int64)
    elif name == "fashion":
        d = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
        X, y = d.data.astype(np.float64), d.target.astype(np.int64)
    elif name == "kmnist":
        d = fetch_openml("Kuzushiji-MNIST", version=1, as_frame=False, parser="liac-arff")
        X, y = d.data.astype(np.float64), d.target.astype(np.int64)
    elif name == "letter":
        d = fetch_openml("letter", version=1, as_frame=False, parser="liac-arff")
        X = d.data.astype(np.float64)
        y = np.searchsorted(np.unique(d.target), d.target).astype(np.int64)
    elif name == "letter10":
        X, y, _ = load_real("letter")
        keep = y < 10
        return X[keep], y[keep], 10
    elif name == "satimage":
        d = fetch_openml("satimage", version=1, as_frame=False, parser="liac-arff")
        X = d.data.astype(np.float64)
        y = np.searchsorted(np.unique(d.target), d.target).astype(np.int64)
    elif name == "robot":
        d = fetch_openml("wall-robot-navigation", version=1, as_frame=False,
                         parser="liac-arff")
        X = d.data.astype(np.float64)
        y = np.searchsorted(np.unique(d.target), d.target).astype(np.int64)
    else:
        raise ValueError(name)
    return X, y, int(y.max()) + 1


def prepare(X, y, seed: int, test_frac: float = 0.25, val_frac: float = 0.15,
            max_train: int | None = None):
    """Split train/val/test; min-max scale to [0,1]^d on train statistics."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_test = int(test_frac * len(X))
    te, rest = perm[:n_test], perm[n_test:]
    n_val = int(val_frac * len(rest))
    va, tr = rest[:n_val], rest[n_val:]
    if max_train is not None and len(tr) > max_train:
        tr = tr[:max_train]
    lo, hi = X[tr].min(axis=0), X[tr].max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    Xs = np.clip((X - lo) / span, 0.0, 1.0)
    return (Xs[tr], y[tr]), (Xs[va], y[va]), (Xs[te], y[te])


def split_tasks(y: np.ndarray, classes_per_task: int = 2):
    """Class-incremental task masks: task t owns classes [t*k, (t+1)*k)."""
    n_classes = int(y.max()) + 1
    tasks = []
    for t in range(0, n_classes, classes_per_task):
        cls = list(range(t, min(t + classes_per_task, n_classes)))
        tasks.append(cls)
    return tasks
