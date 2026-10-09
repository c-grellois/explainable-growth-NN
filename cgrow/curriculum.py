"""Class-order curricula for the flat class-incremental protocol.

A curriculum changes the PROTOCOL: results under it are a separate setting from
the random-order class-IL anchors and must never be placed in the same table.
Difficulty is measured before any organ exists, from the frozen features (or raw
pixels) with a nearest-class-mean fit on the training split: per-class error is
the difficulty. This uses every class's labels up front (an oracle curriculum);
it asks "does order matter and in which direction", not "can the stream be
ordered online".

Policies
  random   the protocol's own order (seed + 400 permutation for CIFAR-style
           caches; ascending class index for the MLP benchmarks) -- unchanged
  easy     increasing NCM difficulty (least confusable classes first)
  hard     decreasing NCM difficulty
  grouped  confusable classes share a task: CIFAR-100 by superclass (each task
           = the fine classes of consecutive superclasses, superclass order
           seeded); otherwise greedy pairing on the NCM confusion matrix
  spread   the opposite of grouped: the most confusable pairs are placed in
           different tasks, as far apart as possible
"""
from __future__ import annotations

import os
import pickle

import numpy as np

POLICIES = ("random", "easy", "hard", "grouped", "spread")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ncm_confusion(X, y, classes):
    """Confusion matrix of cosine nearest-class-mean fitted and scored on the
    same (training) data: rows = true class, columns = predicted."""
    X = np.asarray(X, dtype=np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    M = np.stack([Xn[y == c].mean(axis=0) for c in classes])
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    pred = np.asarray(classes)[(Xn @ M.T).argmax(axis=1)]
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    conf = np.zeros((K, K))
    for t, p in zip(y, pred):
        conf[idx[int(t)], idx[int(p)]] += 1
    return conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)


def class_difficulty(X, y, classes=None):
    """Per-class NCM error on the training split, in [0, 1]. Returns
    (classes, difficulty) with classes sorted ascending."""
    classes = sorted(int(c) for c in (classes if classes is not None
                                      else np.unique(y)))
    conf = _ncm_confusion(X, y, classes)
    return classes, 1.0 - np.diag(conf)


def cifar_superclass_map():
    """fine class id -> coarse class id (0..19) from the CIFAR-100 train file."""
    d = pickle.load(open(os.path.join(HERE, "data", "cifar-100-python", "train"),
                         "rb"), encoding="latin1")
    f = np.asarray(d["fine_labels"]); c = np.asarray(d["coarse_labels"])
    return {int(k): int(c[f == k][0]) for k in range(100)}


def cifar_superclass_names():
    meta = pickle.load(open(os.path.join(HERE, "data", "cifar-100-python", "meta"),
                            "rb"), encoding="latin1")
    return [s.replace("_", " ") for s in meta["coarse_label_names"]]


def _greedy_pairs(conf, classes):
    """Pair classes greedily by symmetric confusion (most confusable first)."""
    K = len(classes)
    S = conf + conf.T
    np.fill_diagonal(S, -1)
    free = set(range(K))
    pairs = []
    while len(free) > 1:
        best, bij = -1, None
        for i in free:
            for j in free:
                if j > i and S[i, j] > best:
                    best, bij = S[i, j], (i, j)
        i, j = bij
        pairs.append((classes[i], classes[j]))
        free -= {i, j}
    if free:
        pairs.append((classes[free.pop()],))
    return pairs


def order_classes(policy: str, X, y, seed: int, inc: int,
                  superclass: dict | None = None):
    """Return the class ORDER (list of class ids) for a policy. Tasks are then
    consecutive chunks of `inc`. `random` returns None (caller keeps its own
    default order)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown order policy {policy!r}; choose from {POLICIES}")
    if policy == "random":
        return None
    classes, diff = class_difficulty(X, y)
    rng = np.random.default_rng(seed + 900)
    if policy in ("easy", "hard"):
        # tie-break randomly but reproducibly, then sort
        key = diff + 1e-6 * rng.random(len(diff))
        idx = np.argsort(key if policy == "easy" else -key)
        return [classes[i] for i in idx]
    conf = _ncm_confusion(X, y, classes)
    if policy == "grouped":
        if superclass is not None:
            supers = sorted(set(superclass[c] for c in classes))
            sorder = rng.permutation(len(supers))
            out = []
            for si in sorder:
                s = supers[si]
                members = [c for c in classes if superclass[c] == s]
                out.extend(sorted(members))
            return out
        pairs = _greedy_pairs(conf, classes)
        pairs = [pairs[i] for i in rng.permutation(len(pairs))]
        return [c for p in pairs for c in p]
    if policy == "spread":
        # most confusable pairs go to different tasks: place pair members
        # `inc * n_tasks/2` apart by round-robin over tasks
        pairs = _greedy_pairs(conf, classes)
        n_tasks = int(np.ceil(len(classes) / inc))
        slots = [[] for _ in range(n_tasks)]
        t = 0
        for p in pairs:
            for k, c in enumerate(p):
                # first member to task t, second to the task furthest from it
                tt = t if k == 0 else (t + n_tasks // 2) % n_tasks
                # find nearest task with room
                for step in range(n_tasks):
                    cand = (tt + step) % n_tasks
                    if len(slots[cand]) < inc:
                        slots[cand].append(c); break
            t = (t + 1) % n_tasks
        return [c for s in slots for c in s]
    raise AssertionError(policy)


def tasks_from_order(order, inc):
    return [sorted(int(c) for c in order[i:i + inc]) for i in range(0, len(order), inc)]
