"""Baselines: plain MLP (static), and sequential fine-tune / EWC / replay for the
class-incremental benchmark. Single head, no task oracle at test time (class-IL,
the brutal setting)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

DTYPE = torch.float32


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        dims = [in_dim, *hidden]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1], dtype=DTYPE), nn.ReLU()]
        layers += [nn.Linear(dims[-1], n_classes, dtype=DTYPE)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def _epochs(model, X, y, epochs, lr, batch, seed, extra_loss=None, buffer=None):
    Xt, yt = torch.as_tensor(X, dtype=DTYPE), torch.as_tensor(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for s in range(0, len(Xt), batch):
            idx = perm[s:s + batch]
            xb, yb = Xt[idx], yt[idx]
            if buffer is not None and len(buffer[0]):
                bi = torch.randint(0, len(buffer[0]), (min(batch, len(buffer[0])),),
                                   generator=g)
                xb = torch.cat([xb, torch.as_tensor(buffer[0][bi.numpy()], dtype=DTYPE)])
                yb = torch.cat([yb, torch.as_tensor(buffer[1][bi.numpy()])])
            loss = nn.functional.cross_entropy(model(xb), yb)
            if extra_loss is not None:
                loss = loss + extra_loss(model)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_plain(X, y, n_classes, hidden=(256,), epochs=30, lr=1e-3, batch=256, seed=0):
    model = MLP(X.shape[1], hidden, n_classes, seed=seed)
    return _epochs(model, X, y, epochs, lr, batch, seed)


@torch.no_grad()
def accuracy(model, X, y):
    pred = model(torch.as_tensor(X, dtype=DTYPE)).argmax(1).numpy()
    return float((pred == y).mean()), pred


def _fisher_diag(model, X, y, n_samples=1024, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    fisher = [torch.zeros_like(p) for p in model.parameters()]
    for i in idx:
        model.zero_grad()
        logp = nn.functional.log_softmax(
            model(torch.as_tensor(X[i:i + 1], dtype=DTYPE)), dim=1)[0, y[i]]
        logp.backward()
        for f, p in zip(fisher, model.parameters()):
            if p.grad is not None:
                f += p.grad.detach() ** 2
    return [f / len(idx) for f in fisher]


def run_sequential(method: str, tasks_data, n_classes, in_dim, hidden=(256,),
                   epochs=20, lr=1e-3, batch=256, seed=0,
                   ewc_lambda=5000.0, buffer_per_class=20):
    """tasks_data: list of (Xtr, ytr). Returns trained model plus per-task test
    hooks left to the caller. method in {finetune, ewc, replay}."""
    model = MLP(in_dim, hidden, n_classes, seed=seed)
    anchors_prev = []          # (params_star, fisher) per past task, for EWC
    buf_X = np.empty((0, in_dim)); buf_y = np.empty((0,), dtype=np.int64)
    for t, (Xtr, ytr) in enumerate(tasks_data):
        extra = None
        if method == "ewc" and anchors_prev:
            def extra(m, anchors_prev=anchors_prev):
                loss = 0.0
                for params_star, fisher in anchors_prev:
                    for p, ps, f in zip(m.parameters(), params_star, fisher):
                        loss = loss + (f * (p - ps) ** 2).sum()
                return ewc_lambda * loss
        buffer = (buf_X, buf_y) if method == "replay" and len(buf_X) else None
        _epochs(model, Xtr, ytr, epochs, lr, batch, seed + t,
                extra_loss=extra, buffer=buffer)
        if method == "ewc":
            params_star = [p.detach().clone() for p in model.parameters()]
            fisher = _fisher_diag(model, Xtr, ytr, seed=seed + t)
            anchors_prev.append((params_star, fisher))
        if method == "replay":
            rng = np.random.default_rng(seed + t)
            for c in np.unique(ytr):
                sel = np.flatnonzero(ytr == c)
                take = rng.choice(sel, size=min(buffer_per_class, len(sel)), replace=False)
                buf_X = np.concatenate([buf_X, Xtr[take]])
                buf_y = np.concatenate([buf_y, ytr[take]])
    return model
