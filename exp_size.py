"""B-size -- how big the grown network actually is: neurons and connections.

The organism is built exactly the way every other battery builds it
(`exp_lifecycle.build`, so the same config, class order, splits and seeds), then
measured and scored from the same object, so size and accuracy cannot drift
apart. Three things are counted separately, because they are different kinds of
thing and lumping them would flatter or inflate the total:

  trunks     the actual neural layers -- every organ's small MLP. Units are the
             hidden and feature units it computes; connections are the weights
             between them. The input layer is shared by every organ and is
             counted once for the organism.
  vetoes     the learned inhibitors: small nets living in a territory's own
             whitened coordinates, which can only remove inputs from what an
             organ claims.
  contract   anchors, territories (mean, principal axes, spreads) and the three
             thresholds. Stored numbers, not units: no activation is computed
             from them, they are compared against. This is also the part that
             doubles as a generator, so it is what replaces a replay buffer.

Run:  python exp_size.py --benchmark mnist --first-seed 2 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_lifecycle import build, MLP_CFG
from cgrow.adolescence import consolidate_protos
from cgrow.organism import Organism

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def count_net(net) -> dict:
    """Units and connections of a stack of linear layers. `units` counts the
    outputs of every layer (the things that compute); the input layer is the
    caller's business, since organs share it."""
    units = weights = biases = nonzero = 0
    in_dim = None
    for m in getattr(net, "layers", []):
        if not isinstance(m, nn.Linear):
            continue
        if in_dim is None:
            in_dim = int(m.in_features)
        units += int(m.out_features)
        weights += int(m.in_features) * int(m.out_features)
        nonzero += int((m.weight != 0).sum())      # pruning leaves exact zeros
        biases += int(m.bias.numel()) if m.bias is not None else 0
    return {"in_dim": in_dim, "units": units, "weights": weights,
            "nonzero_weights": nonzero, "biases": biases}


def prox_floats(prox) -> int:
    """Stored numbers in a territory: per class a mean, q principal axes, q
    eigenvalues and the two calibrated radii."""
    if prox is None:
        return 0
    if hasattr(prox, "members"):
        return sum(prox_floats(m) for m in prox.members)
    if hasattr(prox, "mu") and hasattr(prox, "V"):
        return int(sum(np.asarray(mu).size + np.asarray(V).size + np.asarray(lam).size + 2
                       for mu, V, lam in zip(prox.mu, prox.V, prox.lam)))
    return 0


def organ_size(o) -> dict:
    trunk = count_net(getattr(o, "trunk", None))
    vet = {"nets": len(o.vetoes), "units": 0, "weights": 0, "biases": 0}
    for v in o.vetoes:
        net = v[1]
        c = count_net(net)
        vet["units"] += c["units"]; vet["weights"] += c["weights"]; vet["biases"] += c["biases"]
    anchors = int(o.anchors.numel())
    terr = prox_floats(getattr(o, "prox", None))
    return {"classes": [int(c) for c in o.class_ids], "trunk": trunk, "vetoes": vet,
            "anchor_floats": anchors, "territory_floats": terr,
            "threshold_floats": 3,                      # rho, tau, rho_in
            "contract_floats": anchors + terr + 3}


def organism_size(org: Organism, in_dim: int) -> dict:
    per = [organ_size(o) for o in org.organs]
    t_units = sum(p["trunk"]["units"] for p in per)
    t_w = sum(p["trunk"]["weights"] for p in per)
    t_b = sum(p["trunk"]["biases"] for p in per)
    v_units = sum(p["vetoes"]["units"] for p in per)
    v_w = sum(p["vetoes"]["weights"] for p in per)
    v_b = sum(p["vetoes"]["biases"] for p in per)
    contract = sum(p["contract_floats"] for p in per)
    return {
        "n_organs": len(org.organs),
        "input_units": in_dim,                          # shared by every organ
        "trunk_units": t_units, "trunk_connections": t_w, "trunk_biases": t_b,
        "veto_nets": sum(p["vetoes"]["nets"] for p in per),
        "veto_units": v_units, "veto_connections": v_w, "veto_biases": v_b,
        "neurons_total": in_dim + t_units + v_units,
        "connections_total": t_w + v_w,
        "learned_parameters": t_w + t_b + v_w + v_b,    # weights and biases
        "contract_floats": contract,
        "stored_numbers_total": t_w + t_b + v_w + v_b + contract,
        "per_organ": per,
    }


def growth_flips(org: Organism, Xte: np.ndarray) -> dict:
    """The retention invariant, measured on this organism rather than cited:
    for every prefix of the cascade, an answer already given must survive the
    next organ being appended. `build` does not keep the live counter, so this
    is the same quantity checked afterwards over every prefix."""
    from cgrow.organism import ABSTAIN
    flips, per_step, prev = 0, [], None
    for k in range(1, len(org.organs) + 1):
        sub = Organism(); sub.organs = org.organs[:k]
        pred = sub.predict(Xte)["pred"]
        if prev is not None:
            f = int(((prev != ABSTAIN) & (pred != prev)).sum())
            flips += f; per_step.append(f)
        prev = pred
    return {"total": flips, "per_step": per_step, "steps_checked": len(per_step)}


def run(bench: str, seed: int) -> dict:
    t0 = time.time()
    org, splits, tasks, cfg, Xva_last, yva_last = build(bench, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    in_dim = int(Xtr.shape[1])
    grown = {"scores": org.scores(Xte, yte), "size": organism_size(org, in_dim),
             "growth_flips": growth_flips(org, Xte)}

    n_pool = max(200, min(4000, 40000 // max(len(org.organs), 1)))
    org2, rep = consolidate_protos(org, in_dim, cfg, seed,
                                   Xva_last=Xva_last, yva_last=yva_last,
                                   Xva=Xva, yva=yva, n_per_organ=n_pool)
    consolidated = {"accepted": bool(rep["accepted"]), "why": rep.get("why"),
                    "scores": org2.scores(Xte, yte),
                    "size": organism_size(org2, in_dim)}

    # the same-shape single network the baselines train, for scale
    b_hidden = cfg.get("b_hidden", (256,))
    n_cls = int(max(ytr.max(), yte.max())) + 1
    dims = [in_dim, *b_hidden, n_cls]
    base_w = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    base_u = sum(dims[1:])
    return {"benchmark": bench, "seed": seed, "n_tasks": len(tasks),
            "n_train": int(len(ytr)), "n_test": int(len(yte)), "in_dim": in_dim,
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.items() if k in
                       ("hidden", "feat", "cap", "epochs", "p_target", "prox", "q_cov_in")},
            "grown": grown, "consolidated": consolidated,
            "baseline_mlp": {"dims": dims, "neurons": in_dim + base_u, "connections": base_w},
            "elapsed_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="mnist")
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    out = os.path.join(RES, args.out_name or f"size_{args.benchmark}.json")
    recs = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        r = run(args.benchmark, seed)
        recs.append(r)
        g, c = r["grown"], r["consolidated"]
        print(f"[{args.benchmark} seed {seed}] ({r['elapsed_s']:.0f}s) "
              f"grown strict {g['scores']['strict']:.4f} forced {g['scores']['forced']:.4f} "
              f"cov {g['scores']['coverage']:.3f} | organs {g['size']['n_organs']} "
              f"neurons {g['size']['neurons_total']:,} connections {g['size']['connections_total']:,} "
              f"| growth flips {g['growth_flips']['total']} over {g['growth_flips']['steps_checked']} additions "
              f"| consolidation {'accepted' if c['accepted'] else 'REFUSED'} -> strict "
              f"{c['scores']['strict']:.4f} neurons {c['size']['neurons_total']:,}", flush=True)
        with open(out, "w") as f:
            json.dump(recs, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
