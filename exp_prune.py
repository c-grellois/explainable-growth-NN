"""B-prune -- magnitude pruning at birth: how small can an organ's trunk be?

*Preregistered 2026-10-02, before any run.* The small-network literature on
MNIST (Han et al. 2015, `Learning both Weights and Connections`, Tables 1-3)
reaches 98.4% with a LeNet-300-100 pruned 12x to ~22K parameters, while the
grown organism spends 1.12M connections for 89.3% on the class-incremental
protocol. That gap is partly protocol and partly that nothing here has ever
made a part smaller. This asks how much of it is the second.

**Where pruning is allowed.** At birth, before the organ is frozen: train the
trunk, drop the smallest-|w| fraction of each layer, retrain the survivors with
the dropped weights pinned at zero, and only then solve the thresholds and
freeze. The contract is therefore solved against the network that actually
ships, and every guarantee is untouched. Pruning a *frozen* organ would
invalidate its contract -- its anchors, radii and territory describe a function
that no longer exists -- and is not offered by the code.

**Predictions, recorded in advance.**
1. Growth flips stay exactly 0 at every sparsity. This is structural (pruning
   happens before freezing, and the cascade's first-accept routing is
   unchanged), so a non-zero count is a bug, not a finding.
2. Sparsity 0 reproduces the unpruned run bit for bit; anything else means the
   pruning path leaks into the control.
3. Strict accuracy holds within 1 point up to some sparsity and then falls. The
   question the experiment answers is where, measured, not whether.

Reported per sparsity: strict / forced / coverage, nonzero connections, the
compression against the unpruned organism, and the growth-flip count.

Run:  python exp_prune.py --benchmark mnist --sparsity 0 0.5 0.8 0.9 0.95 --first-seed 2 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import load_real, prepare, split_tasks
from exp_continual import run_ours, CFG as MLP_CFG
from exp_size import organism_size

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run(bench: str, seed: int, sparsity: float, prune_epochs=None) -> dict:
    """Growth exactly as exp_continual runs it, with pruning switched on inside
    each organ's birth. Same data preparation, same task split, same seeding."""
    t0 = time.time()
    cfg = dict(MLP_CFG[bench])
    cfg["prune"] = float(sparsity)
    if prune_epochs is not None:
        cfg["prune_epochs"] = int(prune_epochs)
    X, y, C = load_real(bench)
    splits = prepare(X, y, seed=seed, max_train=cfg["max_train"])
    tasks = split_tasks(y, 2)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    in_dim = int(Xtr.shape[1])
    torch.manual_seed(seed)
    org, stats = run_ours(tasks, splits, in_dim, cfg, seed)
    size = organism_size(org, in_dim)
    nz = sum(p["trunk"]["nonzero_weights"] for p in size["per_organ"])
    nz_v = sum(p["vetoes"]["weights"] for p in size["per_organ"])   # vetoes are not pruned
    return {"benchmark": bench, "seed": seed, "sparsity": float(sparsity),
            "scores": org.scores(Xte, yte),
            "growth_flips": int(stats["growth_flips"]),
            "tighten_flips": stats["tighten_flips"],
            "n_organs": len(org.organs),
            "trunk_connections_dense": size["trunk_connections"],
            "trunk_connections_nonzero": int(nz),
            "veto_connections": int(nz_v),
            "connections_total_nonzero": int(nz + nz_v),
            "neurons_total": size["neurons_total"],
            "contract_floats": size["contract_floats"],
            "elapsed_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="mnist")
    ap.add_argument("--sparsity", type=float, nargs="+",
                    default=[0.0, 0.5, 0.8, 0.9, 0.95])
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--prune-epochs", type=int, default=None,
                    help="retrain epochs after the drop (default: the config's own)")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    out = os.path.join(RES, args.out_name or f"prune_{args.benchmark}.json")
    recs = []
    for sp in args.sparsity:
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            r = run(args.benchmark, seed, sp, args.prune_epochs)
            recs.append(r)
            print(f"[{args.benchmark} sparsity {sp:g} seed {seed}] ({r['elapsed_s']:.0f}s) "
                  f"strict {r['scores']['strict']:.4f} forced {r['scores']['forced']:.4f} "
                  f"cov {r['scores']['coverage']:.3f} | trunk connections "
                  f"{r['trunk_connections_nonzero']:,} of {r['trunk_connections_dense']:,} "
                  f"| growth flips {r['growth_flips']}", flush=True)
            with open(out, "w") as f:
                json.dump(recs, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
