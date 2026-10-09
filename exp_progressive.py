"""B-progressive -- a curriculum of CONSOLIDATION: when should the organism
consolidate what it has grown so far?

Flat CIFAR-100 class-IL on frozen ViT-B features (10 tasks, random class order
of the protocol). Growth is the usual by-need cascade with vetoes (grow_task
from exp_midlife). A `--schedule` names the tasks after which the verified
prototype consolidation (consolidate_protos, relative gate) is PROPOSED:

  end        after task 10 only                 (= adolescence as in the lifecycle)
  mid        after task 5 only                  (= B-midlife)
  every2     after tasks 2, 4, 6, 8, 10          (progressive: each consolidation
                                                  folds the previous C and the
                                                  organs grown since into a new C)
  every2end  after tasks 2..8, then again at 10  (same as every2; kept explicit)

Each proposal passes the same gate as adolescence; a refusal leaves the anatomy
as it is and growth continues. Measured per seed: final strict/forced/coverage,
per-stage acceptances and strict before/after, growth flips outside
consolidation events (must be 0), vetoes over life, share answered by the last
C, parameters, and the organism's depth (organs in the cascade) at the end.

Run:  python exp_progressive.py --schedule every2 [--first-seed 2 --seeds 3]
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
from cgrow.organism import Organism, ABSTAIN
from cgrow.adolescence import consolidate_protos
from exp_cifar import load_feats, CFG
from exp_midlife import grow_task

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCHEDULES = {"end": [10], "mid": [5], "every2": [2, 4, 6, 8, 10], "every2end": [2, 4, 6, 8, 10]}


def vetoes_of(org):
    """Structural count: the organism returned by consolidate_protos does not
    carry the growth log, so count the vetoes the organs actually hold."""
    return int(sum(len(getattr(o, "vetoes", [])) for o in org.organs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="every2", choices=sorted(SCHEDULES))
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="vitb")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    out_name = args.out_name or f"progressive_{args.schedule}_{args.tag}.json"
    feats = os.path.join(RES, f"cifar100_{args.tag}.npz")
    when = SCHEDULES[args.schedule]
    records = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.time()
        splits = load_feats(seed, feats_path=feats)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
        order = np.random.default_rng(seed + 400).permutation(100)
        tasks = [sorted(int(c) for c in order[i * 10:(i + 1) * 10]) for i in range(10)]
        torch.manual_seed(seed)
        org = Organism()
        flips = 0
        events = []
        curve = []
        for t in range(10):
            f, fa = grow_task(org, tasks[t], splits, CFG, seed, t)
            flips += f
            seen = np.isin(yte, [c for cl in tasks[:t + 1] for c in cl])
            sc = org.scores(Xte[seen], yte[seen])
            stage = {"task": t + 1, "strict_seen": sc["strict"], "coverage_seen": sc["coverage"],
                     "organs": len(org.organs), "fa": fa}
            if (t + 1) in when:
                before = org.scores(Xte[seen], yte[seen])["strict"]
                last = np.isin(yva, tasks[t])
                org2, rep = consolidate_protos(org, Xtr.shape[1], CFG, seed,
                                               Xva_last=Xva[last], yva_last=yva[last])
                acc = bool(rep["accepted"])
                if acc:
                    org = org2
                after = org.scores(Xte[seen], yte[seen])["strict"]
                stage["consolidation"] = {"accepted": acc, "why": rep.get("why"),
                                          "strict_before": before, "strict_after": after,
                                          "params_C": rep.get("params_C"),
                                          "organs_after": len(org.organs)}
                events.append(stage["consolidation"])
                print(f"[prog {args.schedule} seed {seed}] after task {t + 1}: consolidation "
                      f"{'ACCEPTED' if acc else 'refused (' + str(rep.get('why')) + ')'} "
                      f"{before:.3f} -> {after:.3f} | organs {len(org.organs)}", flush=True)
            curve.append(stage)
        final = org.scores(Xte, yte)
        r = org.predict(Xte)
        # share answered by the most recent consolidated organ (index 0 if any accepted)
        share_C = float((r["organ"] == 0).mean()) if any(e["accepted"] for e in events) else 0.0
        rec = {"seed": seed, "schedule": args.schedule, "when": when, "final": final,
               "growth_flips": flips, "vetoes_total": vetoes_of(org),
               "consolidations": events, "n_accepted": sum(e["accepted"] for e in events),
               "share_C": share_C, "n_organs": len(org.organs), "n_params": org.n_params(),
               "curve": curve, "elapsed_s": time.time() - t0}
        records.append(rec)
        print(f"[prog {args.schedule} seed {seed}] ({time.time()-t0:.0f}s) FINAL strict {final['strict']:.3f} "
              f"forced {final['forced']:.3f} cov {final['coverage']:.3f} | accepted "
              f"{rec['n_accepted']}/{len(events)} | share_C {share_C:.2f} | organs {len(org.organs)} "
              f"| vetoes {rec['vetoes_total']} | growth flips {flips}", flush=True)
        with open(os.path.join(RES, out_name), "w") as f:
            json.dump(records, f, indent=1, default=float)
    print("wrote", os.path.join(RES, out_name))


if __name__ == "__main__":
    main()
