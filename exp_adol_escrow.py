"""B-adol-escrow -- A/B the adolescence gate, old rule versus the added escrow.

The adolescence gate judged a consolidation on the synthetic contract holdout
plus the LAST task's real validation split. From 2026-09-09 it also faces the
full real-data escrow (McNemar over the whole validation split, Bonferroni
per-class flags, 0.25 hard floor), the same test the arbiter gate uses. This
script replays every cached grown organism through BOTH gates and reports the
two verdicts side by side, so the change is measured rather than assumed:

  old      : cfg["adol_escrow"] = False  (synthetic holdout + last task's val)
  verbatim : the arbiter's escrow transplanted as first written, i.e. with the
             0.25 per-class hard floor and NO condition on how many samples the
             breach amounts to
  new      : the same escrow with the floor's lost-sample condition (>= 5), the
             gate as it now stands

For each (cache, seed) it prints the two verdicts, the escrow's McNemar z, the
consolidated organism's real-validation accuracy against the grown one, and the
TEST strict accuracy each verdict would leave the organism at -- so a refusal
can be scored as a save (test would have dropped) or a cost (test would have
risen). Nothing is retrained: the organisms come from
results/adol_org_<tag>_<seed>.pkl.

Run:  python exp_adol_escrow.py [--tags vitb cub_vitb ...] [--out-name adol_escrow_ab.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.adolescence import consolidate_protos
from exp_cifar import load_feats, CFG

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def feats_for(tag):
    """Map an organism tag to the feature cache it was grown on."""
    for cand in (f"cifar100_{tag}.npz", f"ptm_{tag}.npz"):
        p = os.path.join(RES, cand)
        if os.path.exists(p):
            return p
    return None


def discover():
    """(tag, seed, feats) for every cached organism whose cache still exists."""
    out = []
    for p in sorted(glob.glob(os.path.join(RES, "adol_org_*.pkl"))):
        m = re.match(r"adol_org_(.+)_(\d+)\.pkl$", os.path.basename(p))
        if not m:
            continue
        tag, seed = m.group(1), int(m.group(2))
        f = feats_for(tag)
        if f is None:
            continue
        # a p1 cache is seed-specific: adol_org_<name>_p1s<S>_<S>.pkl
        ms = re.search(r"_p1s(\d+)$", tag)
        if ms and int(ms.group(1)) != seed:
            continue
        out.append((tag, seed, f))
    return out


def main():
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None,
                    help="restrict to these organism tags (default: all cached)")
    ap.add_argument("--out-name", default="adol_escrow_ab.json")
    args = ap.parse_args()

    todo = discover()
    if args.tags:
        todo = [t for t in todo if t[0] in args.tags]
    print(f"{len(todo)} cached organisms to replay\n", flush=True)
    records = []
    for tag, seed, feats in todo:
        with open(os.path.join(RES, f"adol_org_{tag}_{seed}.pkl"), "rb") as fh:
            org = pickle.load(fh)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = load_feats(seed, feats_path=feats)
        n_cls = int(np.load(feats, allow_pickle=False)["n_classes"]) \
            if "n_classes" in np.load(feats, allow_pickle=False).files else int(ytr.max()) + 1
        inc = int(np.load(feats, allow_pickle=False)["inc"]) \
            if "inc" in np.load(feats, allow_pickle=False).files else 10
        order = np.random.default_rng(seed + 400).permutation(n_cls)
        tasks = [sorted(int(c) for c in order[i:i + inc]) for i in range(0, n_cls, inc)]
        last = np.isin(yva, tasks[-1])
        grown = org.scores(Xte, yte)["strict"]

        rec = {"tag": tag, "seed": seed, "grown_test_strict": grown}
        arms = (("old", {"adol_escrow": False}),
                ("verbatim", {"adol_hard_floor_min_lost": 1}),
                ("new", {}))
        for label, kw in arms:
            cfg = {**CFG, **kw}
            org2, rep = consolidate_protos(org, Xtr.shape[1], cfg, seed,
                                           Xva_last=Xva[last], yva_last=yva[last],
                                           Xva=Xva, yva=yva)
            rec[label] = {
                "accepted": bool(rep["accepted"]),
                "why": rep.get("why"), "refused_by": rep.get("refused_by"),
                "test_strict_after": org2.scores(Xte, yte)["strict"],
                "escrow": rep.get("escrow"),
            }
        # what the change costs or saves, in test points
        d = rec["new"]["test_strict_after"] - rec["old"]["test_strict_after"]
        rec["delta_new_minus_old"] = d
        rec["verdict_changed"] = rec["new"]["accepted"] != rec["old"]["accepted"]
        rec["verbatim_changed"] = rec["verbatim"]["accepted"] != rec["old"]["accepted"]
        rec["delta_verbatim_minus_old"] = (rec["verbatim"]["test_strict_after"]
                                           - rec["old"]["test_strict_after"])
        e = rec["new"].get("escrow") or {}
        print(f"[{tag} s{seed}] grown {grown:.3f} | old {'ACC' if rec['old']['accepted'] else 'REJ'} "
              f"-> {rec['old']['test_strict_after']:.3f} | new {'ACC' if rec['new']['accepted'] else 'REJ'} "
              f"-> {rec['new']['test_strict_after']:.3f} | delta {d:+.4f}"
              f" | escrow z {e.get('mcnemar_z', float('nan')):+.2f} "
              f"real val {e.get('overall_ref', float('nan')):.3f}->{e.get('overall_new', float('nan')):.3f}"
              f" flagged {len(e.get('flagged_classes', []))} hard {len(e.get('hard_floor_classes', []))}"
              + (f"  | verbatim {'ACC' if rec['verbatim']['accepted'] else 'REJ'}"
                 f" -> {rec['verbatim']['test_strict_after']:.3f}" if rec["verbatim_changed"] else "")
              + ("  ** NEW-GATE VERDICT CHANGED **" if rec["verdict_changed"] else ""), flush=True)
        records.append(rec)
        with open(os.path.join(RES, args.out_name), "w") as f:
            json.dump(records, f, indent=1, default=float)

    ch = [r for r in records if r["verdict_changed"]]
    vb = [r for r in records if r.get("verbatim_changed")]
    print(f"\nverdicts changed by the FINAL gate: {len(ch)} / {len(records)}")
    print(f"verdicts changed by the VERBATIM transplant (0.25 floor, no count condition):"
          f" {len(vb)} / {len(records)}"
          + (f" -> {[(r['tag'], r['seed'], round(r['delta_verbatim_minus_old'], 4)) for r in vb]}" if vb else ""))
    saves = [r for r in ch if r["delta_new_minus_old"] > 0]
    costs = [r for r in ch if r["delta_new_minus_old"] < 0]
    print(f"  refusals that raised final test accuracy (saves): {len(saves)}"
          + (f" -> {[(r['tag'], r['seed'], round(r['delta_new_minus_old'], 4)) for r in saves]}" if saves else ""))
    print(f"  refusals that lowered it (costs): {len(costs)}"
          + (f" -> {[(r['tag'], r['seed'], round(r['delta_new_minus_old'], 4)) for r in costs]}" if costs else ""))
    print("wrote", os.path.join(RES, args.out_name))


if __name__ == "__main__":
    main()
