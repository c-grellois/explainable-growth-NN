"""B0 -- toy benchmark: growth by abstention on multi-modal 2-D classes.

The organism starts with one deliberately small organ (development means small cheap
parts), which resolves the dominant structure and abstains elsewhere; growth adds
organs on the abstained mass. Three claims are checked, two of them via controls
that MUST fail:

  claim   coverage grows monotonically; answers never flip (retention theorem, bitwise)
  ctrl A  growth INSIDE the parent (appending anchors to organ 0, the contracts
          note's growth-by-need) flips answers -- the 0.88->0.39 crash, reproduced
  ctrl B  fine-tuning the parent trunk on the need set (touching weights, ASDbrain's
          mechanism) damages previously covered answers

Run:  python exp_toy.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgrow.data import toy_modes, prepare
from cgrow.organ import Organ, DTYPE
from cgrow.organism import Organism, ABSTAIN

SEEDS = [0, 1, 2, 3, 4]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)

HID, FEAT, CAP = (16,), 8, 1.4
P_TARGET, MAX_ORGANS, EPOCHS = 0.94, 6, 120
FOUNDER_FRAC = 0.3        # early development sees a fraction of the data


def control_A_anchor_injection(org: Organism, Xtr, ytr, Xte):
    """Growth inside the parent: append anchors (means of declined samples per
    class) to organ 0 -- unverified, exactly the parent note's closed mechanism."""
    before = org.predict(Xte)["pred"].copy()
    o0 = org.organs[0]
    need = org.declined_mask(Xtr)
    Xn, yn = Xtr[need], ytr[need]
    with torch.no_grad():
        z = o0.trunk(torch.as_tensor(Xn, dtype=DTYPE))
        new_anchors, new_ids = [], []
        for c in np.unique(yn):
            sel = torch.as_tensor(yn == c)
            if sel.sum() >= 15:
                new_anchors.append(z[sel].mean(0))
                new_ids.append(c)
        if new_anchors:
            o0.anchors = torch.nn.Parameter(
                torch.cat([o0.anchors.data, torch.stack(new_anchors)]))
            o0.class_ids = np.concatenate([o0.class_ids, np.array(new_ids)])
    after = org.predict(Xte)["pred"]
    was_answered = before != ABSTAIN
    flips = int((was_answered & (after != before) & (after != ABSTAIN)).sum())
    drops = int((was_answered & (after == ABSTAIN)).sum())   # unchosen coverage loss
    return flips + drops, flips, drops


def control_B_finetune(org: Organism, Xtr, ytr, Xte, yte):
    """Touch the parent's weights: fine-tune organ 0's trunk on the need set."""
    before = org.predict(Xte)["pred"].copy()
    was_answered = before != ABSTAIN
    acc_before = float((before[was_answered] == yte[was_answered]).mean())
    o0 = org.organs[0]
    need = org.declined_mask(Xtr)
    o0.frozen = False
    o0.train_proto(Xtr[need], ytr[need], epochs=30)
    o0.frozen = True
    after = org.predict(Xte)["pred"]
    flips = int((was_answered & (after != before)).sum())
    acc_after = float((after[was_answered] == yte[was_answered]).mean())
    return flips, acc_before, acc_after


def main():
    records = []
    for seed in SEEDS:
        X, y, modes, C = toy_modes(seed=seed)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare(X, y, seed=seed)
        torch.manual_seed(seed)

        # -- the organism, grown honestly ---------------------------------- #
        org = Organism()
        stage_scores = []
        for k in range(MAX_ORGANS):
            if org.organs:
                need = org.declined_mask(Xtr)
            else:   # the founder sees only an early-development subsample
                rng = np.random.default_rng(seed)
                need = np.zeros(len(Xtr), bool)
                need[rng.choice(len(Xtr), int(FOUNDER_FRAC * len(Xtr)), replace=False)] = True
            Xn, yn = Xtr[need], ytr[need]
            classes = [c for c in np.unique(yn) if (yn == c).sum() >= 15]
            if len(classes) == 0 or need.sum() < 30:
                break
            grew = False
            for attempt in range(3):     # candidate diversity per round
                hid = [(16,), (32,), (32, 16)][attempt]
                cand = Organ(2, FEAT, hid, cap=CAP, class_ids=classes,
                             seed=seed * 100 + 10 * k + attempt)
                cand.train_proto(Xn, yn, epochs=EPOCHS)
                # the founder is ungated: there is no organism to protect yet
                if org.try_grow(cand, Xva, yva, p_target=P_TARGET,
                                gate=(k > 0), label=f"o{k}"):
                    grew = True
                    break
            if not grew:
                break
            org.retention_snapshot(Xte, tag=f"o{k}")
            stage_scores.append(org.scores(Xte, yte))
            if k == 0:
                import copy
                org_stage0 = copy.deepcopy(org)   # founder-only, for the controls
        ret = org.retention_check(Xte)

        # -- controls, applied at the founder stage: what if, instead of ----- #
        # -- growing organ 1, we had grown INSIDE / fine-tuned the founder? -- #
        import copy
        if not org.organs:
            print(f"seed {seed}: no organism grew -- skipping controls")
            continue
        orgA = copy.deepcopy(org_stage0)
        violA, flipsA_only, dropsA = control_A_anchor_injection(orgA, Xtr, ytr, Xte)
        flipsA = violA   # violations = flips + unchosen drops
        orgB = copy.deepcopy(org_stage0)
        flipsB, accB_before, accB_after = control_B_finetune(orgB, Xtr, ytr, Xte, yte)

        rec = {"seed": seed,
               "n_organs": len(org.organs),
               "stages": stage_scores,
               "final": org.scores(Xte, yte),
               "retention": ret,
               "ctrlA_flips": flipsA,
               "ctrlB_flips": flipsB,
               "ctrlB_covered_acc_before": accB_before,
               "ctrlB_covered_acc_after": accB_after,
               "log": org.log}
        records.append(rec)
        cov = [s["coverage"] for s in stage_scores]
        print(f"seed {seed}: organs={len(org.organs)}  coverage {['%.3f' % c for c in cov]}"
              f"  strict={rec['final']['strict']:.3f}  prec={rec['final']['precision']:.3f}"
              f"  flips={ret['flips']}  | ctrlA flips={flipsA}  ctrlB flips={flipsB} "
              f"(covered acc {accB_before:.3f}->{accB_after:.3f})")

    # ---- verdicts --------------------------------------------------------- #
    flips = sum(r["retention"]["flips"] for r in records)
    mono = all(all(a["coverage"] <= b["coverage"] + 1e-9 for a, b in
                   zip(r["stages"], r["stages"][1:])) for r in records)
    ctrlA_fails = sum(r["ctrlA_flips"] > 0 for r in records)
    ctrlB_damage = sum(r["ctrlB_covered_acc_after"] < r["ctrlB_covered_acc_before"] - 0.01
                       for r in records)
    grew = sum(r["n_organs"] > 1 for r in records)
    print(f"\nVERDICTS over {len(SEEDS)} seeds:")
    print(f"  retention flips (must be 0):            {flips}   {'PASS' if flips == 0 else 'FAIL'}")
    print(f"  coverage monotone (must hold):          {mono}   {'PASS' if mono else 'FAIL'}")
    print(f"  growth accepted somewhere (must hold):  {grew}/{len(SEEDS)}   {'PASS' if grew else 'FAIL'}")
    print(f"  ctrl A anchor-injection violates retention (flips+unchosen drops): {ctrlA_fails}/{len(SEEDS)} seeds   "
          f"{'PASS (control fails as it must)' if ctrlA_fails else 'CONTROL DID NOT FAIL'}")
    print(f"  ctrl B fine-tune damages covered acc:   {ctrlB_damage}/{len(SEEDS)} seeds   "
          f"{'PASS (control fails as it must)' if ctrlB_damage else 'CONTROL DID NOT FAIL'}")

    with open(os.path.join(RES, "toy.json"), "w") as f:
        json.dump(records, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(RES, 'toy.json')}")


def plot_anatomy(seed: int = 4):
    """2-D map of which organ answers where -- the organism's anatomy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y, modes, C = toy_modes(seed=seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare(X, y, seed=seed)
    torch.manual_seed(seed)
    org = Organism()
    snapshots = []
    for k in range(MAX_ORGANS):
        if org.organs:
            need = org.declined_mask(Xtr)
        else:
            rng = np.random.default_rng(seed)
            need = np.zeros(len(Xtr), bool)
            need[rng.choice(len(Xtr), int(FOUNDER_FRAC * len(Xtr)), replace=False)] = True
        Xn, yn = Xtr[need], ytr[need]
        classes = [c for c in np.unique(yn) if (yn == c).sum() >= 15]
        if len(classes) == 0 or need.sum() < 30:
            break
        grew = False
        for attempt in range(3):
            hid = [(16,), (32,), (32, 16)][attempt]
            cand = Organ(2, FEAT, hid, cap=CAP, class_ids=classes,
                         seed=seed * 100 + 10 * k + attempt)
            cand.train_proto(Xn, yn, epochs=EPOCHS)
            if org.try_grow(cand, Xva, yva, p_target=P_TARGET, gate=(k > 0)):
                grew = True
                break
        if not grew:
            break
        snapshots.append((len(org.organs), org.scores(Xte, yte)))

    gx, gy = np.meshgrid(np.linspace(0, 1, 220), np.linspace(0, 1, 220))
    G = np.stack([gx.ravel(), gy.ravel()], axis=1)
    r = org.predict(G)
    organ_map = r["organ"].reshape(gx.shape)
    n_org = len(org.organs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cmap = plt.get_cmap("tab10")
    img = np.ones((*gx.shape, 3))
    for i in range(n_org):
        img[organ_map == i] = np.array(cmap(i)[:3]) * 0.45 + 0.55
    axes[0].imshow(img, origin="lower", extent=[0, 1, 0, 1])
    for c, mk in zip(range(3), ["o", "s", "^"]):
        sel = yte == c
        axes[0].scatter(Xte[sel, 0], Xte[sel, 1], c="k", marker=mk, s=6, alpha=0.5)
    axes[0].set_title(f"anatomy: which organ answers where ({n_org} organs;\n"
                      "white = certified abstention)")
    covs = [s["coverage"] for _, s in snapshots]
    precs = [s["precision"] for _, s in snapshots]
    axes[1].plot(range(1, len(covs) + 1), covs, "o-", label="coverage")
    axes[1].plot(range(1, len(precs) + 1), precs, "s-", label="precision on covered")
    axes[1].set_xlabel("organs"); axes[1].set_ylim(0, 1.02); axes[1].legend()
    axes[1].set_title("development: monotone coverage,\nprecision held, zero flips")
    fig.tight_layout()
    out = os.path.join(RES, "toy_anatomy.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
    plot_anatomy(seed=4)
