"""Aggregate battery 2 (in1k caches) and battery 3 (Phase-1 caches) results into
the PHASE1_RESULTS.md tables (HANDOFF_GPU2.md section 6). Read-only: it only
reads results\\*.json / *.npz and prints markdown. Run from the project root
with the gpu venv:  gpu\\venv\\Scripts\\python.exe gpu\\summarize_phase1.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console defaults to cp1252

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEEDS = [2, 3, 4]
SUITE = ["imr", "ima", "cub", "obj", "omni", "vtab"]
NAMES = {"cifar100": "CIFAR-100", "imr": "ImageNet-R", "ima": "ImageNet-A", "cub": "CUB-200",
         "obj": "ObjectNet-200", "omni": "OmniBenchmark", "vtab": "VTAB"}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def per_seed(cache, tag_json, adol_json, life_json, life_key, seed):
    """One row's raw numbers for (cache, seed)."""
    out = {"seed": seed}
    base = load(os.path.join(RES, tag_json))
    if base:
        r = next((r for r in base if r["seed"] == seed), None)
        if r:
            out["growth_strict"] = r["ours"]["final"]["strict"]
            out["growth_forced"] = r["ours"]["final"]["forced"]
            out["growth_flips"] = r["ours"]["growth_flips"]
            out["ncm"] = r["ncm"]
            out["replay"] = r["baselines"]["replay"]["final_acc"]
            out["joint"] = r["joint"]
            if "contracted_ncm" in r:
                out["cncm_strict"] = r["contracted_ncm"]["strict"]
                out["cncm_forced"] = r["contracted_ncm"]["forced"]
                out["cncm_cov"] = r["contracted_ncm"]["coverage"]
    head = load(os.path.join(RES, f"head_{cache}.json"))
    if head:
        h = next((h for h in head if h["seed"] == seed and h["M"] == 10000), None)
        if h:
            out["head_strict"] = h["strict"]
            out["head_forced"] = h["forced"]
            out["head_cov"] = h["coverage"]
            out["head_flips100"] = 100.0 * h["answer_flips_under_growth"] / h["n_test"]
    adol = load(os.path.join(RES, adol_json))
    if adol:
        a = next((a for a in adol if a["seed"] == seed), None)
        if a:
            out["adol_accepted"] = bool(a["adolescence"]["accepted"])
            out["adol_after"] = a["after"]["strict"] if a["adolescence"]["accepted"] else a["before"]["strict"]
            out["adol_share_C"] = (a.get("audit") or {}).get("share_answered_by_C")
    life = load(os.path.join(RES, life_json))
    if life and life_key in life:
        l = next((l for l in life[life_key] if l["seed"] == seed), None)
        if l:
            out["pfc"] = l["pfc"]
            out["pfc_free"] = l["pfc_free_test"]
            out["pfc_accepted"] = bool(l["pfc_accepted"])
            out["uncert"] = l["uncertified_fraction"]
            out["escrow_z"] = l["escrow_z"]
            out["adol_life"] = l["adolescence"]
    npz = os.path.join(RES, f"{cache}.npz")
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=False)
        if "phase1_val_acc" in d.files:
            out["p1_val"] = float(d["phase1_val_acc"])
            out["p1_params"] = int(d["phase1_tuned_params"])
    return out


def mean_row(rows, key):
    v = [r[key] for r in rows if key in r and r[key] is not None and not isinstance(r[key], bool)]
    return float(np.mean(v)) if v else None


def fmt_cells(label, r, with_p1):
    cells = [label, str(r.get("seed", "mean"))]
    if with_p1:
        cells += [f3(r.get("p1_val")), (f"{r['p1_params']:,}" if r.get("p1_params") else "—")]
    cells.append(f"{f3(r.get('cncm_strict'))} / {f3(r.get('cncm_forced'))}")
    hf = r.get("head_flips100")
    cells.append(f"{f3(r.get('head_strict'))} / {f3(r.get('head_forced'))} / {f3(r.get('head_cov'))} / "
                 f"{'—' if hf is None else f'{hf:.0f}'}")
    gf = r.get("growth_flips")
    cells.append(f"{f3(r.get('growth_strict'))}" + ("" if gf is None else f" (flips {gf})"))
    aa = r.get("adol_accepted")
    cells.append(f"{f3(r.get('adol_after'))}" + ("" if aa is None else (" ✔" if aa else " REJ")))
    pa = r.get("pfc_accepted")
    unc = r.get("uncert")
    if pa is None:
        cells.append(f3(r.get("pfc")))
    else:
        cells.append(f"{f3(r.get('pfc'))} (arb {'✔' if pa else 'REJ'}"
                     + ("" if unc is None else f", uncert {unc:.3f}") + ")")
    cells.append(f"[{f3(r.get('pfc_free'))}]")
    cells.append(f3(r.get("ncm")))
    return cells


def fmt_row(label, r, with_p1):
    return "| " + " | ".join(fmt_cells(label, r, with_p1)) + " |"


def table(title, groups, with_p1):
    hdr = ["dataset", "seed"]
    if with_p1:
        hdr += ["Phase-1 task-0 val acc", "tuned params"]
    hdr += ["contracted protos strict / forced", "contracted head (M=10k) strict / forced / coverage / changes per 100",
            "growth strict", "adolescence strict", "certified lifecycle", "free arbiter", "NCM same feats"]
    print(f"\n### {title}\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for label, rows in groups:
        rows = [r for r in rows if len(r) > 1]
        if not rows:
            print(f"| {label} | — | " + " | ".join(["did not run"] * (len(hdr) - 2)) + " |")
            continue
        for r in rows:
            print(fmt_row(label, r, with_p1))
        if len(rows) > 1:
            m = {k: mean_row(rows, k) for k in ["p1_val", "cncm_strict", "cncm_forced", "head_strict", "head_forced",
                                                  "head_cov", "head_flips100", "growth_strict", "adol_after", "pfc",
                                                  "pfc_free", "ncm", "uncert"]}
            m["seed"] = "mean"
            m["growth_flips"] = sum(r.get("growth_flips", 0) for r in rows)
            m["p1_params"] = rows[0].get("p1_params")
            n_adol = sum(1 for r in rows if r.get("adol_accepted"))
            n_adol_run = sum(1 for r in rows if r.get("adol_accepted") is not None)
            n_pfc = sum(1 for r in rows if r.get("pfc_accepted"))
            n_pfc_run = sum(1 for r in rows if r.get("pfc_accepted") is not None)
            cells = fmt_cells(f"**{label}**", m, with_p1)
            i_adol = 7 if with_p1 else 5
            cells[i_adol] += f" ({n_adol}/{n_adol_run} acc.)"
            uncs = [r["uncert"] for r in rows if r.get("uncert") is not None]
            cells[i_adol + 1] += (f" (arb {n_pfc}/{n_pfc_run}, uncert max {max(uncs):.3f})" if uncs else "")
            print("| " + " | ".join(cells) + " |")


def main():
    # ---- battery 2: in1k caches, seeds 2-4 in one chain each
    groups = []
    for name in ["imr", "ima", "cub"]:
        cache = f"ptm_{name}_vitb1k"
        rows = [per_seed(cache, f"{cache}.json", f"{cache}_adol_protos.json",
                         f"lifecycle_{cache}.json", cache, s) for s in SEEDS]
        groups.append((NAMES[name], rows))
    table("Battery 2 — ViT-B/16 ImageNet-1K-fine-tuned checkpoint (`augreg_in21k_ft_in1k`), frozen, seeds 2–4", groups, False)
    summ = os.path.join(ROOT, "gpu", "out", "cache_summary_vitb1k.txt")
    if os.path.exists(summ):
        print("\nJoint NCM of the in1k caches (cache_summary.py) vs RanPAC 'NCM only' (v3 Tab. 1: 0.612 / 0.493 / 0.851):\n")
        txt = open(summ, encoding="utf-8", errors="replace").read()
        for name, ranpac in [("imr", 0.612), ("ima", 0.493), ("cub", 0.851)]:
            m = re.search(rf"ptm_{name}_vitb1k\.npz.*?joint NCM \(feature quality only\) ([0-9.]+)", txt, re.S)
            if m:
                print(f"- {NAMES[name]}: joint NCM {float(m.group(1)):.3f} (RanPAC NCM-only {ranpac:.3f})")

    # ---- battery 3: Phase-1 caches, one seed each
    groups = []
    rows = []
    for s in SEEDS:
        cache = f"cifar100_vitb_p1s{s}"
        rows.append(per_seed(cache, f"{cache}.json", f"{cache}_adol_protos.json",
                             f"lifecycle_cifar_vitb_p1s{s}.json", f"cifar_vitb_p1s{s}", s))
    groups.append((NAMES["cifar100"], rows))
    for name in SUITE:
        rows = []
        for s in SEEDS:
            cache = f"ptm_{name}_vitb_p1s{s}"
            rows.append(per_seed(cache, f"{cache}.json", f"{cache}_adol_protos.json",
                                 f"lifecycle_{cache}.json", cache, s))
        groups.append((NAMES[name], rows))
    table("Battery 3 — ViT-B/16-in21k + first-session adapters (Phase 1, r = 16, 20 epochs), then frozen; one seed per cache", groups, True)


if __name__ == "__main__":
    main()
