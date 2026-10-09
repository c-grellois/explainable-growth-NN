"""Aggregate battery 4 (HANDOFF_GPU3.md) into the BATTERY4_RESULTS.md tables.
Read-only over results\\*.json / *.npz. Run from the project root:
    gpu\\venv\\Scripts\\python.exe gpu\\summarize_battery4.py
Block A: Phase-1 caches seeds 5-7 (+ seeds 2-4 for the two-sigma comparison);
Block C/D: text and tabular caches, seeds 2-4 in one chain each.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_phase1 import per_seed, f3, NAMES, RES, load  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SUITE = ["imr", "ima", "cub", "obj", "omni", "vtab"]
CONF = [5, 6, 7]
EVAL = [2, 3, 4]


def p1_rows(name, seeds):
    rows = []
    for s in seeds:
        if name == "cifar100":
            cache = f"cifar100_vitb_p1s{s}"
            r = per_seed(cache, f"{cache}.json", f"{cache}_adol_protos.json",
                         f"lifecycle_cifar_vitb_p1s{s}_s{s}.json", f"cifar_vitb_p1s{s}", s)
            if "pfc" not in r:   # battery-3 naming (no _s<seed> suffix)
                r = per_seed(cache, f"{cache}.json", f"{cache}_adol_protos.json",
                             f"lifecycle_cifar_vitb_p1s{s}.json", f"cifar_vitb_p1s{s}", s)
        else:
            cache = f"ptm_{name}_vitb_p1s{s}"
            r = per_seed(cache, f"{cache}.json", f"{cache}_adol_protos.json",
                         f"lifecycle_{cache}_s{s}.json", cache, s)
            if "pfc" not in r:
                r = per_seed(cache, f"{cache}.json", f"{cache}_adol_protos.json",
                             f"lifecycle_{cache}.json", cache, s)
        rows.append(r)
    return rows


def mean_sd(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and not isinstance(r[key], bool)]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)


def cell(r, k):
    return f3(r.get(k))


def block_a():
    print("## Block A — first-session adapters, confirmation seeds 5–7 (one seed per cache)\n")
    hdr = ["dataset", "seed", "Phase-1 task-0 val acc", "tuned params", "contracted protos strict / forced",
           "head M=10k strict / forced / coverage / changes per 100", "growth strict (flips)",
           "adolescence (accepted?)", "certified lifecycle (arbiter?)", "free arbiter", "uncertified fraction", "NCM"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    gate = {"arb_acc": 0, "arb_n": 0, "adol_acc": 0, "adol_n": 0, "regress": [], "refused_better": []}
    comp = []
    for name in ["cifar100"] + SUITE:
        rows = p1_rows(name, CONF)
        ref = p1_rows(name, EVAL)
        done = [r for r in rows if "pfc" in r]
        for r in rows:
            if len(r) == 1:
                print(f"| {NAMES[name]} | {r['seed']} | " + " | ".join(["did not run"] * (len(hdr) - 2)) + " |")
                continue
            hf = r.get("head_flips100")
            aa = r.get("adol_accepted"); pa = r.get("pfc_accepted")
            print("| " + " | ".join([
                NAMES[name], str(r["seed"]), cell(r, "p1_val"),
                f"{r['p1_params']:,}" if r.get("p1_params") else "—",
                f"{cell(r,'cncm_strict')} / {cell(r,'cncm_forced')}",
                f"{cell(r,'head_strict')} / {cell(r,'head_forced')} / {cell(r,'head_cov')} / {'—' if hf is None else f'{hf:.0f}'}",
                f"{cell(r,'growth_strict')} ({r.get('growth_flips','—')})",
                f"{cell(r,'adol_after')} ({'—' if aa is None else ('✔' if aa else 'REJ')})",
                f"{cell(r,'pfc')} ({'—' if pa is None else ('arb ✔' if pa else 'arb REJ')})",
                f"[{cell(r,'pfc_free')}]", cell(r, "uncert"), cell(r, "ncm")]) + " |")
            if pa is not None:
                gate["arb_n"] += 1; gate["arb_acc"] += int(pa)
                cand = None
                life = load(os.path.join(RES, f"lifecycle_{'cifar_vitb' if name=='cifar100' else 'ptm_'+name+'_vitb'}_p1s{r['seed']}_s{r['seed']}.json"))
                if life:
                    rec = list(life.values())[0][0]
                    cand = rec.get("pfc_candidate_test")
                    adol = rec.get("adolescence")
                    if pa and cand is not None and cand < adol:
                        gate["regress"].append((NAMES[name], r["seed"], adol, cand, adol - cand))
                    if (not pa) and cand is not None and cand > adol + 0.01:
                        gate["refused_better"].append((NAMES[name], r["seed"], adol, cand, cand - adol,
                                                       rec.get("pfc_accepted_old"), rec.get("territory_passed"),
                                                       rec.get("escrow_passed"), rec.get("escrow_z")))
            if aa is not None:
                gate["adol_n"] += 1; gate["adol_acc"] += int(aa)
        if done:
            # mean row + comparison with seeds 2-4
            keys = ["p1_val", "cncm_strict", "cncm_forced", "head_strict", "head_forced", "head_cov",
                    "head_flips100", "growth_strict", "adol_after", "pfc", "pfc_free", "ncm"]
            m = {k: mean_sd(done, k)[0] for k in keys}
            hf = m.get("head_flips100")
            n_a = sum(1 for r in done if r.get("adol_accepted")); n_p = sum(1 for r in done if r.get("pfc_accepted"))
            print("| " + " | ".join([
                f"**{NAMES[name]}**", f"mean s5–7 (n={len(done)})", f3(m["p1_val"]), "",
                f"{f3(m['cncm_strict'])} / {f3(m['cncm_forced'])}",
                f"{f3(m['head_strict'])} / {f3(m['head_forced'])} / {f3(m['head_cov'])} / {'—' if hf is None else f'{hf:.0f}'}",
                f"{f3(m['growth_strict'])} ({sum(r.get('growth_flips',0) for r in done)})",
                f"{f3(m['adol_after'])} ({n_a}/{len(done)} acc.)", f"{f3(m['pfc'])} (arb {n_p}/{len(done)})",
                f"[{f3(m['pfc_free'])}]", f"{max(r.get('uncert',0) for r in done):.3f} max", f3(m["ncm"])]) + " |")
            refd = [r for r in ref if "pfc" in r]
            if refd:
                cmp_cells = []
                for k, label in [("cncm_strict", "protos strict"), ("cncm_forced", "protos forced"),
                                 ("head_strict", "head strict"), ("head_forced", "head forced"), ("pfc", "certified")]:
                    m24, sd24, _ = mean_sd(refd, k); m57, _, _ = mean_sd(done, k)
                    if m24 is None or m57 is None:
                        continue
                    z = (m57 - m24) / sd24 if sd24 and sd24 > 0 else float("inf") if abs(m57 - m24) > 1e-9 else 0.0
                    inside = abs(m57 - m24) <= 2 * sd24 if sd24 else abs(m57 - m24) < 1e-9
                    cmp_cells.append(f"{label} {m24:.3f}±{sd24:.3f} → {m57:.3f} ({'inside' if inside else 'OUTSIDE'} 2σ)")
                comp.append((NAMES[name], cmp_cells))
    print("\n### Gate record, seeds 5–7\n")
    print(f"- arbiter accepted {gate['arb_acc']}/{gate['arb_n']}; adolescence accepted {gate['adol_acc']}/{gate['adol_n']}")
    if gate["regress"]:
        for d, s, a, c, dlt in gate["regress"]:
            print(f"- ACCEPTED REGRESSION: {d} seed {s}: candidate test {c:.3f} < adolescence {a:.3f} (−{dlt:.3f})")
    else:
        print("- accepted regressions (candidate test below adolescence test): none")
    if gate["refused_better"]:
        for d, s, a, c, dlt, old, terr, esc, z in gate["refused_better"]:
            print(f"- refused although better by > 0.01: {d} seed {s}: candidate {c:.3f} vs adolescence {a:.3f} (+{dlt:.3f}); "
                  f"original rule {'passed' if old else 'FAILED'}, territory {'passed' if terr else 'FAILED'}, "
                  f"escrow {'passed' if esc else 'FAILED'} (z {z:+.2f})")
    else:
        print("- refused candidates with test accuracy > adolescence + 0.01: none")
    print("\n### Seeds 5–7 vs seeds 2–4 (mean ± sd of seeds 2–4 from the same result files; 'inside' = |Δ| ≤ 2 sd)\n")
    for d, cells in comp:
        print(f"- **{d}**: " + "; ".join(cells))


def block_cd(title, cache, bench):
    print(f"\n## {title}\n")
    base = load(os.path.join(RES, f"{cache}.json")) or []
    adol = load(os.path.join(RES, f"{cache}_adol_protos.json")) or []
    life = (load(os.path.join(RES, f"lifecycle_{bench}.json")) or {}).get(bench, [])
    head = load(os.path.join(RES, f"head_{cache}.json")) or []
    hdr = ["seed", "growth strict / coverage / flips", "contracted protos strict / forced", "head M=10k strict / forced / coverage / changes per 100",
           "adolescence (accepted?)", "certified lifecycle (arbiter?)", "free arbiter", "uncertified", "NCM", "replay", "joint", "EWC", "fine-tune"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    agg = {k: [] for k in ["g", "cs", "cf", "hs", "hf", "hc", "hx", "a", "p", "pf", "n", "r", "j", "e", "ft"]}
    for b in base:
        s = b["seed"]
        a = next((x for x in adol if x["seed"] == s), None)
        l = next((x for x in life if x["seed"] == s), None)
        h = next((x for x in head if x["seed"] == s and x["M"] == 10000), None)
        g = b["ours"]["final"]; c = b.get("contracted_ncm", {})
        vals = dict(g=g["strict"], cs=c.get("strict"), cf=c.get("forced"), hs=h and h["strict"], hf=h and h["forced"],
                    hc=h and h["coverage"], hx=h and 100 * h["answer_flips_under_growth"] / h["n_test"],
                    a=a and (a["after"]["strict"] if a["adolescence"]["accepted"] else a["before"]["strict"]),
                    p=l and l["pfc"], pf=l and l["pfc_free_test"], n=b["ncm"], r=b["baselines"]["replay"]["final_acc"],
                    j=b["joint"], e=b["baselines"]["ewc"]["final_acc"], ft=b["baselines"]["finetune"]["final_acc"])
        for k, v in vals.items():
            if v is not None: agg[k].append(v)
        print("| " + " | ".join([str(s), f"{f3(g['strict'])} / {f3(g['coverage'])} / {b['ours']['growth_flips']}",
                                 f"{f3(vals['cs'])} / {f3(vals['cf'])}",
                                 f"{f3(vals['hs'])} / {f3(vals['hf'])} / {f3(vals['hc'])} / {'—' if vals['hx'] is None else f'{vals[chr(104)+chr(120)]:.0f}'}",
                                 f"{f3(vals['a'])} ({'—' if a is None else ('✔' if a['adolescence']['accepted'] else 'REJ: ' + str(a['adolescence'].get('why','')))})",
                                 f"{f3(vals['p'])} ({'—' if l is None else ('arb ✔' if l['pfc_accepted'] else 'arb REJ')})",
                                 f"[{f3(vals['pf'])}]", f3(l and l["uncertified_fraction"]), f3(vals["n"]), f3(vals["r"]),
                                 f3(vals["j"]), f3(vals["e"]), f3(vals["ft"])]) + " |")
    if base:
        mm = {k: (float(np.mean(v)) if v else None) for k, v in agg.items()}
        n_a = sum(1 for x in adol if x["adolescence"]["accepted"]); n_p = sum(1 for x in life if x["pfc_accepted"])
        print("| " + " | ".join(["**mean**", f"{f3(mm['g'])} / — / {sum(b['ours']['growth_flips'] for b in base)}",
                                 f"{f3(mm['cs'])} / {f3(mm['cf'])}", f"{f3(mm['hs'])} / {f3(mm['hf'])} / {f3(mm['hc'])} / {'—' if mm['hx'] is None else f'{mm[chr(104)+chr(120)]:.0f}'}",
                                 f"{f3(mm['a'])} ({n_a}/{len(adol)} acc.)", f"{f3(mm['p'])} (arb {n_p}/{len(life)})", f"[{f3(mm['pf'])}]",
                                 f"{max((l['uncertified_fraction'] for l in life), default=0):.3f} max", f3(mm['n']), f3(mm['r']), f3(mm['j']), f3(mm['e']), f3(mm['ft'])]) + " |")
        for l in life:
            print(f"- seed {l['seed']} gate: original rule {'passed' if l.get('pfc_accepted_old') else 'failed'}, territory {'passed' if l.get('territory_passed') else 'failed'}, "
                  f"escrow {'passed' if l.get('escrow_passed') else 'failed'} (z {l.get('escrow_z', float('nan')):+.2f}); candidate test {f3(l.get('pfc_candidate_test'))}, adolescence {f3(l.get('adolescence'))}"
                  + (f"; adolescence refused: {l.get('adol_refused_by')}" if l.get('adol_refused_by') else ""))


def _head10k(cache, seed):
    h = load(os.path.join(RES, f"head_{cache}.json")) or []
    return next((x for x in h if x["seed"] == seed and x["M"] == 10000), None)


def block_e():
    """Block E: ImageNet-1K-fine-tuned checkpoint, frozen (vitb1k, seeds 2-4 in one
    chain) and + first-session adapters (vitb1k_p1s<s>, one seed each), steps 1,2,5
    only. RanPAC 'full' anchors are copied from ANCHORS_PTM.md (RanPAC v3 Tab. 1,
    verified 2026-09-06 from the PDF text), not from memory; ObjectNet is not
    evaluated by RanPAC. RanPAC's A_T is a per-task mean, ours is overall B0
    last-step accuracy (named in the write-up)."""
    RANPAC_FULL = {"imr": 0.779, "ima": 0.624, "cub": 0.903, "obj": None, "omni": 0.799, "vtab": 0.922}
    print("\n## Block E — ImageNet-1K-fine-tuned checkpoint (`augreg_in21k_ft_in1k`): frozen, and + first-session adapters (seeds 2–4)\n")
    hdr = ["dataset", "config", "seed", "Phase-1 task-0 val acc", "contracted protos strict / forced",
           "head M=10k strict / forced / coverage / changes per 100", "growth strict (flips)", "NCM", "RanPAC full (A_T)"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for name in SUITE:
        for cfg, caches in [("in1k frozen", [(f"ptm_{name}_vitb1k", s) for s in EVAL]),
                            ("in1k + adapters", [(f"ptm_{name}_vitb1k_p1s{s}", s) for s in EVAL])]:
            rows = []
            for cache, s in caches:
                base = load(os.path.join(RES, f"{cache}.json")) or []
                b = next((r for r in base if r["seed"] == s), None)
                h = _head10k(cache, s)
                npz = os.path.join(RES, f"{cache}.npz")
                pv = None
                if cfg.endswith("adapters") and os.path.exists(npz):
                    d = np.load(npz, allow_pickle=False)
                    pv = float(d["phase1_val_acc"]) if "phase1_val_acc" in d.files else None
                if b is None:
                    print(f"| {NAMES[name]} | {cfg} | {s} | " + " | ".join(["did not run"] * 5) + f" | {f3(RANPAC_FULL[name]) if RANPAC_FULL[name] else 'n.a.'} |")
                    continue
                c = b.get("contracted_ncm", {})
                r = dict(pv=pv, cs=c.get("strict"), cf=c.get("forced"), hs=h and h["strict"], hf=h and h["forced"],
                         hc=h and h["coverage"], hx=h and 100 * h["answer_flips_under_growth"] / h["n_test"],
                         g=b["ours"]["final"]["strict"], gf=b["ours"]["growth_flips"], n=b["ncm"])
                rows.append(r)
                print("| " + " | ".join([NAMES[name], cfg, str(s), f3(pv) if pv is not None else "—",
                                         f"{f3(r['cs'])} / {f3(r['cf'])}",
                                         f"{f3(r['hs'])} / {f3(r['hf'])} / {f3(r['hc'])} / {'—' if r['hx'] is None else f'{r[chr(104)+chr(120)]:.0f}'}",
                                         f"{f3(r['g'])} ({r['gf']})", f3(r["n"]),
                                         f3(RANPAC_FULL[name]) if RANPAC_FULL[name] else "n.a."]) + " |")
            if rows:
                m = {k: (float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None)
                     for k in ["pv", "cs", "cf", "hs", "hf", "hc", "hx", "g", "n"]}
                print("| " + " | ".join([f"**{NAMES[name]}**", f"**{cfg}**", f"mean (n={len(rows)})", f3(m["pv"]) if m["pv"] is not None else "—",
                                         f"{f3(m['cs'])} / {f3(m['cf'])}",
                                         f"{f3(m['hs'])} / {f3(m['hf'])} / {f3(m['hc'])} / {'—' if m['hx'] is None else f'{m[chr(104)+chr(120)]:.0f}'}",
                                         f"{f3(m['g'])} ({sum(r['gf'] for r in rows)})", f3(m["n"]),
                                         f3(RANPAC_FULL[name]) if RANPAC_FULL[name] else "n.a."]) + " |")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--block-e":
        block_e()
        raise SystemExit
    block_a()
    block_cd("Block C — Banking77 (all-mpnet-base-v2, frozen), 77 classes, inc 7, seeds 2–4", "ptm_banking77_mpnet", "ptm_banking77_mpnet")
    block_cd("Block C — CLINC150 in-scope (all-mpnet-base-v2, frozen), 150 classes, inc 15, seeds 2–4", "ptm_clinc150_mpnet", "ptm_clinc150_mpnet")
    block_cd("Block D — NSL-KDD raw rows (122 one-hot/log1p features), 8 classes, inc 2, seeds 2–4", "ptm_nslkdd_raw", "ptm_nslkdd_raw")
