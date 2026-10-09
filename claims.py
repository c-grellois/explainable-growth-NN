"""Regenerate every quantitative claim in REPORT.md from per-seed records.

Run:  python claims.py   ->  results/claims.json + printed summary.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ms(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs)), "n": len(xs),
            "per_seed": [float(x) for x in xs]}


def main():
    claims = {}

    # ---- B0 toy ----------------------------------------------------------- #
    with open(os.path.join(RES, "toy.json")) as f:
        toy = json.load(f)
    claims["B0_toy"] = {
        "seeds": len(toy),
        "retention_flips_total": sum(r["retention"]["flips"] for r in toy),
        "coverage_monotone": all(
            all(a["coverage"] <= b["coverage"] + 1e-9
                for a, b in zip(r["stages"], r["stages"][1:])) for r in toy),
        "seeds_with_growth": sum(r["n_organs"] > 1 for r in toy),
        "final_strict": ms([r["final"]["strict"] for r in toy]),
        "final_precision": ms([r["final"]["precision"] for r in toy]),
        "coverage_first_to_last": [
            [r["stages"][0]["coverage"], r["stages"][-1]["coverage"]] for r in toy],
        "ctrlA_violations_per_seed": [r["ctrlA_flips"] for r in toy],
        "ctrlB_covered_acc_drop": [
            [r["ctrlB_covered_acc_before"], r["ctrlB_covered_acc_after"]] for r in toy],
    }

    # ---- B2 continual ------------------------------------------------------ #
    for ds in ["digits", "mnist", "kmnist", "fashion",
               "letter", "satimage", "robot"]:
        p = os.path.join(RES, f"continual_{ds}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            recs = json.load(f)
        claims[f"B2_continual_{ds}"] = {
            "seeds": len(recs),
            "ours_strict": ms([r["ours"]["final"]["strict"] for r in recs]),
            "ours_forced": ms([r["ours"]["final"]["forced"] for r in recs]),
            "ours_coverage": ms([r["ours"]["final"]["coverage"] for r in recs]),
            "ours_precision": ms([r["ours"]["final"]["precision"] for r in recs]),
            "growth_flips_total": sum(r["ours"]["growth_flips"] for r in recs),
            "tighten_reroutes": {
                "n": sum(r["ours"]["tighten_flips"]["n"] for r in recs),
                "to_right": sum(r["ours"]["tighten_flips"]["to_right"] for r in recs),
                "to_wrong": sum(r["ours"]["tighten_flips"]["to_wrong"] for r in recs)},
            "finetune": ms([r["baselines"]["finetune"]["final_acc"] for r in recs]),
            "ewc": ms([r["baselines"]["ewc"]["final_acc"] for r in recs]),
            "replay": ms([r["baselines"]["replay"]["final_acc"] for r in recs]),
            "joint_upper": ms([r["joint_acc"] for r in recs]),
            "ours_forget_strict": ms([r["ours_forget_strict"] for r in recs]),
            "finetune_forgetting": ms([r["baselines"]["finetune"]["forgetting"] for r in recs]),
            "params_ours": recs[0]["ours"]["n_params"],
            "params_baseline": recs[0]["baselines"]["finetune"]["n_params"],
        }

    # ---- B1 static --------------------------------------------------------- #
    p = os.path.join(RES, "static.json")
    if os.path.exists(p):
        with open(p) as f:
            static = json.load(f)
        claims["B1_static"] = {}
        for ds, recs in static.items():
            claims["B1_static"][ds] = {
                "seeds": len(recs),
                "organism_strict": ms([r["final"]["strict"] for r in recs]),
                "organism_forced": ms([r["final"]["forced"] for r in recs]),
                "organism_coverage": ms([r["final"]["coverage"] for r in recs]),
                "organism_certacc": ms([r["certacc"] for r in recs]),
                "n_organs": [r["n_organs"] for r in recs],
                "single_strict": ms([r["single"]["strict"] for r in recs]),
                "single_forced": ms([r["single"]["forced"] for r in recs]),
                "single_certacc": ms([r["single_certacc"] for r in recs]),
                "plain_acc": ms([r["plain_acc"] for r in recs]),
            }

    # ---- B-replace ---------------------------------------------------------- #
    p = os.path.join(RES, "replace_digits.json")
    if os.path.exists(p):
        with open(p) as f:
            rep = json.load(f)
        claims["B_replace_digits"] = {
            "seeds": len(rep),
            "agree_region": {a: ms([r[a]["agree_region"] for r in rep])
                             for a in ["same", "wider", "deeper"]},
            "agree_elsewhere": {a: ms([r[a]["agree_else"] for r in rep])
                                for a in ["same", "wider", "deeper"]},
            "swapped_agreement": ms([r["swapped_agree_region"] for r in rep]),
        }

    # ---- OOD geometry shoot-out + learned tightening --------------------- #
    p = os.path.join(RES, "ood_geometry.json")
    if os.path.exists(p):
        with open(p) as f:
            claims["OOD_geometry"] = json.load(f)
    p = os.path.join(RES, "veto_pca.json")
    if os.path.exists(p):
        with open(p) as f:
            veto = json.load(f)
        claims["veto_measurement"] = {
            ds: {k: v[k] for k in ["exclusion", "self_loss_real", "transfer_next"]}
            for ds, v in veto.items()}
    p = os.path.join(RES, "ablate_tighten.json")
    if os.path.exists(p):
        with open(p) as f:
            ab = json.load(f)
        claims["ablate_tightening_mnist"] = {
            cell: {"strict": ms([r["strict"] for r in rows]),
                   "coverage": ms([r["coverage"] for r in rows]),
                   "precision": ms([r["precision"] for r in rows]),
                   "growth_flips": sum(r["growth_flips"] for r in rows)}
            for cell, rows in ab.items()}

    # ---- B-sheet: the organism on the spatial substrate -------------------- #
    p = os.path.join(RES, "sheet.json")
    if os.path.exists(p):
        with open(p) as f:
            sheet = json.load(f)
        claims["B_sheet"] = {}
        for kind, recs in sheet.items():
            les_diag, les_lower, les_upper = [], [], []
            for r in recs:
                base = np.array(r["lesion"]["base"])
                L = np.array(r["lesion"]["lesioned"])
                dmg = base[None, :] - L
                k = len(dmg)
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            les_diag.append(dmg[i, j])
                        elif i > j:
                            les_lower.append(abs(dmg[i, j]))
                        else:
                            les_upper.append(dmg[i, j])
            claims["B_sheet"][kind] = {
                "seeds": len(recs),
                "ours_strict": ms([r["ours"]["final"]["strict"] for r in recs]),
                "ours_forced": ms([r["ours"]["final"]["forced"] for r in recs]),
                "ours_coverage": ms([r["ours"]["final"]["coverage"] for r in recs]),
                "ours_precision": ms([r["ours"]["final"]["precision"] for r in recs]),
                "deaf_strict": ms([r["ours_nolisten"]["final"]["strict"] for r in recs]),
                "growth_flips_total": sum(r["ours"]["growth_flips"]
                                          + r["ours_nolisten"]["growth_flips"]
                                          for r in recs),
                "finetune": ms([r["baselines"]["finetune"]["final_acc"] for r in recs]),
                "ewc": ms([r["baselines"]["ewc"]["final_acc"] for r in recs]),
                "replay": ms([r["baselines"]["replay"]["final_acc"] for r in recs]),
                "joint": ms([r["joint"] for r in recs]),
                "anatomy_Q": ms([r["anatomy"]["Q"] for r in recs]),
                "within_frac": ms([r["anatomy"]["within_frac"] for r in recs]),
                "edge_len": ms([r["anatomy"]["mean_edge_len"] for r in recs]),
                "random_pair_len": ms([r["anatomy"]["random_pair_len"] for r in recs]),
                "lesion_diag_damage": ms(les_diag),
                "lesion_lower_tri_absmax": float(max(les_lower)) if les_lower else 0.0,
                "lesion_upper_tri_max": float(max(les_upper)) if les_upper else 0.0,
                "params_ours": recs[0]["ours"]["n_params"],
                "params_mono": recs[0]["baselines"]["finetune"]["n_params"],
            }

    # ---- B-adolescence: the verified metamorphosis ------------------------ #
    p = os.path.join(RES, "adolescence.json")
    if os.path.exists(p):
        with open(p) as f:
            adol = json.load(f)
        claims["B_adolescence"] = {}
        for group, sets in adol.items():
            for ds, recs in sets.items():
                acc = [r for r in recs if r["adolescence"]["accepted"]]
                entry = {
                    "seeds": len(recs), "accepted": len(acc),
                    "before_strict": ms([r["before"]["strict"] for r in recs]),
                }
                if acc:
                    entry.update({
                        "after_strict": ms([r["after"]["strict"] for r in acc]),
                        "after_precision": ms([r["after"]["precision"] for r in acc]),
                        "c_alone_strict": ms([r["c_alone"]["strict"] for r in acc]),
                        "share_C": ms([r["audit"]["share_answered_by_C"] for r in acc]),
                        "flips_right": sum(r["audit"]["flips_to_right"] for r in acc),
                        "flips_wrong": sum(r["audit"]["flips_to_wrong"] for r in acc),
                        "lost_answers": sum(r["audit"]["lost_answers"] for r in acc),
                        "params_before": recs[0].get("params", {}).get("before")
                            or acc[0]["params"]["before"],
                        "params_c_alone": acc[0]["params"]["c_alone"],
                    })
                    if "wiring_before" in acc[0]:
                        entry["edges_before"] = acc[0]["wiring_before"]["live_edges"]
                        entry["edges_c"] = acc[0]["wiring_c"]["live_edges"]
                claims["B_adolescence"][f"{group}_{ds}"] = entry

    # ---- B-renegotiate: fixed vs renegotiated contracts at adolescence ----- #
    p = os.path.join(RES, "renegotiate.json")
    if os.path.exists(p):
        with open(p) as f:
            ren = json.load(f)
        claims["B_renegotiate"] = {}
        for ds, recs in ren.items():
            entry = {"seeds": len(recs),
                     "before_strict": ms([r["before"]["strict"] for r in recs])}
            for pol in ["union", "refit"]:
                acc = [r for r in recs if r[pol]["accepted"]]
                cell = {"accepted": len(acc)}
                if acc:
                    cell.update({
                        "after_strict": ms([r[pol]["after"]["strict"] for r in acc]),
                        "share_C": ms([r[pol]["audit"]["share_answered_by_C"]
                                       for r in acc]),
                        "cov_kept": ms([r[pol]["refinement"]["coverage_kept"]
                                        for r in acc]),
                        "agreement": ms([r[pol]["refinement"]["agreement"]
                                         for r in acc]),
                        "old_terr_acc": ms([r[pol]["refinement"]["old_acc"]
                                            for r in acc]),
                        "new_terr_acc": ms([r[pol]["refinement"]["new_acc"]
                                            for r in acc]),
                    })
                entry[pol] = cell
            claims["B_renegotiate"][ds] = entry

    # ---- B-cifar: pretrained-backbone CIFAR-100 (both backbones) ----------- #
    for tag, key in [("", "B_cifar100"), ("_vitb", "B_cifar100_vitb"),
                     ("_vitb_b50", "B_cifar100_vitb_b50"),
                     # GPU-box backbones (2026-09-06): full 500/class caches
                     ("_vitb_full", "B_cifar100_vitb_full"),
                     ("_dinov2b", "B_cifar100_dinov2b"),
                     ("_vitl", "B_cifar100_vitl"),
                     # first-session adapters (Phase 1), one cache per seed
                     ("_vitb_p1s2", "B_cifar100_vitb_p1s2"),
                     ("_vitb_p1s3", "B_cifar100_vitb_p1s3"),
                     ("_vitb_p1s4", "B_cifar100_vitb_p1s4")]:
        p = os.path.join(RES, f"cifar100{tag}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            cif = json.load(f)
        claims[key] = {
            "seeds": len(cif),
            "ours_strict": ms([r["ours"]["final"]["strict"] for r in cif]),
            "ours_forced": ms([r["ours"]["final"]["forced"] for r in cif]),
            "ours_coverage": ms([r["ours"]["final"]["coverage"] for r in cif]),
            "ours_precision": ms([r["ours"]["final"]["precision"] for r in cif]),
            "growth_flips_total": sum(r["ours"]["growth_flips"] for r in cif),
            "finetune": ms([r["baselines"]["finetune"]["final_acc"] for r in cif]),
            "ewc": ms([r["baselines"]["ewc"]["final_acc"] for r in cif]),
            "replay20": ms([r["baselines"]["replay"]["final_acc"] for r in cif]),
            "ncm": ms([r["ncm"] for r in cif]),
            "joint": ms([r["joint"] for r in cif]),
            "params_ours": cif[0]["ours"]["n_params"],
        }
        if "contracted_ncm" in cif[0]:
            claims[key]["contracted_ncm_strict"] = ms(
                [r["contracted_ncm"]["strict"] for r in cif])
            claims[key]["contracted_ncm_forced"] = ms(
                [r["contracted_ncm"]["forced"] for r in cif])

    # ---- B-drift: the intermediate guarantee (audited prototype drift) ----- #
    for tag in ["r18", "vitb"]:
        p = os.path.join(RES, f"drift_{tag}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            dr = json.load(f)
        claims[f"B_drift_{tag}"] = {
            "seeds": len(dr),
            "final_forced": ms([r["final_forced"] for r in dr]),
            "pred_vs_real_corr_promised": ms(
                [r["pred_vs_real_corr_promised"] for r in dr]),
            "real_over_pred_ratio": ms([r["real_over_pred_ratio"] for r in dr]),
            "flips_right": sum(r["total_to_right"] for r in dr),
            "flips_wrong": sum(r["total_to_wrong"] for r in dr),
        }

    # ---- adolescence on the CIFAR organisms (gate decisions) --------------- #
    for tag in ["r18", "vitb", "r18_adol_protos", "vitb_adol_protos",
                "vitb_full_adol_protos", "dinov2b_adol_protos",
                "vitl_adol_protos"]:
        p = os.path.join(RES, f"cifar100_{tag}_adol.json"
                         if "protos" not in tag
                         else f"cifar100_{tag.replace('_adol_protos', '')}"
                              f"_adol_protos.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            ad = json.load(f)
        entry = {"seeds": len(ad),
                 "accepted": sum(r["adolescence"]["accepted"] for r in ad),
                 "why": [r["adolescence"].get("why") for r in ad],
                 "before_strict": ms([r["before"]["strict"] for r in ad])}
        acc = [r for r in ad if r["adolescence"]["accepted"]]
        if acc:
            entry["after_strict"] = ms([r["after"]["strict"] for r in acc])
            entry["flips_right"] = sum(r["audit"]["flips_to_right"] for r in acc)
            entry["flips_wrong"] = sum(r["audit"]["flips_to_wrong"] for r in acc)
            entry["share_C"] = ms([r["audit"]["share_answered_by_C"] for r in acc])
        claims[f"B_cifar_adol_{tag}"] = entry

    # ---- mid-life adolescence ------------------------------------------------ #
    p = os.path.join(RES, "midlife.json")
    if os.path.exists(p):
        with open(p) as f:
            ml = json.load(f)
        claims["B_midlife"] = {
            "seeds": len(ml),
            "accepted": sum(r["adol_accepted"] for r in ml),
            "mid_before": ms([r["mid_before"]["strict"] for r in ml]),
            "mid_after": ms([r["mid_after"]["strict"] for r in ml]),
            "final_strict": ms([r["final"]["strict"] for r in ml]),
            "flips_phase2": sum(r["flips_phase2_post_metamorphosis"] for r in ml),
            "share_C_on_late": ms([r["share_C_on_late_classes"] for r in ml]),
        }

    # ---- the full lifecycle: PFC arbiter + ultimate consolidation ----------- #
    p = os.path.join(RES, "pfc.json")
    if os.path.exists(p):
        with open(p) as f:
            pf = json.load(f)
        acc = [r for r in pf if r.get("pfc_accepted")]
        claims["B_pfc"] = {
            "seeds": len(pf), "accepted": len(acc),
            "growth": ms([r["growth"]["strict"] for r in pf]),
            "adolescence": ms([r["adolescence"]["strict"] for r in pf]),
            "pfc": ms([r["final"]["strict"] for r in acc]) if acc else None,
            "flips_right": sum(r["audit"]["to_right"] for r in acc),
            "flips_wrong": sum(r["audit"]["to_wrong"] for r in acc),
            "gate": [r["gate"] for r in pf],
        }

    # ---- the lifecycle on all benchmarks ------------------------------------- #
    p = os.path.join(RES, "lifecycle.json")
    if os.path.exists(p):
        with open(p) as f:
            lc = json.load(f)
        claims["B_lifecycle"] = {
            b: {"growth": ms([r["growth"] for r in recs]),
                "adolescence": ms([r["adolescence"] for r in recs]),
                "pfc": ms([r["pfc"] for r in recs]),
                "adol_accepted": sum(r["adol_accepted"] for r in recs),
                "pfc_accepted": sum(r["pfc_accepted"] for r in recs),
                "fidelity_auroc": ms([r["gate"]["fidelity_auroc"] for r in recs])}
            for b, recs in lc.items()}

    # ---- daring ideas: sleep, apoptosis, market ----------------------------- #
    for tag, key in [("sleep_dream", "B_sleep_dream"), ("sleep", "B_sleep")]:
        p = os.path.join(RES, f"{tag}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            sl = json.load(f)
        claims[key] = {b: {"final": ms([r["final"] for r in recs]),
                           "nights_accepted": sum(r.get("sleep_accepted", 0)
                                                  for r in recs),
                           "dreams_accepted": sum(r.get("dream_accepted", 0)
                                                  for r in recs),
                           "flips": sum(r.get("total_flips", 0) for r in recs)}
                      for b, recs in sl.items()}
    p = os.path.join(RES, "apoptosis.json")
    if os.path.exists(p):
        with open(p) as f:
            ap_ = json.load(f)
        claims["B_apoptosis"] = {
            b: {"deaths": [r["deaths"] for r in recs],
                "strict_before": ms([r["strict_before"] for r in recs]),
                "strict_after": ms([r["strict_after"] for r in recs]),
                "params_before": recs[0]["params_before"],
                "params_after": [r["params_after"] for r in recs],
                "genome_sleep_ok": [r["genome_sleep_ok"] for r in recs]}
            for b, recs in ap_.items()}
    p = os.path.join(RES, "market.json")
    if os.path.exists(p):
        with open(p) as f:
            mk = json.load(f)
        claims["B_market"] = {
            b: {"cascade": ms([r["cascade"] for r in recs]),
                "market": ms([r["raw"]["market_strict"] for r in recs]),
                "brier": ms([r["raw"]["pool_brier"] for r in recs]),
                "arbiter": ms([r["raw"]["arbiter_strict"] for r in recs])}
            for b, recs in mk.items()}

    # the market-maker: same per-organ price model, shared-pool calibration
    p = os.path.join(RES, "market_maker.json")
    if os.path.exists(p):
        with open(p) as f:
            mm = json.load(f)

        def _con(recs, field, key):
            xs = [r[field][key] for r in recs if r.get(field)]
            return ms(xs) if xs else None

        claims["B_market_maker"] = {
            b: {"cascade": ms([r["cascade"] for r in recs]),
                "self_raw": ms([r["raw"]["market_strict"] for r in recs]),
                "maker_raw": ms([r["raw_pool"]["market_strict"]
                                 for r in recs]),
                "arbiter_raw": ms([r["raw_pool"]["arbiter_strict"]
                                   for r in recs]),
                "brier_self": ms([r["raw"]["pool_brier"] for r in recs]),
                "brier_maker": ms([r["raw_pool"]["pool_brier"]
                                   for r in recs]),
                "self_con": _con(recs, "consolidated", "market_strict"),
                "maker_con": _con(recs, "consolidated_pool", "market_strict"),
                "arbiter_con": _con(recs, "consolidated_pool",
                                    "arbiter_strict"),
                "n_consolidated": [sum(1 for r in recs if r["C_accepted"]),
                                   len(recs)]}
            for b, recs in mm.items()}

    # the C-indexed auction: generalist as price index vs as participant
    p = os.path.join(RES, "market_indexed.json")
    if os.path.exists(p):
        with open(p) as f:
            mi = json.load(f)

        def _f(recs, path, default=None):
            xs = []
            for r in recs:
                v = r
                for k in path:
                    v = (v or {}).get(k) if isinstance(v, dict) else None
                if v is not None:
                    xs.append(v)
            return ms(xs) if xs else default

        claims["B_market_indexed"] = {}
        for b, recs in mi.items():
            casc = ms([r["cascade"] for r in recs])
            arb_raw = ms([r["raw_pool"]["arbiter_strict"] for r in recs])
            claims["B_market_indexed"][b] = {
                "maker_noC": ms([r["raw_pool"]["market_strict"] for r in recs]),
                "maker_withC": _f(recs, ["consolidated_pool",
                                         "market_strict"]),
                "indexed_c": _f(recs, ["consolidated_pool", "indexed_c",
                                       "strict"]),
                "indexed_abstain": _f(recs, ["consolidated_pool",
                                             "indexed_abstain", "strict"]),
                "indexed_abstain_cov": _f(recs, ["consolidated_pool",
                                                 "indexed_abstain",
                                                 "coverage"]),
                "lams": [r["consolidated_pool"]["indexed_c"]["lam"]
                         for r in recs if r.get("consolidated_pool")],
                "C_alone": _f(recs, ["C_alone", "strict"]),
                "C_forced": [bool(r["C_forced"]) for r in recs],
                "cascade": casc, "arbiter_raw": arb_raw,
                # what ANY router can win on this benchmark
                "headroom": float(arb_raw["mean"] - casc["mean"]),
            }

    # apoptosis v4: per-class death certificates (dev seeds, two escrow sizes)
    v4 = {}
    for tag, path in [("escrow15", "apoptosis_v4_escrow15.json"),
                      ("escrow60", "apoptosis_v4_escrow60.json")]:
        p = os.path.join(os.path.dirname(RES), "results_dev", path)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            a4 = json.load(f)
        for b, recs in a4.items():
            fails, drops, n_ref = {}, [], 0
            for r in recs:
                for x in r.get("refusals", []):
                    n_ref += 1
                    drops.append(x.get("worst_class_drop", 0.0))
                    for fl in x.get("fails", []):
                        fails[fl] = fails.get(fl, 0) + 1
                    if x.get("fails") == ["class"]:
                        fails["class_only"] = fails.get("class_only", 0) + 1
            v4[f"{b}_{tag}"] = {
                "deaths": [r["deaths"] for r in recs],
                "strict_before": ms([r["strict_before"] for r in recs]),
                "strict_after": ms([r["strict_after"] for r in recs]),
                "n_refusals": n_ref, "fail_causes": fails,
                "worst_class_drop": ms(drops) if drops else None,
                "max_class_drop": float(max(drops)) if drops else None,
                # mean-based gates that would have APPROVED these deaths
                "cov_kept_min": float(min(
                    (x["cov_kept"] for r in recs for x in r["refusals"]),
                    default=0.0)),
                "global_acc_vs_anchor": ms(
                    [x["global_acc"] - x["global_anchor"]
                     for r in recs for x in r["refusals"]]),
                "val_vs_anchor": ms(
                    [x["val_without"] - x["val_anchor"]
                     for r in recs for x in r["refusals"]]),
            }
    if v4:
        claims["B_apoptosis_v4"] = v4

    # ---- B-ensemble: redundant growth (twins) ----------------------------- #
    p = os.path.join(RES, "redundant.json")
    if os.path.exists(p):
        with open(p) as f:
            rd = json.load(f)
        claims["B_redundant"] = {
            b: {"verdicts": {v: [r["verdict"] for r in recs].count(v)
                             for v in sorted(set(r["verdict"] for r in recs))},
                "organs_before": recs[0]["organs_before"],
                "deaths": ms([r["deaths"] for r in recs]),
                "death_frac": ms([r["death_frac"] for r in recs]),
                "strict_twin_a": ms([r["strict_A"] for r in recs]),
                "strict_twin_b": ms([r["strict_B"] for r in recs]),
                "strict_merged": ms([r["strict_merged"] for r in recs]),
                "strict_after": ms([r["strict_after"] for r in recs]),
                "accuracy_cost": ms([r["strict_merged"] - r["strict_after"]
                                     for r in recs]),
                "flips": ms([r["total_flips"] for r in recs]),
                "params_before": ms([r["params_before"] for r in recs]),
                "params_after": ms([r["params_after"] for r in recs])}
            for b, recs in rd.items()}

    p = os.path.join(RES, "ensemble.json")
    if os.path.exists(p):
        with open(p) as f:
            en = json.load(f)
        claims["B_ensemble"] = {}
        for b, recs in en.items():
            K = len(recs[0]["per_k"])
            per_k = []
            for i in range(K):
                col = [r["per_k"][i] for r in recs]
                per_k.append({
                    "k": col[0]["k"],
                    "strict": ms([c["strict"] for c in col]),
                    "coverage": ms([c["coverage"] for c in col]),
                    "precision": ms([c["precision"] for c in col]),
                    "flips_vs_prefix_total": sum(c["flips_vs_prefix"]
                                                 for c in col),
                    "newly_answered": ms([c["newly_answered"] for c in col]),
                    "acc_on_newly_answered": (
                        ms(_na) if (_na := [c["acc_on_newly_answered"]
                                            for c in col
                                            if c["acc_on_newly_answered"]
                                            is not None]) else None),
                    "params": col[0]["params"]})
            last = [r["per_k"][-1] for r in recs]
            claims["B_ensemble"][b] = {
                "seeds": len(recs), "per_k": per_k,
                "vote_forced": ms([c["vote_forced"] for c in last]),
                "vote_accepting": ms([c["vote_accepting"] for c in last]),
                "arbiter": ms([c["arbiter"] for c in last]),
                "market_maker": ms([c["market_maker"] for c in last]),
                "wide_strict": ms([r["wide"]["strict"] for r in recs]),
                "wide_coverage": ms([r["wide"]["coverage"] for r in recs]),
                "wide_params": recs[0]["wide"]["params"]}

    p = os.path.join(RES, "arbiter_source.json")
    if os.path.exists(p):
        with open(p) as f:
            asrc = json.load(f)
        claims["B_arbiter_source"] = {}
        for b, recs in asrc.items():
            ks = sorted({q["k"] for r in recs for q in r["per_k"]})
            per_k = {}
            for k in ks:
                col = [q for r in recs for q in r["per_k"] if q["k"] == k]
                per_k[f"k{k}"] = {
                    "cascade": ms([c["cascade_strict"] for c in col]),
                    "arbiter": ms([c["arbiter_strict"] for c in col]),
                    "delta": ms([c["delta"] for c in col]),
                    "cascade_coverage": ms([c["cascade_coverage"]
                                            for c in col]),
                    "arbiter_coverage": ms([c["arbiter_coverage"]
                                            for c in col]),
                    "channel_reroute": ms([c["channels"]["reroute"]
                                           ["contribution"] for c in col]),
                    "channel_fill": ms([c["channels"]["fill"]["contribution"]
                                        for c in col]),
                    "channel_drop": ms([c["channels"]["drop"]["contribution"]
                                        for c in col]),
                    "reroute_wrong_to_right": sum(
                        c["channels"]["reroute"]["wrong_to_right"]
                        for c in col),
                    "reroute_right_to_wrong": sum(
                        c["channels"]["reroute"]["right_to_wrong"]
                        for c in col),
                    "reroute_inside_territory": ms(
                        [c["channels"]["reroute"]["inside_territory"]
                         for c in col]),
                    # answers taken from an organ that did NOT accept
                    "outside_territory_share": ms(
                        [c["picked_organ_did_not_accept"] for c in col]),
                    "outside_territory_gain": ms(
                        [c["gain_from_outside_territory"] for c in col]),
                    # the TERRITORY-RESPECTING arbiter: restricted to organs
                    # that accept, so every answer is contract-covered
                    "gated_strict": ms([c["gated"]["strict"] for c in col
                                        if "gated" in c]),
                    "gated_coverage": ms([c["gated"]["coverage"] for c in col
                                          if "gated" in c]),
                    "gated_vs_cascade": ms([c["gated"]["delta_vs_cascade"]
                                            for c in col if "gated" in c]),
                    "gated_vs_free": ms([c["gated"]["delta_vs_free_arbiter"]
                                         for c in col if "gated" in c]),
                    "gated_outside_territory_max": max(
                        [c["gated"]["answers_outside_territory"]
                         for c in col if "gated" in c] or [0.0]),
                    "gated_reroute_wrong_to_right": sum(
                        c["gated"]["reroute_wrong_to_right"] for c in col
                        if "gated" in c),
                    "gated_reroute_right_to_wrong": sum(
                        c["gated"]["reroute_right_to_wrong"] for c in col
                        if "gated" in c)}
            per_k["redundancy_bonus"] = ms([r["redundancy_bonus"]
                                            for r in recs])
            claims["B_arbiter_source"][b] = per_k

    # ---- the territory-respecting arbiter on the flagship ---------------- #
    p = os.path.join(RES, "arbiter_cifar.json")
    if os.path.exists(p):
        with open(p) as f:
            ac = json.load(f)
        claims["B_arbiter_cifar"] = {
            "seeds": len(ac),
            "growth": ms([r["growth"] for r in ac]),
            "adolescence_cascade": ms([r["cascade_strict"] for r in ac]),
            "free_arbiter": ms([r["arbiter_strict"] for r in ac]),
            "gated_arbiter": ms([r["gated"]["strict"] for r in ac]),
            "cascade_coverage": ms([r["cascade_coverage"] for r in ac]),
            "free_coverage": ms([r["arbiter_coverage"] for r in ac]),
            "gated_coverage": ms([r["gated"]["coverage"] for r in ac]),
            "free_delta": ms([r["delta"] for r in ac]),
            "gated_delta": ms([r["gated"]["delta_vs_cascade"] for r in ac]),
            # how much of the free arbiter's gain is NOT contract-covered
            "outside_territory_share": ms(
                [r["picked_organ_did_not_accept"] for r in ac]),
            "outside_territory_gain": ms(
                [r["gain_from_outside_territory"] for r in ac]),
            "gated_outside_territory_max": max(
                r["gated"]["answers_outside_territory"] for r in ac),
            "channel_fill": ms([r["channels"]["fill"]["contribution"]
                                for r in ac]),
            "channel_reroute": ms([r["channels"]["reroute"]["contribution"]
                                   for r in ac]),
            "channel_drop": ms([r["channels"]["drop"]["contribution"]
                                for r in ac]),
            "free_reroute_inside_territory": ms(
                [r["channels"]["reroute"]["inside_territory"] for r in ac]),
            "free_reroute_wrong_to_right": sum(
                r["channels"]["reroute"]["wrong_to_right"] for r in ac),
            "free_reroute_right_to_wrong": sum(
                r["channels"]["reroute"]["right_to_wrong"] for r in ac),
            "gated_reroute_wrong_to_right": sum(
                r["gated"]["reroute_wrong_to_right"] for r in ac),
            "gated_reroute_right_to_wrong": sum(
                r["gated"]["reroute_right_to_wrong"] for r in ac),
            # the phase-4 acceptance gate cannot see the difference
            "gate_accepted_free": sum(r["gate_free"]["accepted"] for r in ac),
            "gate_accepted_gated": sum(r["gate_gated"]["accepted"]
                                       for r in ac),
        }

    # ---- B-negotiate: the market used to AGREE THE CONTRACTS -------------- #
    NEG = [("digits_eval", "negotiate_digits_eval.json", "digits"),
           ("digits_dev_sweep", "negotiate_sweep.json", "digits"),
           ("cifar_noC", "negotiate_cifar_s2.json", "cifar_vitb"),
           ("cifar_withC", "negotiate_cifar_c_s2.json", "cifar_vitb"),
           ("cifar_withC_s34", "negotiate_cifar_c_s34.json", "cifar_vitb")]
    neg = {}
    for tag, fname, bench in NEG:
        p = os.path.join(RES, fname)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            data = json.load(f)
        if bench not in data:
            continue
        recs = data[bench]
        arms = sorted({k for r in recs for k in r["arms"]})
        entry = {"seeds": [r["seed"] for r in recs],
                 "cascade": ms([r["cascade"] for r in recs])}
        if "gated_arbiter" in recs[0]:
            entry["gated_arbiter"] = ms([r["gated_arbiter"] for r in recs])
            entry["free_arbiter"] = ms([r["free_arbiter"] for r in recs])
        for a in arms:
            xs = [r["arms"][a] for r in recs if a in r["arms"]]
            if not xs:
                continue
            right = sum(x["moved_to_right"] for x in xs)
            wrong = sum(x["moved_to_wrong"] for x in xs)
            entry[a] = {
                "strict": ms([x["strict"] for x in xs]),
                "delta": ms([x["delta"] for x in xs]),
                "coverage": ms([x["coverage"] for x in xs]),
                "vetoes_added": sum(x["vetoes_added"] for x in xs),
                "ceded_total": sum(x["ceded_total"] for x in xs),
                "moved": sum(x["moved"] for x in xs),
                "moved_to_right": right, "moved_to_wrong": wrong,
                # the diagnostic: a veto REMOVES a claim but cannot direct it,
                # so renegotiation only pays when first-accept fallback is good
                "fallback_ratio": (right / wrong) if wrong else None,
                "gate_accepted": sum(x["gate"]["accepted"] for x in xs),
                "n": len(xs)}
        neg[tag] = entry
    if neg:
        claims["B_negotiate"] = neg

    # ---- B-gatefix: the repaired acceptance gate -------------------------- #
    grecs = []
    for fname in ("gatefix.json", "gatefix_s34.json"):
        pth = os.path.join(RES, fname)
        if os.path.exists(pth):
            with open(pth) as f:
                grecs += json.load(f)
    if grecs:
        seen, uniq = set(), []
        for r in grecs:
            if r["seed"] not in seen:
                seen.add(r["seed"]); uniq.append(r)
        names = sorted({k for r in uniq for k in r["candidates"]})
        claims["B_gatefix"] = {
            "seeds": sorted(seen),
            "escrow_per_class": uniq[0]["escrow_per_class"],
            "escrow_n": uniq[0]["escrow_n"],
            "escrow_resolution": uniq[0]["escrow_resolution"],
            "candidates": {}}
        for nm in names:
            xs = [r["candidates"][nm] for r in uniq if nm in r["candidates"]]
            claims["B_gatefix"]["candidates"][nm] = {
                "test_delta": ms([x["test_delta"] for x in xs]),
                "old_gate_accepts": sum(bool(x["old_gate"]) for x in xs),
                "new_gate_accepts": sum(bool(x["new_gate"]) for x in xs),
                "n": len(xs),
                "uncertified_fraction": ms([x["uncertified_fraction"]
                                            for x in xs]),
                "mcnemar_z": ms([x["checks"]["escrow"]["mcnemar_z"]
                                 for x in xs]),
                "failed_reasons": sorted({f for x in xs for f in x["failed"]}),
            }

    # ---- B-gate-audit: every lifecycle gate decision, re-judged ----------- #
    arows = {}
    for fname in ("gate_audit_robot.json", "gate_audit.json",
                  "gate_audit_full.json"):
        pth = os.path.join(RES, fname)
        if os.path.exists(pth):
            with open(pth) as f:
                arows.update(json.load(f))
    if arows:
        flat = [(b, r) for b, recs in arows.items() for r in recs]
        harm = [r for _, r in flat if r["test_delta"] < -0.01]
        help_ = [r for _, r in flat if r["test_delta"] > 0.01]
        near = [r for _, r in flat if abs(r["test_delta"]) <= 0.01]
        claims["B_gate_audit"] = {
            "runs": len(flat),
            "benchmarks": sorted(arows),
            # the territory check refuses the whole CLASS: a policy, not a
            # detector -- it refuses helpful and harmful arbiters alike
            "territory_refuses": sum(not r["territory_passed"]
                                     for _, r in flat),
            "uncertified_fraction": ms([r["uncertified_fraction"]
                                        for _, r in flat]),
            "escrow_refuses": sum(not r["escrow_passed"] for _, r in flat),
            "decisions_changed": sum(r["old_accept"] != r["new_accept"]
                                     for _, r in flat),
            "harmful": {"n": len(harm),
                        "territory_refuses": sum(not r["territory_passed"]
                                                 for r in harm),
                        "escrow_refuses": sum(not r["escrow_passed"]
                                              for r in harm)},
            "helpful": {"n": len(help_),
                        "territory_refuses": sum(not r["territory_passed"]
                                                 for r in help_),
                        "escrow_refuses": sum(not r["escrow_passed"]
                                              for r in help_)},
            "near_neutral": {"n": len(near),
                             "escrow_refuses": sum(not r["escrow_passed"]
                                                   for r in near)},
            "per_run": [{"bench": b, "seed": r["seed"],
                         "delta": r["test_delta"],
                         "old": bool(r["old_accept"]),
                         "new": bool(r["new_accept"]),
                         "uncertified": r["uncertified_fraction"],
                         "escrow_z": r["escrow_z"]} for b, r in flat],
        }

    # ---- B-lifecycle-gated: the CERTIFIED lifecycle ----------------------- #
    p = os.path.join(RES, "lifecycle_gated_full.json")
    if not os.path.exists(p):
        p = os.path.join(RES, "lifecycle_gated.json")
    if os.path.exists(p):
        with open(p) as f:
            lg = json.load(f)
        fam = {}
        for b, recs in lg.items():
            acc_reg = [r["pfc"] - r["adolescence"] for r in recs
                       if r["pfc_accepted"] and r["pfc"] < r["adolescence"]]
            fam[b] = {
                "seeds": [r["seed"] for r in recs],
                "growth": ms([r["growth"] for r in recs]),
                "adolescence": ms([r["adolescence"] for r in recs]),
                "certified_final": ms([r["pfc"] for r in recs]),
                "free_arbiter": ms([r["pfc_free_test"] for r in recs]),
                "arbiter_accepted": sum(bool(r["pfc_accepted"])
                                        for r in recs),
                "arbiter_accepted_old_rule": sum(
                    bool(r.get("pfc_accepted_old", r["pfc_accepted"]))
                    for r in recs),
                # every answer inside a contract, by construction
                "uncertified_fraction_max": max(
                    r["uncertified_fraction"] for r in recs),
                "escrow_z": ms([r["escrow_z"] for r in recs]),
                "accepted_regressions": acc_reg,
            }
        flat = [r for recs in lg.values() for r in recs]
        claims["B_lifecycle_gated"] = {
            "per_benchmark": fam,
            "runs": len(flat),
            "arbiter_accepted": sum(bool(r["pfc_accepted"]) for r in flat),
            "arbiter_accepted_old_rule": sum(
                bool(r.get("pfc_accepted_old", r["pfc_accepted"]))
                for r in flat),
            "uncertified_fraction_max": max(r["uncertified_fraction"]
                                            for r in flat),
            # the residual hole: accepted changes that still regress
            "accepted_regressions": sorted(
                [{"bench": b, "seed": r["seed"],
                  "delta": r["pfc"] - r["adolescence"], "z": r["escrow_z"]}
                 for b, recs in lg.items() for r in recs
                 if r["pfc_accepted"] and r["pfc"] < r["adolescence"]],
                key=lambda d: d["delta"]),
        }

    # ---- B-ptm: the pretrained-model class-IL suite (gpu/extract_suite.ps1) --
    # results/ptm_<name>_<tag>.json (+ contracted_ncm), ptm_*_adol_protos.json,
    # lifecycle_ptm_<name>_<tag>.json -> claims["B_ptm"][<name>_<tag>]
    for p in sorted(glob.glob(os.path.join(RES, "ptm_*.json"))):
        base = os.path.basename(p)[len("ptm_"):-len(".json")]
        if base.endswith("_adol_protos") or base.endswith("_adol"):
            continue
        with open(p) as f:
            cif = json.load(f)
        entry = {
            "seeds": len(cif),
            "growth_strict": ms([r["ours"]["final"]["strict"] for r in cif]),
            "growth_forced": ms([r["ours"]["final"]["forced"] for r in cif]),
            "growth_flips_total": sum(r["ours"]["growth_flips"] for r in cif),
            "finetune": ms([r["baselines"]["finetune"]["final_acc"] for r in cif]),
            "ewc": ms([r["baselines"]["ewc"]["final_acc"] for r in cif]),
            "replay20": ms([r["baselines"]["replay"]["final_acc"] for r in cif]),
            "ncm": ms([r["ncm"] for r in cif]),
            "joint": ms([r["joint"] for r in cif]),
        }
        if "contracted_ncm" in cif[0]:
            entry["contracted_ncm_strict"] = ms(
                [r["contracted_ncm"]["strict"] for r in cif])
            entry["contracted_ncm_forced"] = ms(
                [r["contracted_ncm"]["forced"] for r in cif])
        pa = os.path.join(RES, f"ptm_{base}_adol_protos.json")
        if os.path.exists(pa):
            with open(pa) as f:
                ad = json.load(f)
            acc = [r for r in ad if r["adolescence"]["accepted"]]
            entry["adol_accepted"] = f"{len(acc)}/{len(ad)}"
            if acc:
                entry["adol_after_strict"] = ms([r["after"]["strict"] for r in acc])
        pl = os.path.join(RES, f"lifecycle_ptm_{base}.json")
        if os.path.exists(pl):
            with open(pl) as f:
                lg = json.load(f)
            recs = lg[f"ptm_{base}"]
            entry["lifecycle"] = {
                "certified_final": ms([r["pfc"] for r in recs]),
                "free_arbiter": ms([r["pfc_free_test"] for r in recs]),
                "arbiter_accepted": sum(bool(r["pfc_accepted"]) for r in recs),
                "uncertified_fraction_max": max(
                    r["uncertified_fraction"] for r in recs),
                "escrow_z": ms([r["escrow_z"] for r in recs]),
            }
        claims.setdefault("B_ptm", {})[base] = entry

    # ---- B-order: class-order CURRICULA on the flat protocol (a different
    # protocol; compared only to the random-order lifecycle of the same seeds).
    # results/lifecycle_order_<policy>_{cifar,mlp}.json -> claims["B_order"]
    for pol in ["easy", "hard", "grouped", "spread"]:
        for part in ["cifar", "mlp"]:
            p = os.path.join(RES, f"lifecycle_order_{pol}_{part}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                lg = json.load(f)
            for b, recs in lg.items():
                ref = (claims.get("B_lifecycle_gated", {}).get("per_benchmark", {})
                       .get(b, {}))
                ent = {
                    "seeds": [r["seed"] for r in recs],
                    "growth": ms([r["growth"] for r in recs]),
                    "growth_coverage": ms([r.get("growth_coverage", float("nan")) for r in recs]),
                    "vetoes_total": ms([r.get("vetoes_total", float("nan")) for r in recs]),
                    "adolescence": ms([r["adolescence"] for r in recs]),
                    "adol_accepted": sum(bool(r["adol_accepted"]) for r in recs),
                    "certified_final": ms([r["pfc"] for r in recs]),
                    "free_arbiter": ms([r["pfc_free_test"] for r in recs]),
                    "arbiter_accepted": sum(bool(r["pfc_accepted"]) for r in recs),
                    "uncertified_fraction_max": max(r["uncertified_fraction"] for r in recs),
                    "accepted_regressions": [r["pfc"] - r["adolescence"] for r in recs
                                             if r["pfc_accepted"] and r["pfc"] < r["adolescence"]],
                }
                if ref:
                    ent["delta_growth_vs_random"] = ent["growth"]["mean"] - ref["growth"]["mean"]
                    ent["delta_certified_vs_random"] = (ent["certified_final"]["mean"]
                                                        - ref["certified_final"]["mean"])
                claims.setdefault("B_order", {}).setdefault(pol, {})[b] = ent

    # ---- B-hier: coarse-to-fine growth on CIFAR-100 (containers, then
    # refinement inside them). results/hier_cifar_<tag>_s<seed>.json
    for tag, suffix in [(t, sfx) for t in ["vitb", "dinov2b", "vitl"]
                        for sfx in ["", "_feat", "_proto", "_twostep", "_prototwostep", "_feat_proto"]]:
        recs = []
        for p in sorted(glob.glob(os.path.join(RES, f"hier_cifar_{tag}{suffix}_s*.json"))):
            with open(p) as f:
                recs.extend(json.load(f))
        if not recs:
            continue
        fin = [r["final"] for r in recs]
        claims.setdefault("B_hier", {})[tag + suffix] = {
            "variant": {"containers": recs[0].get("containers", "taxonomy"),
                        "child": recs[0].get("child", "organ")},
            "within_vetoes_total": sum(r.get("within_vetoes_total", 0) for r in recs),
            "within_growth_flips_total": sum(r.get("within_growth_flips_total", 0) for r in recs),
            "children_params": ms([r["children_params"] for r in recs]),
            "seeds": [r["seed"] for r in recs],
            "coarse_strict": ms([r["stageA"]["coarse_strict"] for r in recs]),
            "coarse_coverage": ms([r["stageA"]["coarse_coverage"] for r in recs]),
            "container_vetoes": ms([r["stageA"]["vetoes_total"] for r in recs]),
            "container_growth_flips": sum(r["stageA"]["growth_flips"] for r in recs),
            "fine_strict": ms([x["fine_strict"] for x in fin]),
            "fine_forced": ms([x["fine_forced"] for x in fin]),
            "fine_precision": ms([x["fine_precision"] for x in fin]),
            "answered_fine": ms([x["answered_fine"] for x in fin]),
            "answered_coarse_only": ms([x["answered_coarse_only"] for x in fin]),
            "coarse_only_coarse_acc": ms([x["coarse_only_coarse_acc"] for x in fin]),
            "abstained": ms([x["abstained"] for x in fin]),
            "fine_flips_total": sum(r["fine_flips"] for r in recs),
            "children": ms([r["n_children"] for r in recs]),
            "params_total": ms([r["stageA"]["n_params"] + r["children_params"] for r in recs]),
            "flat_random_growth": claims.get(f"B_cifar100_{tag}", {}).get("ours_strict", {}).get("mean"),
            "flat_random_certified": (claims.get("B_lifecycle_gated", {}).get("per_benchmark", {})
                                      .get(f"cifar_{tag}", {}).get("certified_final", {}).get("mean")),
            "flat_protos_strict": claims.get(f"B_cifar100_{tag}", {}).get("contracted_ncm_strict", {}).get("mean"),
            "flat_grouped_growth": (claims.get("B_order", {}).get("grouped", {})
                                    .get(f"cifar_{tag}", {}).get("growth", {}).get("mean")),
            "flat_grouped_certified": (claims.get("B_order", {}).get("grouped", {})
                                       .get(f"cifar_{tag}", {}).get("certified_final", {}).get("mean")),
        }

    # ---- B-head: the contracted head (RanPAC-style RP + ridge under a contract)
    for p in sorted(glob.glob(os.path.join(RES, "head_*.json"))):
        cache = os.path.basename(p)[len("head_"):-len(".json")]
        with open(p) as f:
            recs = json.load(f)
        for M in sorted(set(r["M"] for r in recs)):
            rs = [r for r in recs if r["M"] == M]
            ent = {
                "seeds": [r["seed"] for r in rs],
                "forced": ms([r["forced"] for r in rs]),
                "strict": ms([r["strict"] for r in rs]),
                "coverage": ms([r["coverage"] for r in rs]),
                "precision": ms([r["precision"] for r in rs]),
                "strict_margin_only": ms([r["strict_margin_only"] for r in rs]),
                "answer_flips_frac": ms([r["answer_flips_under_growth"] / r["n_test"] for r in rs]),
                "lambda": [r["lambda"] for r in rs],
                "head_params": rs[0]["head_params"], "gram_floats": rs[0]["gram_floats"],
            }
            ref_key = {"cifar100_vitb": "B_cifar100_vitb", "cifar100_vitb_full": "B_cifar100_vitb_full",
                       "cifar100_dinov2b": "B_cifar100_dinov2b", "cifar100_vitl": "B_cifar100_vitl"}.get(cache)
            if ref_key and ref_key in claims:
                ent["protos_strict_ref"] = claims[ref_key]["contracted_ncm_strict"]["mean"]
                ent["protos_forced_ref"] = claims[ref_key]["contracted_ncm_forced"]["mean"]
                ent["ncm_ref"] = claims[ref_key]["ncm"]["mean"]
            elif cache.startswith("ptm_") and "B_ptm" in claims and cache[4:] in claims["B_ptm"]:
                e = claims["B_ptm"][cache[4:]]
                ent["protos_strict_ref"] = e["contracted_ncm_strict"]["mean"]
                ent["protos_forced_ref"] = e["contracted_ncm_forced"]["mean"]
                ent["ncm_ref"] = e["ncm"]["mean"]
            claims.setdefault("B_head", {}).setdefault(cache, {})[f"M{M}"] = ent

    # ---- B-progressive: consolidation schedules on flat CIFAR-ViT ----------
    for sch in ["end", "mid", "every2", "every2end"]:
        recs = []
        for p in sorted(glob.glob(os.path.join(RES, f"progressive_{sch}_vitb_s*.json"))):
            with open(p) as f:
                recs.extend(json.load(f))
        if not recs:
            continue
        claims.setdefault("B_progressive", {})[sch] = {
            "seeds": [r["seed"] for r in recs],
            "when": recs[0]["when"],
            "final_strict": ms([r["final"]["strict"] for r in recs]),
            "final_forced": ms([r["final"]["forced"] for r in recs]),
            "final_coverage": ms([r["final"]["coverage"] for r in recs]),
            "proposals": sum(len(r["consolidations"]) for r in recs),
            "accepted": sum(r["n_accepted"] for r in recs),
            "growth_flips_total": sum(r["growth_flips"] for r in recs),
            "vetoes_total": ms([r["vetoes_total"] for r in recs]),
            "share_C": ms([r["share_C"] for r in recs]),
            "n_organs_final": ms([r["n_organs"] for r in recs]),
            "n_params_final": ms([r["n_params"] for r in recs]),
            "per_stage_accepted": [[e["accepted"] for e in r["consolidations"]] for r in recs],
            "per_stage_delta": [[round(e["strict_after"] - e["strict_before"], 4)
                                 for e in r["consolidations"]] for r in recs],
        }

    # ---- B-head-blocks: a contracted head that never re-solves --------------
    for p in sorted(glob.glob(os.path.join(RES, "headblocks_*.json"))):
        base = os.path.basename(p)[len("headblocks_"):-len(".json")]
        with open(p) as f:
            recs = json.load(f)
        for M in sorted(set(r["M"] for r in recs)):
            rs = [r for r in recs if r["M"] == M]
            cache = base.replace("_notighten", "")
            ent = {
                "seeds": [r["seed"] for r in rs],
                "tighten": bool(rs[0].get("tighten", True)),
                "strict": ms([r["strict"] for r in rs]),
                "forced": ms([r["forced"] for r in rs]),
                "coverage": ms([r["coverage"] for r in rs]),
                "precision": ms([r["precision"] for r in rs]),
                # the invariant: appending a block must never change an answer
                "growth_flips_total": sum(r["growth_flips"] for r in rs),
                # tightening repairs, logged as the flat protocol logs them
                "tighten_to_abstain": sum(r["tighten_to_abstain"] for r in rs),
                "tighten_to_other": sum(r["tighten_to_other"] for r in rs),
                "tighten_to_right": sum(r["tighten_to_right"] for r in rs),
                "tighten_to_wrong": sum(r["tighten_to_wrong"] for r in rs),
                "n_blocks": rs[0]["n_blocks"],
                "head_params": rs[0]["head_params"],
            }
            h = claims.get("B_head", {}).get(cache, {}).get(f"M{M}", {})
            if h:
                ent["resolved_head_strict"] = h.get("strict", {}).get("mean")
                ent["resolved_head_changes_per100"] = (
                    h.get("answer_flips_frac", {}).get("mean") or 0) * 100
            claims.setdefault("B_head_blocks", {}).setdefault(base, {})[f"M{M}"] = ent

    # ---- B-horizon: longer streams on the same caches (inc override) ---------
    import re as _re
    hz = {}
    for p in sorted(glob.glob(os.path.join(RES, "head_*_inc*.json"))):
        m = _re.match(r"head_(.+)_inc(\d+)\.json", os.path.basename(p))
        cache, inc = m.group(1), int(m.group(2))
        with open(p) as f:
            recs = json.load(f)
        for M in sorted(set(r["M"] for r in recs)):
            rs = [r for r in recs if r["M"] == M]
            hz.setdefault(cache, {}).setdefault("head", {}).setdefault(f"M{M}", {})[f"inc{inc}"] = {
                "seeds": [r["seed"] for r in rs], "n_tasks": len(rs[0]["curve"]),
                "strict": ms([r["strict"] for r in rs]), "forced": ms([r["forced"] for r in rs]),
                "coverage": ms([r["coverage"] for r in rs]),
                # cumulative answer changes over the run, per 100 test points
                "changes_per100": ms([100.0 * r["answer_flips_under_growth"] / r["n_test"] for r in rs]),
            }
    # the inc-10 reference comes from the ordinary head files
    for cache in list(hz):
        ref = os.path.join(RES, f"head_{cache}.json")
        if os.path.exists(ref):
            with open(ref) as f:
                recs = json.load(f)
            for M in sorted(set(r["M"] for r in recs)):
                rs = [r for r in recs if r["M"] == M and r["seed"] in (2, 3, 4)]
                if rs:
                    hz[cache]["head"].setdefault(f"M{M}", {})[f"inc{rs[0]['inc']}"] = {
                        "seeds": [r["seed"] for r in rs], "n_tasks": len(rs[0]["curve"]),
                        "strict": ms([r["strict"] for r in rs]), "forced": ms([r["forced"] for r in rs]),
                        "coverage": ms([r["coverage"] for r in rs]),
                        "changes_per100": ms([100.0 * r["answer_flips_under_growth"] / r["n_test"] for r in rs])}
    for p in sorted(glob.glob(os.path.join(RES, "lifecycle_*_inc*_s*.json"))):
        m = _re.match(r"lifecycle_(.+)_inc(\d+)_s(\d+)\.json", os.path.basename(p))
        tag, inc, seed = m.group(1), int(m.group(2)), int(m.group(3))
        with open(p) as f:
            d = json.load(f)
        for bench, recs in d.items():
            for r in recs:
                cache = ("cifar100_" + tag) if bench.startswith("cifar_") else ("ptm_" + tag)
                e = hz.setdefault(cache, {}).setdefault("lifecycle", {}).setdefault(f"inc{inc}", {"per_seed": []})
                e["per_seed"].append({
                    "seed": r["seed"], "n_organs": r["n_organs"], "vetoes": r["vetoes_total"],
                    "growth": r["growth"], "growth_coverage": r["growth_coverage"],
                    "adol_accepted": r["adol_accepted"], "adolescence": r["adolescence"],
                    "arbiter_accepted": r["pfc_accepted"], "arbiter_candidate_test": r["pfc_candidate_test"],
                    "arbiter_refused_by": r["repaired_failed"] if not r["pfc_accepted"] and r["repaired_failed"]
                                          else ("original_rule" if not r["pfc_accepted"] else None),
                    "escrow_z": r["escrow_z"], "escrow_detail": r.get("escrow_detail"),
                    "uncertified_fraction": r["uncertified_fraction"], "certified": r["pfc"],
                    "free_arbiter": r["pfc_free_test"]})
    for cache, fam in hz.items():
        for inc, e in fam.get("lifecycle", {}).items():
            ps = sorted(e["per_seed"], key=lambda x: x["seed"])
            e["per_seed"] = ps
            e.update({"seeds": [x["seed"] for x in ps],
                      "growth": ms([x["growth"] for x in ps]),
                      "adolescence": ms([x["adolescence"] for x in ps]),
                      "certified": ms([x["certified"] for x in ps]),
                      "free_arbiter": ms([x["free_arbiter"] for x in ps]),
                      "adol_accepted": sum(x["adol_accepted"] for x in ps),
                      "arbiter_accepted": sum(x["arbiter_accepted"] for x in ps),
                      "uncertified_fraction_max": max(x["uncertified_fraction"] for x in ps),
                      # a refused arbiter whose candidate beat the incumbent on test by > 0.01
                      "helpful_refused": [{"seed": x["seed"], "delta": x["arbiter_candidate_test"] - x["adolescence"],
                                           "by": x["arbiter_refused_by"], "z": x["escrow_z"]}
                                          for x in ps if not x["arbiter_accepted"]
                                          and x["arbiter_candidate_test"] - x["adolescence"] > 0.01],
                      "harmful_refused": [{"seed": x["seed"], "delta": x["arbiter_candidate_test"] - x["adolescence"],
                                           "by": x["arbiter_refused_by"], "z": x["escrow_z"]}
                                          for x in ps if not x["arbiter_accepted"]
                                          and x["arbiter_candidate_test"] - x["adolescence"] < -0.01]})
        # growth flips at every horizon come from the growth records of exp_cifar (adol files)
        for p in glob.glob(os.path.join(RES, f"{cache}_adol_protos_inc*.json")):
            inc = int(_re.search(r"_inc(\d+)\.json", p).group(1))
            with open(p) as f:
                recs = json.load(f)
            fam.setdefault("adol_file", {})[f"inc{inc}"] = {
                "seeds_in_file": [r["seed"] for r in recs],
                "accepted_pre_escrow": [bool(r["adolescence"]["accepted"]) for r in recs]}
    pc = os.path.join(RES, "horizon_protos_check.json")
    if os.path.exists(pc):
        with open(pc) as f:
            d = json.load(f)
        hz["protos_horizon_check"] = {k: {"identical": v["identical_across_horizons"],
                                          "strict_by_inc": {i: r["strict"] for i, r in v["per_inc"].items()}}
                                      for k, v in d.items()}
        hz["protos_horizon_invariant_all"] = all(v["identical_across_horizons"] for v in d.values())
    gf = os.path.join(RES, "horizon_growth_flips_check.json")
    if os.path.exists(gf):
        with open(gf) as f:
            d = json.load(f)
        # the --adolescence path does not save run_ours's live counter; this is
        # the same invariant measured post hoc on every prefix of the cached
        # anatomy (appending an organ may only fill abstentions)
        hz["growth_flips_check"] = {"per_organism": d,
                                    "total_flips": int(sum(v["growth_flips_total"] for v in d.values())),
                                    "organisms": len(d),
                                    "steps_checked": int(sum(v["n_organs"] - 1 for v in d.values()))}
    if hz:
        claims["B_horizon"] = hz

    # ---- B-ood: the certificate slack as an OOD / selective-prediction score --
    for p in sorted(glob.glob(os.path.join(RES, "ood_*.json"))):
        base = os.path.basename(p)[len("ood_"):-len(".json")]
        if base == "geometry":
            continue                          # the older OOD_geometry family
        with open(p) as f:
            recs = json.load(f)
        ent = {"seeds": [r["seed"] for r in recs], "ood_sets": recs[0]["ood_sets"],
               "n_id_test": recs[0]["n_id_test"], "configs": {}, "baselines": {}}
        for name in recs[0]["configs"]:
            rs = [r["configs"][name] for r in recs if name in r["configs"]]
            e = {k: ms([x[k] for x in rs]) for k in
                 ("id_coverage", "id_strict", "id_precision", "forced_acc", "aurc",
                  "risk_at_own_coverage", "risk_at_80", "risk_at_90")}
            e["n_seeds"] = len(rs)
            e["ood"] = {o: {k: ms([x["ood"][o][k] for x in rs if o in x["ood"]])
                            for k in ("auroc", "fpr95", "ood_accept_rate")}
                        for o in rs[0]["ood"]}
            ent["configs"][name] = e
        for name in recs[0]["baselines"]:
            rs = [r["baselines"][name] for r in recs]
            e = {k: ms([x[k] for x in rs]) for k in ("forced_acc", "aurc", "risk_at_80", "risk_at_90")}
            e["ood"] = {o: {k: ms([x["ood"][o][k] for x in rs]) for k in ("auroc", "fpr95")}
                        for o in rs[0]["ood"]}
            ent["baselines"][name] = e
        claims.setdefault("B_ood", {})[base] = ent

    # ---- B-cost: state / MACs / update time per configuration -----------------
    for p in sorted(glob.glob(os.path.join(RES, "cost_*_s*.json"))):
        with open(p) as f:
            d = json.load(f)
        claims.setdefault("B_cost", {})[d["cache"]] = d

    # ---- B-size: how big the grown organism is (exp_size.py) -----------------
    for p in sorted(glob.glob(os.path.join(RES, "size_*.json"))):
        bench = os.path.basename(p)[len("size_"):-len(".json")]
        with open(p) as f:
            recs = json.load(f)
        sz = [r["grown"]["size"] for r in recs]
        conn = ms([s["connections_total"] for s in sz])
        base = recs[0]["baseline_mlp"]
        claims[f"B_size_{bench}"] = {
            "seeds": [r["seed"] for r in recs],
            **{k: ms([s[k] for s in sz])
               for k in ("n_organs", "neurons_total", "input_units", "trunk_units",
                         "veto_units", "connections_total", "trunk_connections",
                         "veto_connections", "contract_floats")},
            "strict": ms([r["grown"]["scores"]["strict"] for r in recs]),
            "coverage": ms([r["grown"]["scores"]["coverage"] for r in recs]),
            "growth_flips_total": sum(r["grown"]["growth_flips"]["total"] for r in recs),
            "growth_steps_checked": sum(r["grown"]["growth_flips"]["steps_checked"]
                                        for r in recs),
            "baseline_mlp": base,
            "connections_vs_baseline": conn["mean"] / base["connections"],
        }

    # ---- B-prune: magnitude pruning at birth, swept (exp_prune.py) -----------
    for p in sorted(glob.glob(os.path.join(RES, "prune_*.json"))):
        bench = os.path.basename(p)[len("prune_"):-len(".json")]
        with open(p) as f:
            recs = json.load(f)
        rows = {}
        for sp in sorted({r["sparsity"] for r in recs}):
            rs = [r for r in recs if r["sparsity"] == sp]
            rows[f"{sp:g}"] = {
                "seeds": [r["seed"] for r in rs],
                "n_organs": [r["n_organs"] for r in rs],
                "strict": ms([r["scores"]["strict"] for r in rs]),
                "forced": ms([r["scores"]["forced"] for r in rs]),
                "coverage": ms([r["scores"]["coverage"] for r in rs]),
                **{k: ms([r[k] for r in rs])
                   for k in ("trunk_connections_nonzero", "veto_connections",
                             "connections_total_nonzero")},
                "growth_flips_total": sum(r["growth_flips"] for r in rs),
            }
        dense = rows["0"]["connections_total_nonzero"]["mean"] if "0" in rows else None
        for row in rows.values():
            row["compression"] = (dense / row["connections_total_nonzero"]["mean"]
                                  if dense else None)
        claims[f"B_prune_{bench}"] = rows

    # ---- B-adol-escrow: the adolescence gate's real-data escrow, A/B --------
    p = os.path.join(RES, "adol_escrow_ab.json")
    if os.path.exists(p):
        with open(p) as f:
            ab = json.load(f)
        zs = [(r["new"].get("escrow") or {}).get("mcnemar_z") for r in ab]
        zs = [x for x in zs if x is not None and not np.isnan(x)]
        claims["B_adol_escrow"] = {
            "organisms_replayed": len(ab),
            "accepted_old_rule": sum(bool(r["old"]["accepted"]) for r in ab),
            "accepted_verbatim_floor": sum(bool(r["verbatim"]["accepted"]) for r in ab),
            "accepted_final_gate": sum(bool(r["new"]["accepted"]) for r in ab),
            "decisions_reaching_escrow": len(zs),
            "escrow_z": ms(zs) if zs else None,
            "escrow_z_negative": sum(1 for x in zs if x < 0),
            "verbatim_false_refusals": [
                {"tag": r["tag"], "seed": r["seed"],
                 "test_cost": r["delta_verbatim_minus_old"]}
                for r in ab if r.get("verbatim_changed")],
            "final_gate_changes": [
                {"tag": r["tag"], "seed": r["seed"],
                 "test_cost": r["delta_new_minus_old"],
                 "why": r["new"]["why"],
                 "hard_floor_classes": (r["new"].get("escrow") or {}).get("hard_floor_classes")}
                for r in ab if r["verdict_changed"]],
        }

    # ---- B-phase1: first-session adapters (battery 3) -----------------------
    # One seed-specific cache per (dataset, seed); aggregate the three seeds.
    def _npz_meta(path):
        try:
            d = np.load(path, allow_pickle=False)
            return {"phase1_val_acc": float(d["phase1_val_acc"]),
                    "phase1_tuned_params": int(d["phase1_tuned_params"]),
                    "phase1_epochs": int(d["phase1_epochs"]), "phase1_r": int(d["phase1_r"])}
        except Exception:
            return {}
    for name, ref_key, cache_fmt, res_fmt, life_fmt, bench_fmt in [
            ("cifar100", "B_cifar100_vitb_full", "cifar100_vitb_p1s{s}.npz", "cifar100_vitb_p1s{s}.json",
             "lifecycle_cifar_vitb_p1s{s}.json", "cifar_vitb_p1s{s}"),
            *[(n, None, f"ptm_{n}_vitb_p1s{{s}}.npz", f"ptm_{n}_vitb_p1s{{s}}.json",
               f"lifecycle_ptm_{n}_vitb_p1s{{s}}.json", f"ptm_{n}_vitb_p1s{{s}}")
              for n in ["imr", "ima", "cub", "obj", "omni", "vtab"]]]:
        per = []
        for s_ in (2, 3, 4):
            pr = os.path.join(RES, res_fmt.format(s=s_))
            if not os.path.exists(pr):
                continue
            with open(pr) as f:
                rec = json.load(f)
            rec = [r for r in rec if r["seed"] == s_]
            if not rec:
                continue
            r = rec[0]
            row = {"seed": s_, "growth_strict": r["ours"]["final"]["strict"],
                   "growth_flips": r["ours"]["growth_flips"],
                   "ncm": r["ncm"], "replay20": r["baselines"]["replay"]["final_acc"], "joint": r["joint"]}
            if "contracted_ncm" in r:
                row["protos_strict"] = r["contracted_ncm"]["strict"]
                row["protos_forced"] = r["contracted_ncm"]["forced"]
            hp = os.path.join(RES, "head_" + cache_fmt.format(s=s_).replace(".npz", ".json"))
            if os.path.exists(hp):
                with open(hp) as f:
                    hr = [x for x in json.load(f) if x["seed"] == s_ and x["M"] == 10000]
                if hr:
                    row.update({"head_strict": hr[0]["strict"], "head_forced": hr[0]["forced"],
                                "head_coverage": hr[0]["coverage"],
                                "head_changes_per100": 100 * hr[0]["answer_flips_under_growth"] / hr[0]["n_test"]})
            lp = os.path.join(RES, life_fmt.format(s=s_))
            if os.path.exists(lp):
                with open(lp) as f:
                    lg = json.load(f).get(bench_fmt.format(s=s_), [])
                lg = [x for x in lg if x["seed"] == s_]
                if lg:
                    row.update({"adolescence": lg[0]["adolescence"], "adol_accepted": bool(lg[0]["adol_accepted"]),
                                "certified": lg[0]["pfc"], "arbiter_accepted": bool(lg[0]["pfc_accepted"]),
                                "free_arbiter": lg[0]["pfc_free_test"],
                                "uncertified_fraction": lg[0]["uncertified_fraction"]})
            row.update(_npz_meta(os.path.join(RES, cache_fmt.format(s=s_))))
            per.append(row)
        if not per:
            continue
        agg = {"seeds": [r["seed"] for r in per], "per_seed": per}
        for k in ["growth_strict", "protos_strict", "protos_forced", "head_strict", "head_forced",
                  "head_coverage", "head_changes_per100", "adolescence", "certified", "free_arbiter",
                  "ncm", "replay20", "joint", "phase1_val_acc"]:
            vals = [r[k] for r in per if k in r]
            if vals:
                agg[k] = ms(vals)
        agg["growth_flips_total"] = sum(r.get("growth_flips", 0) for r in per)
        # denominators: a seed whose lifecycle has not run yet must not be
        # counted as a refusal
        life = [r for r in per if "adol_accepted" in r]
        agg["lifecycle_seeds"] = len(life)
        agg["adol_accepted"] = sum(bool(r["adol_accepted"]) for r in life)
        agg["arbiter_accepted"] = sum(bool(r.get("arbiter_accepted")) for r in life)
        agg["uncertified_fraction_max"] = max([r.get("uncertified_fraction", 0.0) for r in per] or [0.0])
        # frozen in21k references (same protocol, no Phase 1)
        if name == "cifar100":
            ref = claims.get("B_cifar100_vitb_full", {})
            # B_lifecycle_backbones is built later in this function, so read the
            # frozen CIFAR reference from its file directly
            cert = None
            lp = os.path.join(RES, "lifecycle_vitb_full.json")
            if os.path.exists(lp):
                with open(lp) as f:
                    recs = json.load(f).get("cifar_vitb_full", [])
                if recs:
                    cert = float(np.mean([r["pfc"] for r in recs]))
            agg["ref_frozen"] = {"protos_strict": ref.get("contracted_ncm_strict", {}).get("mean"),
                                 "protos_forced": ref.get("contracted_ncm_forced", {}).get("mean"),
                                 "head_strict": claims.get("B_head", {}).get("cifar100_vitb_full", {}).get("M10000", {}).get("strict", {}).get("mean"),
                                 "certified": cert,
                                 "growth": ref.get("ours_strict", {}).get("mean")}
        else:
            ref = claims.get("B_ptm", {}).get(f"{name}_vitb", {})
            agg["ref_frozen"] = {"protos_strict": ref.get("contracted_ncm_strict", {}).get("mean"),
                                 "protos_forced": ref.get("contracted_ncm_forced", {}).get("mean"),
                                 "head_strict": claims.get("B_head", {}).get(f"ptm_{name}_vitb", {}).get("M10000", {}).get("strict", {}).get("mean"),
                                 "certified": ref.get("lifecycle", {}).get("certified_final", {}).get("mean"),
                                 "growth": ref.get("growth_strict", {}).get("mean")}
        claims.setdefault("B_phase1", {})[name] = agg

    # ---- battery 4 (2026-09-15): the Phase-1 collector as a function, for the
    # ---- confirmation seeds (5-7, in21k) and the 1k checkpoint + adapters -----
    def _phase1_family(tag, seeds, with_lifecycle=True):
        fam = {}
        for name in ["cifar100", "imr", "ima", "cub", "obj", "omni", "vtab"]:
            if name == "cifar100":
                cache_fmt, res_fmt, bench_fmt = f"cifar100_{tag}_p1s{{s}}.npz", f"cifar100_{tag}_p1s{{s}}.json", f"cifar_{tag}_p1s{{s}}"
                life_fmts = [f"lifecycle_cifar_{tag}_p1s{{s}}_s{{s}}.json", f"lifecycle_cifar_{tag}_p1s{{s}}.json"]
            else:
                cache_fmt, res_fmt, bench_fmt = f"ptm_{name}_{tag}_p1s{{s}}.npz", f"ptm_{name}_{tag}_p1s{{s}}.json", f"ptm_{name}_{tag}_p1s{{s}}"
                life_fmts = [f"lifecycle_ptm_{name}_{tag}_p1s{{s}}_s{{s}}.json", f"lifecycle_ptm_{name}_{tag}_p1s{{s}}.json"]
            per = []
            for s_ in seeds:
                pr = os.path.join(RES, res_fmt.format(s=s_))
                if not os.path.exists(pr):
                    continue
                with open(pr) as f:
                    rec = [r for r in json.load(f) if r["seed"] == s_]
                if not rec:
                    continue
                r = rec[0]
                row = {"seed": s_, "growth_strict": r["ours"]["final"]["strict"],
                       "growth_flips": r["ours"]["growth_flips"], "ncm": r["ncm"],
                       "replay20": r["baselines"]["replay"]["final_acc"], "joint": r["joint"]}
                if "contracted_ncm" in r:
                    row["protos_strict"] = r["contracted_ncm"]["strict"]
                    row["protos_forced"] = r["contracted_ncm"]["forced"]
                hp = os.path.join(RES, "head_" + cache_fmt.format(s=s_).replace(".npz", ".json"))
                if os.path.exists(hp):
                    with open(hp) as f:
                        hr = [x for x in json.load(f) if x["seed"] == s_ and x["M"] == 10000]
                    if hr:
                        row.update({"head_strict": hr[0]["strict"], "head_forced": hr[0]["forced"],
                                    "head_coverage": hr[0]["coverage"],
                                    "head_changes_per100": 100 * hr[0]["answer_flips_under_growth"] / hr[0]["n_test"]})
                if with_lifecycle:
                    for lf in life_fmts:
                        lp = os.path.join(RES, lf.format(s=s_))
                        if not os.path.exists(lp):
                            continue
                        with open(lp) as f:
                            lg = [x for x in json.load(f).get(bench_fmt.format(s=s_), []) if x["seed"] == s_]
                        if lg:
                            row.update({"adolescence": lg[0]["adolescence"], "adol_accepted": bool(lg[0]["adol_accepted"]),
                                        "certified": lg[0]["pfc"], "arbiter_accepted": bool(lg[0]["pfc_accepted"]),
                                        "arbiter_candidate_test": lg[0]["pfc_candidate_test"],
                                        "arbiter_refused_by": (lg[0]["repaired_failed"] if not lg[0]["pfc_accepted"] and lg[0]["repaired_failed"]
                                                               else ("original_rule" if not lg[0]["pfc_accepted"] else None)),
                                        "escrow_z": lg[0]["escrow_z"], "free_arbiter": lg[0]["pfc_free_test"],
                                        "uncertified_fraction": lg[0]["uncertified_fraction"]})
                            break
                row.update(_npz_meta(os.path.join(RES, cache_fmt.format(s=s_))))
                per.append(row)
            if not per:
                continue
            agg = {"seeds": [r["seed"] for r in per], "per_seed": per}
            for k in ["growth_strict", "protos_strict", "protos_forced", "head_strict", "head_forced",
                      "head_coverage", "head_changes_per100", "adolescence", "certified", "free_arbiter",
                      "ncm", "replay20", "joint", "phase1_val_acc"]:
                vals = [r[k] for r in per if k in r]
                if vals:
                    agg[k] = ms(vals)
            agg["growth_flips_total"] = sum(r.get("growth_flips", 0) for r in per)
            life = [r for r in per if "adol_accepted" in r]
            agg["lifecycle_seeds"] = len(life)
            agg["adol_accepted"] = sum(bool(r["adol_accepted"]) for r in life)
            agg["arbiter_accepted"] = sum(bool(r.get("arbiter_accepted")) for r in life)
            agg["uncertified_fraction_max"] = max([r.get("uncertified_fraction", 0.0) for r in per] or [0.0])
            agg["accepted_regressions"] = [{"seed": r["seed"], "delta": r["arbiter_candidate_test"] - r["adolescence"], "z": r["escrow_z"]}
                                           for r in life if r.get("arbiter_accepted") and r["arbiter_candidate_test"] < r["adolescence"]]
            agg["helpful_refused"] = [{"seed": r["seed"], "delta": r["arbiter_candidate_test"] - r["adolescence"], "by": r["arbiter_refused_by"], "z": r["escrow_z"]}
                                      for r in life if not r.get("arbiter_accepted") and r["arbiter_candidate_test"] - r["adolescence"] > 0.01]
            fam[name] = agg
        return fam

    conf = _phase1_family("vitb", (5, 6, 7))
    if conf:
        # the eval-seed reference (2-4) sits in B_phase1; put the comparison beside it
        for name, agg in conf.items():
            ref = claims.get("B_phase1", {}).get(name, {})
            agg["ref_eval_seeds"] = {k: ref[k] for k in ("protos_strict", "protos_forced", "head_strict", "head_forced", "certified")
                                     if k in ref}
        claims["B_phase1_confirm"] = conf
    p1k = _phase1_family("vitb1k", (2, 3, 4), with_lifecycle=False)
    if p1k:
        claims["B_in1k_phase1"] = p1k

    # ---- B-in1k: the 1k-checkpoint caches (battery 2), next to RanPAC's rows -
    for name in ["imr", "ima", "cub", "obj", "omni", "vtab"]:
        e = claims.get("B_ptm", {}).get(f"{name}_vitb1k")
        if not e:
            continue
        h = claims.get("B_head", {}).get(f"ptm_{name}_vitb1k", {}).get("M10000", {})
        claims.setdefault("B_in1k", {})[name] = {
            "protos_strict": e["contracted_ncm_strict"], "protos_forced": e["contracted_ncm_forced"],
            "ncm": e["ncm"], "growth_strict": e["growth_strict"],
            "certified": e.get("lifecycle", {}).get("certified_final"),
            "head_strict": h.get("strict"), "head_forced": h.get("forced"), "head_coverage": h.get("coverage"),
            "in21k_protos_strict": claims.get("B_ptm", {}).get(f"{name}_vitb", {}).get("contracted_ncm_strict", {}).get("mean"),
            "in21k_head_strict": claims.get("B_head", {}).get(f"ptm_{name}_vitb", {}).get("M10000", {}).get("strict", {}).get("mean"),
        }

    # ---- B-lifecycle-backbones: the certified lifecycle on the GPU-box caches
    # (2026-09-06: vitb_full 500/class, DINOv2-B, ViT-L; eval seeds 2-4) ----- #
    for tag in ["vitb_full", "dinov2b", "vitl", "vitb_p1s2", "vitb_p1s3", "vitb_p1s4"]:
        p = os.path.join(RES, f"lifecycle_{tag}.json")
        if not os.path.exists(p) and tag.startswith("vitb_p1s"):
            p = os.path.join(RES, f"lifecycle_cifar_{tag}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            lg = json.load(f)
        for b, recs in lg.items():
            claims.setdefault("B_lifecycle_backbones", {})[b] = {
                "seeds": [r["seed"] for r in recs],
                "growth": ms([r["growth"] for r in recs]),
                "adolescence": ms([r["adolescence"] for r in recs]),
                "certified_final": ms([r["pfc"] for r in recs]),
                "free_arbiter": ms([r["pfc_free_test"] for r in recs]),
                "arbiter_accepted": sum(bool(r["pfc_accepted"]) for r in recs),
                "uncertified_fraction_max": max(
                    r["uncertified_fraction"] for r in recs),
                "escrow_z": ms([r["escrow_z"] for r in recs]),
                "accepted_regressions": [
                    r["pfc"] - r["adolescence"] for r in recs
                    if r["pfc_accepted"] and r["pfc"] < r["adolescence"]],
            }

    # ---- B-lifecycle-confirm: frozen gate on untouched seeds 5-7 ---------- #
    p = os.path.join(RES, "lifecycle_gated_confirm.json")
    if os.path.exists(p):
        with open(p) as f:
            lc = json.load(f)
        flat = [r for recs in lc.values() for r in recs]
        claims["B_lifecycle_confirm"] = {
            "seeds": sorted({r["seed"] for r in flat}),
            "runs": len(flat),
            "arbiter_accepted": sum(bool(r["pfc_accepted"]) for r in flat),
            "uncertified_fraction_max": max(r["uncertified_fraction"]
                                            for r in flat),
            # the out-of-sample guarantee: nothing above stated resolution
            "accepted_regressions": sorted(
                [{"bench": b, "seed": r["seed"],
                  "delta": r["pfc"] - r["adolescence"], "z": r["escrow_z"]}
                 for b, recs in lc.items() for r in recs
                 if r["pfc_accepted"] and r["pfc"] < r["adolescence"]],
                key=lambda x: x["delta"]),
            "helpful_refused": [
                {"bench": b, "seed": r["seed"],
                 "delta": r["pfc_candidate_test"] - r["adolescence"]}
                for b, recs in lc.items() for r in recs
                if not r["pfc_accepted"]
                and r["pfc_candidate_test"] - r["adolescence"] > 0.01],
            "per_benchmark": {
                b: {"certified_final": ms([r["pfc"] for r in recs]),
                    "adolescence": ms([r["adolescence"] for r in recs]),
                    "free_arbiter": ms([r["pfc_free_test"] for r in recs]),
                    "accepted": sum(bool(r["pfc_accepted"]) for r in recs)}
                for b, recs in lc.items()},
        }

    # ---- session-30 batteries: read-if-present so families light up ------- #
    def _load(name):
        pth = os.path.join(RES, name)
        if os.path.exists(pth):
            with open(pth) as f:
                return json.load(f)
        return None

    # speaker-disjoint FSDD: lifecycle + full baseline battery
    lf = _load("lifecycle_fsdd_spk.json")
    sh = _load("sheet_fsdd_spk.json")
    if lf and "fsdd_spk" in lf:
        recs = lf["fsdd_spk"]
        fam = {"seeds": [r["seed"] for r in recs],
               "growth": ms([r["growth"] for r in recs]),
               "certified_final": ms([r["pfc"] for r in recs]),
               "free_arbiter": ms([r["pfc_free_test"] for r in recs]),
               "arbiter_accepted": sum(bool(r["pfc_accepted"]) for r in recs),
               # val and test are DIFFERENT speakers here: the escrow's
               # exchangeability assumption is broken by construction
               "accepted_regressions": [
                   {"seed": r["seed"], "delta": r["pfc"] - r["adolescence"],
                    "z": r["escrow_z"]} for r in recs
                   if r["pfc_accepted"] and r["pfc"] < r["adolescence"]],
               "shared_speaker_growth_reference": (
                   claims["B_lifecycle_gated"]["per_benchmark"]["fsdd"]
                   ["growth"]["mean"] if "B_lifecycle_gated" in claims
                   else None)}
        if sh and "fsdd_spk" in sh:
            b = sh["fsdd_spk"]
            fam["baselines"] = {
                m: ms([r["baselines"][m]["final_acc"] for r in b])
                for m in ("finetune", "ewc", "replay")}
            fam["joint"] = ms([r["joint"] for r in b])
            fam["ours_sheet_run"] = ms([r["ours"]["final"]["strict"]
                                        for r in b])
        claims["B_fsdd_spk"] = fam

    # certified voting (majority among accepting organs vs first-accept)
    vt = _load("vote.json")
    if vt:
        claims["B_vote"] = {
            b: {"first_accept": ms([r["certified"]["first_accept"]
                                    for r in recs]),
                "vote": ms([r["certified"]["vote"] for r in recs]),
                "delta": ms([r["certified"]["vote"]
                             - r["certified"]["first_accept"] for r in recs]),
                "contest_rate": ms([r["certified"]["contest_rate"]
                                    for r in recs]),
                "changed": sum(r["certified"]["changed"] for r in recs),
                "changed_to_right": sum(r["certified"]["changed_to_right"]
                                        for r in recs),
                "changed_to_wrong": sum(r["certified"]["changed_to_wrong"]
                                        for r in recs)}
            for b, recs in vt.items()}

    # the lifetime drift ledger on the sleep organism
    for tag, fname in (("escrow_only", "sleep_dream_gated_escrow.json"),
                       ("budget_0.02", "sleep_dream_gated_budget0.02.json")):
        sl = _load(fname)
        if sl:
            claims.setdefault("B_sleep_ledger", {})[tag] = {
                b: {"final": ms([r["final"] for r in recs]),
                    "nights": len(recs[0]["nights"]),
                    "sleep_accepted": sum(r["sleep_accepted"] for r in recs),
                    "dream_accepted": sum(r["dream_accepted"] for r in recs),
                    "refused_by_escrow": sum(r["refused_by_escrow"]
                                             for r in recs),
                    "refused_by_ledger": sum(r["refused_by_ledger"]
                                             for r in recs),
                    "cumulative_drift": ms([r["ledger"]["cumulative"]
                                            for r in recs
                                            if r.get("ledger")]) if any(
                        r.get("ledger") for r in recs) else None}
                for b, recs in sl.items()}

    # escrow-independence ablation and class-order robustness
    for key, fname in (("B_lifecycle_split", "lifecycle_gated_split.json"),
                       ("B_lifecycle_order11", "lifecycle_order11.json"),
                       ("B_lifecycle_order22", "lifecycle_order22.json")):
        lx = _load(fname)
        if lx:
            claims[key] = {
                b: {"certified_final": ms([r["pfc"] for r in recs]),
                    "adolescence": ms([r["adolescence"] for r in recs]),
                    "growth": ms([r["growth"] for r in recs]),
                    "arbiter_accepted": sum(bool(r["pfc_accepted"])
                                            for r in recs),
                    "accepted_regressions": sum(
                        1 for r in recs
                        if r["pfc_accepted"] and r["pfc"] < r["adolescence"]),
                    "verdicts": [bool(r["pfc_accepted"]) for r in recs]}
                for b, recs in lx.items()}

    p = os.path.join(RES, "meta_sla.json")
    if os.path.exists(p):
        with open(p) as f:
            ms_ = json.load(f)
        claims["B_meta_sla"] = {
            tag: {"events": sum(len(r["events"]) for r in recs),
                  "violations": sum(r["violations"] for r in recs),
                  "prior_violations": sum(r["prior_violations"] for r in recs),
                  "median_tightness": ms([r["median_tightness"] for r in recs])}
            for tag, recs in ms_.items()}
    p = os.path.join(RES, "fractal.json")
    if os.path.exists(p):
        with open(p) as f:
            fr = json.load(f)
        claims["B_fractal"] = {
            "seeds": len(fr),
            "strict": ms([r["final"]["strict"] for r in fr]),
            "coverage": ms([r["final"]["coverage"] for r in fr]),
            "growth_flips": sum(r["growth_flips"] for r in fr),
            "super_replacement_agreement": ms(
                [r["super_replacement_agreement"] for r in fr])}

    p = os.path.join(RES, "isolate.json")
    if os.path.exists(p):
        with open(p) as f:
            iso = json.load(f)
        claims["B_isolate"] = {
            key: {"growth": ms([r["growth"] for r in recs]),
                  "final": ms([r["final"] for r in recs]),
                  "pfc_accepted": sum(r["pfc_ok"] for r in recs)}
            for key, recs in iso.items()}

    # ---- sampler fidelity shoot-out ----------------------------------------- #
    p = os.path.join(RES, "sampler.json")
    if os.path.exists(p):
        with open(p) as f:
            claims["B_sampler"] = json.load(f)

    # ---- per-organ sheet refinement audit ---------------------------------- #
    p = os.path.join(RES, "sheet_audit.json")
    if os.path.exists(p):
        with open(p) as f:
            sa = json.load(f)
        cov, agree, acc_b, acc_a = [], [], [], []
        for kind, recs in sa.items():
            for r in recs:
                cells = r.get("union")
                if isinstance(cells, list):
                    for c in cells:
                        cov.append(c["coverage_kept"]); agree.append(c["agreement"])
                        acc_b.append(c["acc_before"]); acc_a.append(c["acc_after"])
        claims["B_sheet_audit"] = {
            "n_cells": len(cov), "coverage_kept_min": float(min(cov)),
            "agreement": ms(agree),
            "acc_before": ms(acc_b), "acc_after": ms(acc_a),
        }

    # provenance: hash every results JSON so each claim traces to a file
    import glob as _glob, hashlib as _hl
    prov = {}
    for pth in sorted(_glob.glob(os.path.join(RES, "*.json"))):
        nm = os.path.basename(pth)
        if nm in ("claims.json", "MANIFEST.json"):
            continue
        with open(pth, "rb") as fh:
            prov[nm] = _hl.sha256(fh.read()).hexdigest()[:16]
    claims["_provenance"] = prov

    out = os.path.join(RES, "claims.json")
    with open(out, "w") as f:
        json.dump(claims, f, indent=1)
    print(json.dumps(claims, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
