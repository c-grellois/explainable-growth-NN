"""Build the anatomy explainer, explainer/anatomy.html.

Reads only existing result files (no experiment is run): the growth logs in
results/continual_<bench>.json and results/cifar100_vitb.json, the certified
lifecycle verdicts in results/lifecycle_gated_full.json, the adolescence audits,
the CIFAR organism pickle for per-organ sizes (optional: untracked, rebuilt by
exp_cifar.py), the CIFAR-100 class names (data/cifar-100-python, downloaded by
torchvision), and a few claims for the epilogue. Writes
explainer/anatomy_trace.json (evaluation seed 2 throughout) and
explainer/anatomy.html, which is explainer/anatomy_src.html with the trace
inlined so the page opens from disk with no server.

    python trace_anatomy.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "explainer", "anatomy_trace.json")
SRC_HTML = os.path.join(HERE, "explainer", "anatomy_src.html")
OUT_HTML = os.path.join(HERE, "explainer", "anatomy.html")
SEED = 2

FASHION = ["T-shirt", "trouser", "pullover", "dress", "coat", "sandal",
           "shirt", "sneaker", "bag", "ankle boot"]


def class_names(bench, n):
    if bench == "fashion":
        return FASHION[:n]
    if bench.startswith("cifar"):
        meta = pickle.load(open(os.path.join(HERE, "data", "cifar-100-python", "meta"),
                                "rb"), encoding="latin1")
        return [s.replace("_", " ") for s in meta["fine_label_names"]]
    return [str(i) for i in range(n)]


def tasks_for(bench, n_cls, seed):
    if bench.startswith("cifar"):
        order = np.random.default_rng(seed + 400).permutation(100)
        return [sorted(int(c) for c in order[i * 10:(i + 1) * 10]) for i in range(10)]
    return [[2 * i, 2 * i + 1] for i in range(n_cls // 2)]


def parse_log(log):
    """Sequential growth log -> per-task events. For task t (t>0) the order is:
    tighten events on every existing organ (the incoming task's data measured
    against each old contract), then grow (the newborn), then birth_veto."""
    tasks = []
    cur = {"tighten": [], "grow": None, "birth_veto": None}
    for e in log:
        if e["event"] == "tighten":
            cur["tighten"].append({"organ": e["organ"], "excluded": e["excluded"],
                                   "self_loss": e["self_loss"],
                                   "vetoes_added": e["vetoes_added"]})
        elif e["event"] == "grow":
            cur["grow"] = {k: e[k] for k in ("net", "cov_on_declined",
                                             "prec_on_declined", "rho", "tau", "L")}
            tasks.append(cur)
            cur = {"tighten": [], "grow": None, "birth_veto": None}
        elif e["event"] == "birth_veto":
            tasks[-1]["birth_veto"] = e["vetoes"]
    return tasks


def organ_sizes(bench, seed):
    """Per-organ storage from the cached organism pickle (CIFAR only)."""
    tag = {"cifar_vitb": "vitb", "cifar_r18": "r18"}.get(bench)
    if tag is None:
        return None
    p = os.path.join(RES, f"adol_org_{tag}_{seed}.pkl")
    if not os.path.exists(p):
        return None
    sys.path.insert(0, HERE)
    org = pickle.load(open(p, "rb"))
    out = []
    for o in org.organs:
        trunk = int(sum(p.numel() for p in o.trunk.parameters()))
        con = int(np.asarray(o.anchors.detach().cpu().numpy()).size) if hasattr(o.anchors, "detach") \
            else int(np.asarray(o.anchors).size)
        for name in ("mu", "V", "lam", "r_q", "w_q"):
            for v in getattr(o.prox, name, []):
                con += int(np.asarray(v).size)
        vet = 0
        for tup in o.vetoes:
            for w in tup:
                if hasattr(w, "parameters"):
                    vet += int(sum(q.numel() for q in w.parameters()))
                elif isinstance(w, np.ndarray):
                    vet += int(w.size)
        out.append({"trunk": trunk, "contract": con, "veto_floats": vet,
                    "n_vetoes": len(o.vetoes),
                    "pca_rank": int(np.asarray(o.prox.V[0]).shape[0]) if len(o.prox.V) else None})
    return out


def lifecycle_rec(bench, seed):
    lg = json.load(open(os.path.join(RES, "lifecycle_gated_full.json")))
    for r in lg.get(bench, []):
        if r["seed"] == seed:
            return r
    return None


def adol_audit(bench, seed):
    if bench == "cifar_vitb":
        p = os.path.join(RES, "cifar100_vitb_adol_protos.json")
        if os.path.exists(p):
            for r in json.load(open(p)):
                if r["seed"] == seed and r["adolescence"].get("accepted"):
                    a = r["audit"]
                    return {"share_C": a["share_answered_by_C"],
                            "flips_right": a["flips_to_right"],
                            "flips_wrong": a["flips_to_wrong"],
                            "gained": a["gained_answers"], "lost": a["lost_answers"],
                            "params_C": r["adolescence"].get("params_C"),
                            "before": r["before"]["strict"], "after": r["after"]["strict"],
                            "c_alone": r["c_alone"]["strict"]}
        return None
    p = os.path.join(RES, "adolescence.json")
    if not os.path.exists(p):
        return None
    recs = json.load(open(p)).get("mlp", {}).get(bench)
    if not recs:
        return None
    rec = None
    for r in (recs if isinstance(recs, list) else [recs]):
        if isinstance(r, dict) and r.get("seed") == seed:
            rec = r
    if rec is None:
        rec = recs[0] if isinstance(recs, list) else recs
    a = rec.get("audit", {})
    return {"share_C": a.get("share_answered_by_C"),
            "flips_right": a.get("flips_to_right"), "flips_wrong": a.get("flips_to_wrong"),
            "gained": a.get("gained_answers"), "lost": a.get("lost_answers"),
            "params_C": rec.get("adolescence", {}).get("params_C"),
            "before": (rec.get("before") or {}).get("strict"),
            "after": (rec.get("after") or {}).get("strict"),
            "c_alone": (rec.get("c_alone") or {}).get("strict"),
            "seed_used": rec.get("seed")}


def build(bench):
    if bench.startswith("cifar"):
        src = json.load(open(os.path.join(RES, "cifar100_vitb.json")))
    else:
        src = json.load(open(os.path.join(RES, f"continual_{bench}.json")))
    rec = next(r for r in src if r["seed"] == SEED)
    ours = rec["ours"]
    T = len(ours["per_task"])
    n_cls = 100 if bench.startswith("cifar") else 2 * T
    names = class_names(bench, n_cls)
    tasks = tasks_for(bench, n_cls, SEED)
    events = parse_log(ours["log"])
    assert len(events) == T, (bench, len(events), T)
    per_task = ours["per_task"]
    strict_after = [float(np.mean([s["strict"] for s in row])) for row in per_task]
    cov_after = [float(np.mean([s["coverage"] for s in row])) for row in per_task]
    retention = [[round(s["strict"], 4) for s in row] for row in per_task]
    fa = {x["task"]: x for x in ours["false_accepts"]}
    sizes = organ_sizes(bench, SEED)
    organs = []
    for t in range(T):
        g = events[t]["grow"]
        organs.append({
            "id": t, "born_at": t, "classes": tasks[t],
            "class_names": [names[c] for c in tasks[t]],
            "need": g["net"], "cov_on_need": g["cov_on_declined"],
            "prec_on_need": g["prec_on_declined"], "rho": g["rho"], "tau": g["tau"],
            "birth_vetoes": events[t]["birth_veto"] or 0,
            "size": sizes[t] if sizes else None,
            "own_task_strict_over_time": [row[t]["strict"] if t < len(row) else None
                                          for row in per_task],
        })
    arrivals = []
    for t in range(T):
        arrivals.append({
            "task": t, "classes": tasks[t], "class_names": [names[c] for c in tasks[t]],
            "fa_before": fa[t]["fa_before"], "fa_after": fa[t]["fa_after"],
            "tighten": events[t]["tighten"],
            "vetoes_added_total": int(sum(x["vetoes_added"] for x in events[t]["tighten"])),
            "strict_after": strict_after[t], "coverage_after": cov_after[t],
        })
    lc = lifecycle_rec(bench, SEED) or {}
    gate = lc.get("gate", {})
    return {
        "bench": bench, "seed": SEED, "n_tasks": T, "n_classes": n_cls,
        "backbone": "ViT-B/16-in21k, frozen (768-d features)" if bench == "cifar_vitb" else
                    ("64 raw pixels" if bench == "digits" else "784 raw pixels"),
        "organs": organs, "arrivals": arrivals, "retention": retention,
        "growth_flips": ours["growth_flips"], "tighten_flips": ours["tighten_flips"],
        "n_params": ours["n_params"], "final": ours["final"],
        "lifecycle": {
            "growth": lc.get("growth"), "adolescence": lc.get("adolescence"),
            "adol_accepted": lc.get("adol_accepted"),
            "certified": lc.get("pfc"), "arbiter_accepted": lc.get("pfc_accepted"),
            "arbiter_candidate": lc.get("pfc_candidate_test"),
            "free_arbiter": lc.get("pfc_free_test"),
            "escrow_z": lc.get("escrow_z"), "territory_passed": lc.get("territory_passed"),
            "escrow_passed": lc.get("escrow_passed"),
            "uncertified_fraction": lc.get("uncertified_fraction"),
            "syn_mean": gate.get("syn_mean"), "syn_mean_ref": gate.get("syn_mean_ref"),
            "val_new": gate.get("val_new"), "val_ref": gate.get("val_ref"),
        },
        "adolescence": adol_audit(bench, SEED),
    }


def epilogue():
    c = json.load(open(os.path.join(RES, "claims.json")))
    g = lambda *ks: _dig(c, ks)
    return {
        "apoptosis_grown": g("B_apoptosis_v4", "cifar_vitb", "deaths") or 0,
        "twin_kmnist_organs_before": g("B_redundant", "kmnist", "organs_before"),
        "twin_kmnist_params_before": _mean(g("B_redundant", "kmnist", "params_before")),
        "twin_kmnist_params_after": _mean(g("B_redundant", "kmnist", "params_after")),
        "negotiate_digits_vetoes": g("B_negotiate", "digits_eval", "market@0.1", "vetoes_added"),
        "negotiate_digits_gain": _mean(g("B_negotiate", "digits_eval", "market@0.1", "delta_strict"))
                                 or _mean(g("B_negotiate", "digits_eval", "market@0.1", "gain")),
        "gate_runs": 102, "gate_harmful_refused": "4/4", "gate_miss": 1,
        "uncertified_runs": 45,
    }


def _dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _mean(v):
    if isinstance(v, dict) and "mean" in v:
        return v["mean"]
    return v


if __name__ == "__main__":
    out = {"seed": SEED, "benches": {}, "epilogue": epilogue()}
    for b in ["digits", "mnist", "cifar_vitb"]:
        out["benches"][b] = build(b)
        x = out["benches"][b]
        print(f"{b}: {x['n_tasks']} tasks, growth {x['lifecycle']['growth']} -> adol "
              f"{x['lifecycle']['adolescence']} -> certified {x['lifecycle']['certified']}"
              f" | vetoes/arrival {[a['vetoes_added_total'] for a in x['arrivals']]}"
              f" | birth vetoes {[o['birth_vetoes'] for o in x['organs']]}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    trace = json.dumps(out, indent=1, default=float)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(trace)
    print("wrote", OUT, os.path.getsize(OUT) // 1000, "KB")
    with open(SRC_HTML, encoding="utf-8") as f:
        html = f.read().replace("__TRACE__", trace)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", OUT_HTML, os.path.getsize(OUT_HTML) // 1000, "KB")
