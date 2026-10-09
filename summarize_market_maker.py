"""Comparative table for the market-maker arc (results/market_maker.json).

Four routing schemes on the same frozen organisms, eval seeds 2-4:
  cascade      first-accept (birth order)
  self-priced  each organ prices itself from its own birth statistics
  maker        same per-organ price model, calibrated on the SHARED pool
  arbiter      central multi-label net over the joint evidence vector
Reported on the raw cascade and on the consolidated ([C]+cascade) organism,
with pool Brier score for the two market variants.
"""
from __future__ import annotations

import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(np.mean(xs)), float(np.std(xs)), len(xs)


def fmt(ms, prec=3):
    if ms is None:
        return "--"
    m, s, n = ms
    return f"{m:.{prec}f} ± {s:.{prec}f}"


def main():
    with open(os.path.join(RES, "market_maker.json")) as f:
        d = json.load(f)

    rows = []
    for bench, recs in d.items():
        casc = mean_sd([r["cascade"] for r in recs])
        sraw = mean_sd([r["raw"]["market_strict"] for r in recs])
        praw = mean_sd([r["raw_pool"]["market_strict"] for r in recs])
        araw = mean_sd([r["raw_pool"]["arbiter_strict"] for r in recs])
        sbr = mean_sd([r["raw"]["pool_brier"] for r in recs])
        pbr = mean_sd([r["raw_pool"]["pool_brier"] for r in recs])
        scon = mean_sd([r["consolidated"]["market_strict"]
                        for r in recs if r["consolidated"]])
        pcon = mean_sd([r["consolidated_pool"]["market_strict"]
                        for r in recs if r["consolidated_pool"]])
        acon = mean_sd([r["consolidated_pool"]["arbiter_strict"]
                        for r in recs if r["consolidated_pool"]])
        n_c = sum(1 for r in recs if r["C_accepted"])
        rows.append((bench, casc, sraw, praw, araw, sbr, pbr,
                     scon, pcon, acon, n_c, len(recs)))

    print("\n=== RAW CASCADE (no consolidation) ===")
    print(f"{'benchmark':<12} {'cascade':<16} {'self-priced':<16} "
          f"{'market-maker':<16} {'arbiter':<16}")
    for r in rows:
        print(f"{r[0]:<12} {fmt(r[1]):<16} {fmt(r[2]):<16} "
              f"{fmt(r[3]):<16} {fmt(r[4]):<16}")

    print("\n=== POOL CALIBRATION (Brier, lower is better) ===")
    print(f"{'benchmark':<12} {'self-priced':<16} {'market-maker':<16} "
          f"{'improvement':<12}")
    for r in rows:
        imp = (f"{r[5][0] / max(r[6][0], 1e-9):.0f}x"
               if r[5] and r[6] else "--")
        print(f"{r[0]:<12} {fmt(r[5]):<16} {fmt(r[6]):<16} {imp:<12}")

    print("\n=== CONSOLIDATED ([C] + cascade) ===")
    print(f"{'benchmark':<12} {'self-priced':<16} {'market-maker':<16} "
          f"{'arbiter':<16} {'C accepted':<12}")
    for r in rows:
        print(f"{r[0]:<12} {fmt(r[7]):<16} {fmt(r[8]):<16} "
              f"{fmt(r[9]):<16} {str(r[10]) + '/' + str(r[11]):<12}")

    print("\n=== THE DISSOCIATION ===")
    for r in rows:
        if not (r[5] and r[6] and r[2] and r[3]):
            continue
        dbr = r[5][0] - r[6][0]
        dacc = r[3][0] - r[2][0]
        print(f"{r[0]:<12} Brier {dbr:+.3f} (better) but accuracy "
              f"{dacc:+.3f} {'(better)' if dacc > 0 else '(WORSE)'}")

    out = {r[0]: {"cascade": r[1], "self_raw": r[2], "maker_raw": r[3],
                  "arbiter_raw": r[4], "brier_self": r[5],
                  "brier_maker": r[6], "self_con": r[7], "maker_con": r[8],
                  "arbiter_con": r[9], "n_consolidated": [r[10], r[11]]}
           for r in rows}
    with open(os.path.join(RES, "market_maker_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/market_maker_summary.json")


if __name__ == "__main__":
    main()
