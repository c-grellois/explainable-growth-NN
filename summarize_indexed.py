"""The C-indexed auction: does the generalist help as a PRICE INDEX or just as
a PARTICIPANT? (results/market_indexed.json, eval seeds 2-4, --force-c.)

Isolates three things that the flat market-maker table conflated:
  maker_noC   pool-priced auction over specialists only
  maker_C     same auction with the consolidated generalist as an ordinary bidder
  indexed_c   generalist as the PRICE REFERENCE (specialist must outbid lam x C)
plus C's standalone accuracy (a refused C can still be a useful bidder) and the
routing HEADROOM the cascade leaves (arbiter - cascade), which bounds what any
router can win.
"""
from __future__ import annotations

import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def m(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    with open(os.path.join(RES, "market_indexed.json")) as f:
        d = json.load(f)
    with open(os.path.join(RES, "market_maker.json")) as f:
        base = json.load(f)

    rows = []
    for b, recs in d.items():
        casc = m([r["cascade"] for r in recs])
        arb_raw = m([r["raw_pool"]["arbiter_strict"] for r in recs])
        maker_noC = m([r["raw_pool"]["market_strict"] for r in recs])
        maker_C = m([r["consolidated_pool"]["market_strict"] for r in recs])
        idx_c = m([r["consolidated_pool"]["indexed_c"]["strict"]
                   for r in recs])
        idx_ab = m([r["consolidated_pool"]["indexed_abstain"]["strict"]
                    for r in recs])
        cov_ab = m([r["consolidated_pool"]["indexed_abstain"]["coverage"]
                    for r in recs])
        arb_C = m([r["consolidated_pool"]["arbiter_strict"] for r in recs])
        c_alone = m([r["C_alone"]["strict"] for r in recs])
        lams = [r["consolidated_pool"]["indexed_c"]["lam"] for r in recs]
        n_forced = sum(1 for r in recs if r["C_forced"])
        rows.append(dict(bench=b, cascade=casc, arb_raw=arb_raw,
                         maker_noC=maker_noC, maker_C=maker_C, idx_c=idx_c,
                         idx_ab=idx_ab, cov_ab=cov_ab, arb_C=arb_C,
                         c_alone=c_alone, lams=lams, n_forced=n_forced,
                         n=len(recs)))

    print("\n=== WHAT THE GENERALIST BUYS (eval seeds 2-4, mean strict) ===")
    print(f"{'benchmark':<12} {'maker no C':>11} {'maker +C':>10} "
          f"{'gain':>7} {'indexed':>9} {'idx-gain':>9} {'C alone':>8} "
          f"{'C forced':>9}")
    for r in rows:
        print(f"{r['bench']:<12} {r['maker_noC']:>11.3f} {r['maker_C']:>10.3f} "
              f"{r['maker_C'] - r['maker_noC']:>+7.3f} {r['idx_c']:>9.3f} "
              f"{r['idx_c'] - r['maker_C']:>+9.3f} {r['c_alone']:>8.3f} "
              f"{str(r['n_forced']) + '/' + str(r['n']):>9}")

    print("\n=== ROUTING HEADROOM: what ANY router can win ===")
    print(f"{'benchmark':<12} {'cascade':>9} {'arbiter':>9} {'headroom':>9} "
          f"{'maker +C':>10} {'vs cascade':>11} {'vs arbiter':>11}")
    for r in rows:
        print(f"{r['bench']:<12} {r['cascade']:>9.3f} {r['arb_raw']:>9.3f} "
              f"{r['arb_raw'] - r['cascade']:>+9.3f} {r['maker_C']:>10.3f} "
              f"{r['maker_C'] - r['cascade']:>+11.3f} "
              f"{r['maker_C'] - r['arb_C']:>+11.3f}")

    print("\n=== lambda selected by the pool sweep (floor = 0.40) ===")
    for r in rows:
        at_floor = sum(1 for x in r["lams"] if x <= 0.45)
        print(f"{r['bench']:<12} {r['lams']}  at-floor {at_floor}/{r['n']}")

    print("\n=== indexed_abstain (routing isolated from C answering) ===")
    for r in rows:
        print(f"{r['bench']:<12} strict {r['idx_ab']:.3f} at coverage "
              f"{r['cov_ab']:.2f}")

    out = {r["bench"]: r for r in rows}
    with open(os.path.join(RES, "market_indexed_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/market_indexed_summary.json")


if __name__ == "__main__":
    main()
