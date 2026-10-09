"""Figures for the paper, regenerated from results/claims.json.

fig_lifecycle.pdf  the certified lifecycle per benchmark: growth -> adolescence
                   -> territory-respecting arbiter, with the unconstrained
                   arbiter's (uncovered) score as a hollow marker.
fig_laws.pdf       (a) routing headroom vs. gate acceptance -- the gate
                   reproduces the headroom law; (b) the fallback ratio vs. the
                   outcome of boundary renegotiation -- selection cannot be
                   replicated by monotone edits.

Run:  python make_figures.py   -> paper/fig_lifecycle.pdf, paper/fig_laws.pdf
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "paper")

ORDER = ["digits", "mnist", "kmnist", "fashion", "letter", "satimage", "robot",
         "cifar_r18", "cifar_vitb", "sweeps", "shapes", "fsdd"]
LABEL = {"digits": "digits", "mnist": "MNIST", "kmnist": "KMNIST",
         "fashion": "Fashion", "letter": "letter", "satimage": "satimage",
         "robot": "wall-robot", "cifar_r18": "CIFAR-100\nR18",
         "cifar_vitb": "CIFAR-100\nViT", "sweeps": "sheet\nsweeps",
         "shapes": "sheet\nshapes", "fsdd": "spoken\ndigits"}


def fig_lifecycle(c):
    lg = c["B_lifecycle_gated"]["per_benchmark"]
    b = [x for x in ORDER if x in lg]
    g = [lg[x]["growth"]["mean"] for x in b]
    a = [lg[x]["adolescence"]["mean"] for x in b]
    f = [lg[x]["certified_final"]["mean"] for x in b]
    u = [lg[x]["free_arbiter"]["mean"] for x in b]
    x = np.arange(len(b)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.bar(x - w, g, w, color="#b8c4cc", label="growth (cascade)")
    ax.bar(x, a, w, color="#6f8fa6", label="+ adolescence")
    ax.bar(x + w, f, w, color="#1f4e79", label="+ territory-respecting arbiter (certified)")
    ax.scatter(x + w, u, s=34, facecolors="none", edgecolors="#c0392b",
               linewidths=1.4, zorder=5,
               label="unconstrained arbiter (answers outside territory; not claimed)")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[k] for k in b], fontsize=8)
    ax.set_ylabel("strict accuracy (abstention = error)")
    ax.set_ylim(0.3, 1.02); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left", frameon=False)
    ax.set_title("The certified lifecycle, evaluation seeds 2–4: every answer contract-covered, "
                 "0 accepted regressions", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lifecycle.pdf"))
    plt.close(fig)


def fig_laws(c):
    lg = c["B_lifecycle_gated"]["per_benchmark"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))

    # (a) headroom vs acceptance
    for k in ORDER:
        if k not in lg:
            continue
        d = lg[k]
        adol = d["adolescence"]["mean"]
        free_h = d["free_arbiter"]["mean"] - adol
        gated_h = d["certified_final"]["mean"] - adol
        acc = d["arbiter_accepted"]
        col = {3: "#1f4e79", 2: "#6f8fa6", 1: "#b8c4cc", 0: "#c0392b"}[acc]
        ax1.scatter(free_h, gated_h, s=70, color=col, zorder=4,
                    edgecolors="k", linewidths=0.5)
        if abs(free_h) > 0.035 or gated_h > 0.01:
            ax1.annotate(LABEL[k].replace("\n", " "), (free_h, gated_h),
                         fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax1.annotate("KMNIST, Fashion, letter, MNIST, satimage, sheets:\n"
                 "headroom absent or uncertified -> refused, 0 gain",
                 (0.012, 0.0), fontsize=6.5, xytext=(8, -34),
                 textcoords="offset points", color="#444",
                 arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    lim = [-0.13, 0.11]
    ax1.plot(lim, lim, ls=":", color="grey", lw=1, label="all headroom captured")
    ax1.axhline(0, color="grey", lw=0.6); ax1.axvline(0, color="grey", lw=0.6)
    ax1.set_xlim(lim); ax1.set_ylim(-0.02, 0.09)
    ax1.set_xlabel("routing headroom of the unconstrained arbiter (arbiter − cascade)")
    ax1.set_ylabel("gain kept by the certified arbiter")
    ax1.set_title("(a) the gate enforces the headroom law\ncolour = arbiters accepted of 3 (dark 3 … red 0)",
                  fontsize=9)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax1.grid(alpha=0.3)

    # (b) fallback ratio vs renegotiation outcome
    neg = c.get("B_negotiate", {})
    pts = []
    for tag, v in neg.items():
        for arm in ("market@0.1", "oracle@0.1"):
            if arm in v and v[arm]["fallback_ratio"] is not None:
                pts.append((v[arm]["fallback_ratio"], v[arm]["delta"]["mean"],
                            tag.split("_")[0], arm.split("@")[0]))
    seen = set()
    for fr, dl, bench, arm in pts:
        ax2.scatter(fr, dl, s=70, zorder=4, edgecolors="k", linewidths=0.5,
                    color="#1f4e79" if bench == "digits" else "#c0392b",
                    marker="o" if arm == "market" else "s")
        if (bench, arm) not in seen:
            seen.add((bench, arm))
            ax2.annotate(f"{bench} {arm}", (fr, dl), fontsize=7,
                         xytext=(5, 3), textcoords="offset points")
    ax2.axhline(0, color="grey", lw=0.6); ax2.axvline(1, ls=":", color="grey", lw=1)
    ax2.set_xscale("log")
    ax2.set_xticks([0.3, 0.5, 1, 2, 3])
    ax2.set_xticklabels(["0.3", "0.5", "1", "2", "3"])
    ax2.minorticks_off()
    ax2.text(1.05, -0.0057, "pays →", fontsize=7, color="#444")
    ax2.text(0.95, -0.0057, "← costs", fontsize=7, color="#444", ha="right")
    ax2.set_xlabel("fallback ratio (answers moved right : moved wrong), log scale")
    ax2.set_ylabel("Δ strict from boundary renegotiation")
    ax2.set_title("(b) a veto removes a claim but cannot direct it\n"
                  "circle = market, square = oracle; blue digits, red CIFAR-ViT",
                  fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_laws.pdf"))
    plt.close(fig)


def main():
    c = json.load(open(os.path.join(RES, "claims.json")))
    fig_lifecycle(c)
    fig_laws(c)
    print("wrote paper/fig_lifecycle.pdf, paper/fig_laws.pdf")


if __name__ == "__main__":
    main()
