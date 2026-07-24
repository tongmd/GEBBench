#!/usr/bin/env python3
"""Figure: independent formal distance replicates the confusion geometry.

Left panel : CDF of error percentile ranks under the independent typed GED
             (and the equation-token distance), vs the uniform diagonal.
Right panel: normalized GED for pairs inside vs outside a designed
             confusable family (the Mantel validation, visualized).

Output: eval/analysis_out/independent_distance.pdf / .png
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/stone/GEB-benchmark")
OUT = ROOT / "eval" / "analysis_out"

import sys
sys.path.insert(0, str(OUT))
from analyze_independent_distance import (family_comembership, load_errors,
                                          percentile_ranks)

BG = "#F7F4EF"; SURFACE = "#FBF9F5"; MUTED = "#5C635D"; INK = "#1F2421"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans"],
    "font.size": 10, "figure.facecolor": BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#E7E1D7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    ind = json.loads((OUT / "independent_distance.json").read_text())
    res = json.loads((OUT / "independent_distance_results.json").read_text())
    motifs = ind["motifs"]
    GED = np.array(ind["ged_normalized"])
    EQ = np.array(ind["eq_token_distance"])
    FAM = family_comembership(motifs)
    _, errors = load_errors(motifs)

    r_ged = percentile_ranks(errors, GED)
    r_eq = percentile_ranks(errors, EQ)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.1),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- left: CDF of percentile ranks --------------------------------
    xs = np.linspace(0, 1, 401)
    for r, color, label in [
            (r_ged, "#B3402A", "typed graph edit distance"),
            (r_eq, "#E8A03E", "equation token distance")]:
        cdf = [(r <= x).mean() for x in xs]
        ax1.plot(xs, cdf, color=color, lw=2.0, label=label)
    ax1.plot([0, 1], [0, 1], color="#999999", lw=1.2, ls=":",
             label="uniform over 24 wrong options")
    g = res["error_concentration"]["ged"]
    ax1.set_xlabel("percentile rank of the chosen wrong motif\n"
                   "(0 = formally nearest of the 24 options)")
    ax1.set_ylabel("cumulative share of errors")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.text(0.03, 0.94,
             f"mean rank {g['mean_percentile_rank']:.2f} vs 0.50\n"
             f"$p={g['p_perm_lower']:.4f}$ (10k perms)",
             transform=ax1.transAxes, fontsize=8.5, va="top")

    # ---- right: GED inside vs outside families ------------------------
    iu = np.triu_indices(len(motifs), k=1)
    inside = GED[iu][FAM[iu] == 1]
    outside = GED[iu][FAM[iu] == 0]
    parts = ax2.violinplot([inside, outside], showmeans=True, widths=0.8)
    for pc, c in zip(parts["bodies"], ["#B3402A", "#3B4A8C"]):
        pc.set_facecolor(c); pc.set_alpha(0.45)
    for k in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[k].set_color(MUTED); parts[k].set_linewidth(1.0)
    ax2.set_xticks([1, 2], ["same designed\nfamily", "different\nfamily"])
    ax2.set_ylabel("typed graph edit distance\n(normalized)")
    man = res["mantel"]["ged_vs_family"]
    ax2.set_title(f"Mantel $r={man['mantel_r']:.2f}$, "
                  f"$p={man['p']:.4f}$", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUT / "independent_distance.pdf")
    fig.savefig(OUT / "independent_distance.png", dpi=200)
    print("wrote independent_distance.pdf/png",
          f"n_inside={len(inside)} n_outside={len(outside)}")


if __name__ == "__main__":
    main()
