#!/usr/bin/env python3
"""Emit paper/tables/ged_concentration.tex from eval/analysis_v2/confusion_v2.json.

One row per error pool: how far the chosen wrong motif sits from the true one
in the independent graph metric, as a percentile rank among the 24 wrong
options (0 = nearest). Uniform choice gives 0.500 and a 25% nearest quartile.
No number is typed by hand; the table is a view on the analysis output.
"""
import json
import os

ROOT = "/Users/stone/GEB-benchmark"
SRC = os.path.join(ROOT, "eval", "analysis_v2", "confusion_v2.json")
OUT = os.path.join(ROOT, "paper", "tables", "ged_concentration.tex")

ROWS = [("pooled", "All errors"),
        ("frontier", "Frontier models only"),
        ("vision_tasks", "Scene and skeleton voices"),
        ("text_tasks", "Story and theorem voices")]


def fmt_p(p):
    return r"$10^{-4}$" if p < 2e-4 else f"{p:.3f}"


def main():
    g = json.load(open(SRC))["ged_concentration"]
    lines = [r"\begin{tabular}{lrccc}", r"\toprule",
             r"Error pool & $n$ & Mean rank & Nearest quartile & $p$ \\",
             r"\midrule"]
    for key, label in ROWS:
        d = g[key]
        lines.append(
            f"{label} & {d['n_errors']:,} & "
            f"{d['mean_percentile_rank']:.3f} & "
            f"{100 * d['share_in_nearest_quartile']:.0f}\\% & "
            f"{fmt_p(d['p_perm_lower'])} \\\\".replace(",", "{,}"))
    lines += [r"\midrule",
              r"\textit{Uniform choice} & -- & $0.500$ & $25\%$ & -- \\",
              r"\bottomrule", r"\end{tabular}"]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    for key, label in ROWS:
        d = g[key]
        print(f"  {label:28s} n={d['n_errors']:5d} "
              f"rank={d['mean_percentile_rank']:.3f}")


if __name__ == "__main__":
    main()
