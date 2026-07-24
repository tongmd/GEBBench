#!/usr/bin/env python3
"""Experiment ② — analysis: does side-by-side comparison unlock abstraction?

Per model:
  - per-condition accuracy (25 items each)
  - within-item exact McNemar tests, each pair condition vs single
  - money contrasts:
      near vs far            (near-analog advantage — the Gentner & Namy
                              signature; equals the (near−single)−(far−single)
                              interaction because contrasts share the baseline)
      foil vs single         (does BAD comparison hurt?)
      instructed vs far      (alignment machinery present but not spontaneous?)
  - verdict: spontaneous / instructed-only / no comparison effect

Figure: dot-and-CI ladder of Δaccuracy (condition − single) per model,
paired bootstrap 95% CIs -> paper/figs/comparison_effect.pdf

Usage:  python3 exp2_comparison/analyze_comparison.py [result.json ...]
        (default: all files in exp2_comparison/results/)
Output: exp2_comparison/results.json + paper/figs/comparison_effect.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "exp2_comparison" / "results"
CONDS = ["single", "near", "far", "foil", "instructed"]

COLORS = {"near": "#2A7F62", "far": "#C4612F",
          "foil": "#9E2B25", "instructed": "#3B4A8C"}


def load(path: Path) -> tuple[str, dict[str, dict[str, bool]]]:
    """-> model, {condition: {motif: correct}}"""
    d = json.loads(path.read_text())
    table: dict[str, dict[str, bool]] = {c: {} for c in CONDS}
    for r in d["records"]:
        table[r["condition"]][r["motif"]] = bool(r["correct"])
    return d["model"], table


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """Exact McNemar (binomial on discordant pairs), a vs b over shared motifs."""
    motifs = sorted(set(a) & set(b))
    b01 = sum(1 for m in motifs if a[m] and not b[m])   # a right, b wrong
    b10 = sum(1 for m in motifs if b[m] and not a[m])   # b right, a wrong
    n = b01 + b10
    p = binomtest(b01, n, 0.5).pvalue if n else 1.0
    return {"n_discordant": n, "a_only": b01, "b_only": b10,
            "p": round(float(p), 4)}


def boot_delta(cond: dict[str, bool], base: dict[str, bool],
               n_boot=10000, seed=0) -> tuple[float, float, float]:
    """Paired bootstrap over motifs: mean(cond−single) and 95% CI."""
    motifs = sorted(set(cond) & set(base))
    d = np.array([cond[m] - base[m] for m in motifs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), \
        float(np.quantile(means, 0.975))


def verdict(stats: dict) -> str:
    sig = lambda k: stats["mcnemar_vs_single"].get(k, {}).get("p", 1) < 0.05
    dpos = lambda k: stats["delta"][k]["mean"] > 0
    if (sig("near") and dpos("near")) or (sig("far") and dpos("far")):
        return "spontaneous: yes — uninstructed pairs already beat single scenes"
    if sig("instructed") and dpos("instructed"):
        return ("instructed-only — comparison helps only when explicitly "
                "requested: alignment machinery present but not spontaneously deployed")
    return ("no — neither spontaneous nor instructed comparison reliably "
            "improves abstraction over single scenes")


def analyze(path: Path) -> dict:
    model, table = load(path)
    stats = {
        "model": model,
        "accuracy": {c: round(sum(v.values()) / len(v), 4)
                     for c, v in table.items() if v},
        "n_items": {c: len(v) for c, v in table.items()},
        "mcnemar_vs_single": {c: mcnemar(table[c], table["single"])
                              for c in CONDS[1:] if table[c]},
        "contrasts": {
            "near_vs_far (interaction)": mcnemar(table["near"], table["far"]),
            "foil_vs_single": mcnemar(table["foil"], table["single"]),
            "instructed_vs_far": mcnemar(table["instructed"], table["far"]),
        },
        "delta": {},
    }
    for c in CONDS[1:]:
        if table[c]:
            m, lo, hi = boot_delta(table[c], table["single"])
            stats["delta"][c] = {"mean": round(m, 4),
                                 "ci95": [round(lo, 4), round(hi, 4)]}
    stats["verdict"] = verdict(stats)
    return stats


def figure(all_stats: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, SURFACE, MUTED, INK = "#F7F4EF", "#FBF9F5", "#5C635D", "#1F2421"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.facecolor": BG, "axes.facecolor": SURFACE,
        "axes.edgecolor": "#E7E1D7", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#E7E1D7", "grid.alpha": 0.4,
    })

    conds = CONDS[1:]
    labels = {"near": "near pair", "far": "far pair",
              "foil": "foil pair", "instructed": "instructed (far)"}
    n = len(all_stats)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n + 0.8, 2.9),
                             sharey=True, squeeze=False)
    span = max((abs(x) for st in all_stats for d in st["delta"].values()
                for x in d["ci95"]), default=0.3) + 0.14
    for ax, st in zip(axes[0], all_stats):
        ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
        for i, c in enumerate(conds):
            if c not in st["delta"]:
                continue
            d = st["delta"][c]
            y = len(conds) - 1 - i
            ax.plot(d["ci95"], [y, y], color=COLORS[c], lw=2.2,
                    solid_capstyle="round", zorder=2)
            ax.scatter([d["mean"]], [y], s=46, color=COLORS[c],
                       edgecolor="white", lw=0.8, zorder=3)
            p = st["mcnemar_vs_single"].get(c, {}).get("p")
            if p is not None:
                ax.annotate(f"p={p:.3f}" if p >= 0.001 else "p<.001",
                            (d["ci95"][1], y), xytext=(4, -3),
                            textcoords="offset points", fontsize=6.5,
                            color=MUTED)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([labels[c] for c in reversed(conds)])
        base = st["accuracy"].get("single", float("nan"))
        ax.set_title(f"{st['model']}  (single = {base:.0%})")
        ax.set_xlabel("Δ accuracy vs single scene")
        ax.grid(axis="x", zorder=0)
        ax.set_xlim(-span, span)
    fig.suptitle("Does side-by-side comparison unlock abstraction?  "
                 "(25 motifs, within-item, paired bootstrap 95% CI)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    out = ROOT / "paper" / "figs" / "comparison_effect.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print("figure ->", out)


def main():
    paths = ([Path(p) for p in sys.argv[1:]] or
             sorted(RESULTS.glob("*.json")))
    all_stats = [analyze(p) for p in paths]
    outf = ROOT / "exp2_comparison" / "results.json"
    outf.write_text(json.dumps(all_stats, indent=1, ensure_ascii=False))
    for st in all_stats:
        print(f"\n== {st['model']}")
        print("  acc:", st["accuracy"])
        for c, t in st["mcnemar_vs_single"].items():
            print(f"  {c:>10} vs single: Δ={st['delta'][c]['mean']:+.3f} "
                  f"CI{st['delta'][c]['ci95']} McNemar p={t['p']}")
        for name, t in st["contrasts"].items():
            print(f"  {name}: {t}")
        print("  VERDICT:", st["verdict"])
    figure(all_stats)
    print("\nresults ->", outf)


if __name__ == "__main__":
    main()
