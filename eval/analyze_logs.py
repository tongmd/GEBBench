#!/usr/bin/env python3
"""Zero-API-cost analyses on existing eval logs.

(a) Formal distance vs confusion rate + permutation test
(b) Error consistency across models (Cohen's kappa, shared-error overlap)
(c) Crossvoice conditional analysis: P(cross correct | scene correct)

Outputs: paper/figs/distance_confusion.pdf, printed stats for the paper.
"""

import json
import itertools
import random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RES = ROOT / "eval" / "results"
PAPER = ROOT / "paper"

FAMILIES = [
    ["self_reference", "strange_loop", "recursion", "nesting"],
    ["cycle", "strange_loop", "mobius", "spiral", "alternation"],
    ["symmetry", "figure_ground", "isomorphism", "mobius"],
    ["tiling", "symmetry_breaking", "interference", "alternation", "braiding"],
    ["convergence", "bottleneck", "hub"],
    ["diagonalization", "self_reference", "symmetry_breaking"],
    ["infinite_descent", "recursion", "contraction_fixed_point", "convergence"],
    ["coprime_orbit", "cycle", "interference", "pigeonhole"],
    ["pigeonhole", "convergence", "bottleneck", "hub"],
    ["projective_duality", "isomorphism", "figure_ground", "symmetry"],
    ["contraction_fixed_point", "spiral", "convergence", "nesting"],
    ["aperiodic_tiling", "tiling", "interference", "alternation"],
]

LIB = json.loads((ROOT / "data" / "motif_library.json").read_text())
MOTIFS = [m["id"] for m in LIB]

BG, SURFACE = "#F7F4EF", "#FBF9F5"
TERRACOTTA, MUTED, INK = "#C4612F", "#5C635D", "#1F2421"
plt.rcParams.update({
    "font.size": 9, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC", "text.color": INK,
})

ID_TASKS = {"scene2motif", "skeleton2motif", "story2motif", "theorem2motif"}


def same_family(a, b):
    return any(a in f and b in f for f in FAMILIES)


def shares_member(a, b):
    fa = set().union(*[set(f) for f in FAMILIES if a in f]) if any(a in f for f in FAMILIES) else set()
    fb = set().union(*[set(f) for f in FAMILIES if b in f]) if any(b in f for f in FAMILIES) else set()
    return len((fa - {a}) & (fb - {b})) > 0


def distance(a, b):
    if same_family(a, b):
        return 1
    if shares_member(a, b):
        return 2
    return 3


def load_records():
    all_recs = {}
    for f in RES.glob("*.json"):
        if f.name.startswith("_"):
            continue
        d = json.loads(f.read_text())
        all_recs[f.stem] = d.get("records", [])
    return all_recs


def analysis_a(all_recs):
    """Formal distance vs confusion, with permutation test."""
    errors = []  # (true, pred)
    for model, recs in all_recs.items():
        for r in recs:
            if r["task"] in ID_TASKS and not r["correct"] and r.get("pred") in MOTIFS:
                errors.append((r["answer"], r["pred"]))

    obs = Counter(distance(a, b) for a, b in errors)
    n = len(errors)
    print(f"\n=== (a) Formal distance vs confusion ({n} pooled errors) ===")
    for d in (1, 2, 3):
        print(f"  distance {d}: {obs[d]} errors ({obs[d]/n:.1%})")

    # Chance model: for each error, what fraction of the 24 wrong options
    # sit at each distance from the true motif?
    exp = np.zeros(4)
    for a, _ in errors:
        for b in MOTIFS:
            if b != a:
                exp[distance(a, b)] += 1 / 24
    print("  chance expectation:", {d: f"{exp[d]:.1f} ({exp[d]/n:.1%})" for d in (1, 2, 3)})

    # Permutation test on within-family (d=1) mass: shuffle predicted motif
    # uniformly over the 24 wrong options
    rng = random.Random(0)
    obs_d1 = obs[1]
    count_ge = 0
    B = 10000
    for _ in range(B):
        sim = 0
        for a, _ in errors:
            b = rng.choice([m for m in MOTIFS if m != a])
            if distance(a, b) == 1:
                sim += 1
        if sim >= obs_d1:
            count_ge += 1
    p = (count_ge + 1) / (B + 1)
    print(f"  permutation test: observed d=1 mass {obs_d1}/{n}, p = {p:.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    dists = [1, 2, 3]
    obs_frac = [obs[d] / n * 100 for d in dists]
    exp_frac = [exp[d] / n * 100 for d in dists]
    x = np.arange(3)
    ax.bar(x - 0.18, obs_frac, 0.36, label="Observed errors", color=TERRACOTTA)
    ax.bar(x + 0.18, exp_frac, 0.36, label="Chance (uniform over wrong options)",
           color="#B9B3A8")
    ax.set_xticks(x)
    ax.set_xticklabels(["Same family\n(d=1)", "Shares a member\n(d=2)", "Unrelated\n(d=3)"])
    ax.set_ylabel("Share of all errors (%)")
    ax.set_title("Errors concentrate at minimal formal distance", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(PAPER / "figs" / "distance_confusion.pdf", bbox_inches="tight")
    plt.savefig(PAPER / "figs" / "distance_confusion.png", dpi=150, bbox_inches="tight")
    print("  wrote figs/distance_confusion.pdf")
    return obs, exp, p


def analysis_b(all_recs):
    """Error consistency across models."""
    print("\n=== (b) Error consistency across models ===")
    # per-item correctness
    by_model = {}
    preds = {}
    for model, recs in all_recs.items():
        by_model[model] = {r["id"]: r["correct"] for r in recs if r["task"] in ID_TASKS}
        preds[model] = {r["id"]: r.get("pred") for r in recs if r["task"] in ID_TASKS}

    models = sorted(by_model)
    for m1, m2 in itertools.combinations(models, 2):
        shared = set(by_model[m1]) & set(by_model[m2])
        if len(shared) < 20:
            continue
        v1 = np.array([by_model[m1][i] for i in shared])
        v2 = np.array([by_model[m2][i] for i in shared])
        po = (v1 == v2).mean()
        p1, p2 = v1.mean(), v2.mean()
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0
        both_wrong = [i for i in shared if not by_model[m1][i] and not by_model[m2][i]]
        same_wrong = sum(1 for i in both_wrong
                         if preds[m1][i] == preds[m2][i] and preds[m1][i] is not None)
        overlap = same_wrong / len(both_wrong) if both_wrong else float("nan")
        print(f"  {m1} vs {m2}: kappa={kappa:.2f}, "
              f"both-wrong={len(both_wrong)}, same wrong answer={overlap:.0%}")


def analysis_c(all_recs):
    """Crossvoice conditioned on scene recognition."""
    print("\n=== (c) Crossvoice | scene recognition ===")
    for model, recs in all_recs.items():
        scene = {r["id"].split(":")[1]: r["correct"] for r in recs if r["task"] == "scene2motif"}
        cross = {r["id"].split(":")[1]: r["correct"] for r in recs if r["task"] == "crossvoice"}
        shared = set(scene) & set(cross)
        if not shared:
            continue
        c_given_s = [cross[k] for k in shared if scene[k]]
        c_given_not_s = [cross[k] for k in shared if not scene[k]]
        if c_given_s and c_given_not_s:
            print(f"  {model}: P(cross✓|scene✓)={np.mean(c_given_s):.0%} (n={len(c_given_s)}), "
                  f"P(cross✓|scene✗)={np.mean(c_given_not_s):.0%} (n={len(c_given_not_s)})")


if __name__ == "__main__":
    recs = load_records()
    print("models:", ", ".join(sorted(recs)))
    analysis_a(recs)
    analysis_b(recs)
    analysis_c(recs)
