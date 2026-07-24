#!/usr/bin/env python3
"""Experiment ③ analysis — the psychophysical function of abstraction.

Per model:
  - logistic regression of correctness vs naturalization level, motif as
    fixed-effect intercepts (approximation of a motif random intercept)
  - 2-parameter psychometric sigmoid (threshold, slope) with guess floor
    fixed at chance (1/25), fit by binomial maximum likelihood
  - bootstrap CI on the threshold (cluster bootstrap over motifs)

Across models:
  - likelihood-ratio test for slope×model interaction (shared-slope law)

Per motif:
  - pooled-model threshold vs confusable-family membership/degree

Figure: classic psychophysics plot -> paper/figs/psychophysics.pdf
Summary  -> exp3_ladder/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
EXP = ROOT / "exp3_ladder"
FIGS = ROOT / "paper" / "figs"

GUESS = 1 / 25          # 25-way menu chance floor
LEVELS = np.arange(0, 7)

# same confusable families as eval/build_items.py
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

# paper palette
TERRACOTTA, MUTED, INK = "#C4612F", "#5C635D", "#1F2421"
MODEL_COLORS = [TERRACOTTA, "#3B4A8C", "#5C8A5C", "#8C5C8A", "#B89B4A",
                "#4A8CB8"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans", "Helvetica"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def family_degree(mid: str) -> int:
    nb = set()
    for f in FAMILIES:
        if mid in f:
            nb |= set(f)
    return len(nb - {mid})


# ---------------------------------------------------------------------------
# psychometric fit: p(L) = GUESS + (1 - GUESS) / (1 + exp(slope * (L - thr)))
# slope > 0 => accuracy decreases with naturalization level
# ---------------------------------------------------------------------------

def sigmoid(L, thr, slope):
    return GUESS + (1 - GUESS) / (1 + np.exp(slope * (np.asarray(L) - thr)))


def fit_sigmoid(levels, corrects):
    levels = np.asarray(levels, float)
    corrects = np.asarray(corrects, float)

    def nll(p):
        thr, slope = p
        pr = np.clip(sigmoid(levels, thr, slope), 1e-9, 1 - 1e-9)
        return -np.sum(corrects * np.log(pr) + (1 - corrects) * np.log(1 - pr))

    best = None
    for thr0 in (2.0, 4.0, 6.0):
        for sl0 in (0.5, 1.5):
            r = optimize.minimize(nll, [thr0, sl0], method="Nelder-Mead",
                                  options={"xatol": 1e-4, "fatol": 1e-6,
                                           "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    thr, slope = best.x
    return {"threshold": float(np.clip(thr, -5, 15)), "slope": float(slope),
            "nll": float(best.fun)}


def bootstrap_threshold(df, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    motifs = df["motif"].unique()
    groups = {m: df[df["motif"] == m] for m in motifs}
    thrs = []
    for _ in range(n_boot):
        pick = rng.choice(motifs, size=len(motifs), replace=True)
        sub = pd.concat([groups[m] for m in pick])
        f = fit_sigmoid(sub["level"], sub["correct"])
        thrs.append(f["threshold"])
    return [float(np.percentile(thrs, 2.5)), float(np.percentile(thrs, 97.5))]


# ---------------------------------------------------------------------------

def load():
    rows = []
    for f in sorted((EXP / "results").glob("*.json")):
        d = json.loads(f.read_text())
        for r in d["records"]:
            if r.get("pred") is None:
                continue
            rows.append({"model": d["model"], "motif": r["answer"],
                         "level": r["level"], "seed": r["seed"],
                         "correct": int(r["correct"])})
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        print("no results yet in exp3_ladder/results/")
        return
    models = sorted(df["model"].unique())
    print(f"{len(df)} records, models: {models}")

    out = {"n_records": len(df), "models": {}, "per_level": {}}

    # per-level accuracies
    for m in models:
        acc = df[df.model == m].groupby("level")["correct"].agg(["mean", "count"])
        out["per_level"][m] = {f"L{int(l)}": {"acc": round(r["mean"], 4),
                                              "n": int(r["count"])}
                               for l, r in acc.iterrows()}

    # per-model GLM (motif fixed intercepts) + sigmoid + bootstrap CI
    for m in models:
        sub = df[df.model == m]
        res = {"n": len(sub)}
        try:
            glm = smf.glm("correct ~ level + C(motif)", data=sub,
                          family=sm.families.Binomial()).fit()
            res["glm_level_coef"] = round(float(glm.params["level"]), 4)
            res["glm_level_se"] = round(float(glm.bse["level"]), 4)
            res["glm_level_p"] = float(glm.pvalues["level"])
        except Exception as e:
            res["glm_error"] = str(e)
        fit = fit_sigmoid(sub["level"], sub["correct"])
        res["sigmoid_threshold"] = round(fit["threshold"], 3)
        res["sigmoid_slope"] = round(fit["slope"], 3)
        res["threshold_ci95"] = [round(x, 3) for x in
                                 bootstrap_threshold(sub)]
        out["models"][m] = res
        print(m, res)

    # LRT: slope x model interaction (shared-slope law)
    if len(models) >= 2:
        full = smf.glm("correct ~ level * C(model) + C(motif)", data=df,
                       family=sm.families.Binomial()).fit()
        red = smf.glm("correct ~ level + C(model) + C(motif)", data=df,
                      family=sm.families.Binomial()).fit()
        lr = 2 * (full.llf - red.llf)
        dof = int(full.df_model - red.df_model)
        p = float(stats.chi2.sf(lr, dof))
        out["slope_equality_LRT"] = {"lr_chi2": round(float(lr), 3),
                                     "df": dof, "p": p,
                                     "shared_slope_supported": p > 0.05}
        print("slope-equality LRT:", out["slope_equality_LRT"])

    # per-motif threshold (pooled models) vs confusable-family degree
    per_motif = []
    for mid, sub in df.groupby("motif"):
        f = fit_sigmoid(sub["level"], sub["correct"])
        per_motif.append({"motif": mid, "threshold": round(f["threshold"], 3),
                          "slope": round(f["slope"], 3),
                          "family_degree": family_degree(mid),
                          "overall_acc": round(sub["correct"].mean(), 4)})
    out["per_motif"] = per_motif
    pm = pd.DataFrame(per_motif)
    ok = pm[(pm.threshold > -5) & (pm.threshold < 15)]
    if len(ok) > 3:
        rho, p = stats.spearmanr(ok["threshold"], ok["family_degree"])
        out["threshold_vs_family_degree"] = {"spearman_rho": round(float(rho), 3),
                                             "p": float(p), "n_motifs": len(ok)}
        print("threshold vs family degree:", out["threshold_vs_family_degree"])

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xs = np.linspace(-0.2, 6.2, 300)
    for i, m in enumerate(models):
        c = MODEL_COLORS[i % len(MODEL_COLORS)]
        sub = df[df.model == m]
        acc = sub.groupby("level")["correct"].agg(["mean", "count"])
        se = np.sqrt(acc["mean"] * (1 - acc["mean"]) / acc["count"])
        ax.errorbar(acc.index, acc["mean"], yerr=se, fmt="o", color=c,
                    ms=4.5, capsize=2, lw=1, zorder=3)
        r = out["models"][m]
        thr, sl = r["sigmoid_threshold"], r["sigmoid_slope"]
        ax.plot(xs, sigmoid(xs, thr, sl), color=c, lw=1.8, zorder=2,
                label=f"{m}  (θ={thr:.1f}, β={sl:.1f})")
        if -0.2 < thr < 6.2:
            half = GUESS + (1 - GUESS) / 2
            ax.plot([thr, thr], [0, half], color=c, lw=0.9, ls=":", zorder=1)
            lo, hi = r["threshold_ci95"]
            ax.plot([max(lo, -0.2), min(hi, 6.2)], [0.015, 0.015], color=c,
                    lw=2.5, alpha=0.55, solid_capstyle="butt")
    ax.axhline(GUESS, color=MUTED, lw=0.8, ls="--", alpha=0.7)
    ax.text(6.15, GUESS + 0.015, "chance (1/25)", ha="right", fontsize=7.5,
            color=MUTED)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels(["L0\nskeleton", "L1\nsketch", "L2\nstylized",
                        "L3\nminimal", "L4\nscene", "L5\nnatural",
                        "L6\ncluttered"], fontsize=7.5)
    ax.set_xlabel("naturalization level")
    ax.set_ylabel("scene→motif accuracy (25-way)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.35, 6.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "psychophysics.pdf", bbox_inches="tight")
    fig.savefig(EXP / "psychophysics.png", dpi=200, bbox_inches="tight")
    print("figure ->", FIGS / "psychophysics.pdf")

    (EXP / "results.json").write_text(json.dumps(out, indent=1,
                                                 ensure_ascii=False))
    print("summary ->", EXP / "results.json")


if __name__ == "__main__":
    main()
