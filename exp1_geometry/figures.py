#!/usr/bin/env python3
"""Exp1 — paper figures (matplotlib, white bg, terracotta accent).

  (a) exp1_confusion_vs_distance.pdf : pooled dyadic confusion rate vs formal
      GED and vs perceptual DINOv2 distance, partial-r annotated.
  (b) exp1_betas.pdf : per-model beta_formal vs beta_perceptual grouped bars
      with permutation-p stars.

PDFs -> paper/figs/
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import sys
sys.path.insert(0, "/Users/stone/GEB-benchmark")
import figstyle
figstyle.use()

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "exp1_geometry"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(exist_ok=True)

TERRA = figstyle.ACCENT      # formal / highlight accent
INK = figstyle.NEUTRAL       # drawn means / reference lines / stars
GREY = "#9a9a9a"             # de-emphasized point clouds

# display names for model tags (tick labels only; order untouched)
TAG2NAME = {
    "openai/gpt-5.5": "GPT-5.5",
    "openai/gpt-5.2": "GPT-5.2",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "moonshotai/kimi-k2.5": "Kimi K2.5",
    "qwen/qwen2.5-vl-72b-instruct": "Qwen2.5-VL-72B",
    "qwen2.5-vl-32b": "Qwen2.5-VL-32B",
    "qwen2-vl-7b": "Qwen2-VL-7B",
}

VISION_TASKS = ("scene2motif", "skeleton2motif")


def load():
    F = json.loads((EXP / "matrices_formal.json").read_text())
    motifs = F["motifs"]
    idx = {m: i for i, m in enumerate(motifs)}
    n = len(motifs)
    z = np.load(EXP / "embeddings.npz", allow_pickle=True)
    paths = [str(p) for p in z["paths"]]
    E = z["dino"] / np.linalg.norm(z["dino"], axis=1, keepdims=True)
    cent = np.zeros((n, E.shape[1]))
    for i, m in enumerate(motifs):
        rows = [k for k, p in enumerate(paths) if p.split("/")[0] == m]
        c = E[rows].mean(0)
        cent[i] = c / np.linalg.norm(c)
    dino = 1.0 - cent @ cent.T
    C = np.zeros((n, n))
    expo = np.zeros(n)
    per_model = {}
    for f in (sorted(glob.glob(str(ROOT / "eval/results/*.json")))
              + sorted(glob.glob(str(EXP / "results/*.json")))):
        d = json.loads(Path(f).read_text())
        if "records" not in d:
            continue
        recs = [r for r in d["records"] if r["task"] in VISION_TASKS]
        if not recs:
            continue
        Cm = np.zeros((n, n))
        Em = np.zeros(n)
        for r in recs:
            Em[idx[r["answer"]]] += 1
            if not r["correct"] and r.get("pred") in idx \
               and r["pred"] != r["answer"]:
                Cm[idx[r["answer"]], idx[r["pred"]]] += 1
        if d["model"] in per_model:
            C0, E0 = per_model[d["model"]]
            per_model[d["model"]] = (C0 + Cm, E0 + Em)
        else:
            per_model[d["model"]] = (Cm, Em)
        C += Cm
        expo += Em
    return (motifs, np.array(F["formal_family"]), np.array(F["formal_ged"]),
            np.array(F["label_prior"]), dino, C, expo, per_model)


def partial_r(y, x, controls):
    def resid(v, Z):
        Z = np.column_stack([np.ones(len(v))] + list(Z))
        return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    ry = resid(y, controls)
    rx = resid(x, controls)
    return float(np.corrcoef(ry, rx)[0, 1])


def fig_scatter(motifs, fam, ged, lab, dino, C, expo):
    n = len(motifs)
    mask = ~np.eye(n, dtype=bool)
    rate = (C / np.maximum(expo[:, None], 1))[mask]
    g, d, f, l = ged[mask], dino[mask], (1 - fam)[mask], lab[mask]
    pr_f = partial_r(rate, f, [-d, -l])
    pr_d = partial_r(rate, -d, [f, -l])
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.3), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.4]})
    # left: primary formal predictor (binary same-family) as strip + means
    ax = axes[0]
    for val, lbl, col in ((1, "same\nfamily", TERRA),
                          (0, "different\nfamily", GREY)):
        s = f == val
        x0 = 1 - val
        ax.scatter(x0 + rng.uniform(-.13, .13, s.sum()), rate[s], s=10,
                   alpha=0.45, color=col, edgecolors="none")
        ax.hlines(rate[s].mean(), x0 - 0.22, x0 + 0.22, color=INK, lw=1.6,
                  zorder=5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["same\nfamily", "different\nfamily"])
    ax.set_xlabel("formal distance (confusable family)")
    ratio = rate[f == 1].mean() / max(rate[f == 0].mean(), 1e-9)
    ax.annotate(f"mean rate ×{ratio:.1f}\npartial $r$ = {pr_f:+.2f}",
                xy=(0.96, 0.86), xycoords="axes fraction", ha="right",
                fontweight="bold", color=TERRA)
    # right: perceptual DINOv2 distance scatter + binned means
    ax = axes[1]
    ax.scatter(d, rate, s=10, alpha=0.45, color=GREY, edgecolors="none")
    qs = np.quantile(d, np.linspace(0, 1, 7))
    for lo, hi in zip(qs[:-1], qs[1:]):
        s = (d >= lo) & (d <= hi)
        if s.sum():
            ax.plot([(lo + hi) / 2], [rate[s].mean()], "o", color=INK,
                    ms=5, zorder=5)
    ax.set_xlabel("perceptual distance (DINOv2 centroid cosine)")
    ax.annotate(f"partial $r$ = {pr_d:+.2f}", xy=(0.96, 0.92),
                xycoords="axes fraction", ha="right",
                color=INK)
    axes[0].set_ylabel("pairwise confusion rate")
    fig.suptitle("Pooled VLM confusions vs formal and perceptual distance",
                 fontsize=matplotlib.rcParams["axes.titlesize"], y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "exp1_confusion_vs_distance.pdf", bbox_inches="tight")
    print("wrote", FIGS / "exp1_confusion_vs_distance.pdf")


def fig_betas():
    R = json.loads((EXP / "results.json").read_text())["regression"]
    models = [m for m in R if m != "POOLED"] + ["POOLED"]
    keys = ["formal_family", "formal_ged", "perceptual_dino", "label_prior"]
    labels = ["formal: same family", "formal: GED",
              "perceptual: DINOv2", "label prior"]
    colors = [TERRA, "#DBA07A", GREY, "#cfcfcf"]
    x = np.arange(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(figstyle.FULL, 2.6))
    for k, (key, lb, c) in enumerate(zip(keys, labels, colors)):
        vals = [R[m]["beta"][key] for m in models]
        ps = [R[m]["p_perm"][key] for m in models]
        ax.bar(x + (k - 1.5) * w, vals, w, color=c, label=lb,
               edgecolor="white", linewidth=0.5)
        for xi, v, p in zip(x + (k - 1.5) * w, vals, ps):
            star = ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.05 else "")
            if star:
                ax.text(xi, v + (0.03 if v >= 0 else -0.10), star,
                        ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    tick_labels = [TAG2NAME.get(m, m) for m in models]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(r"dyadic Poisson $\beta$ (std. similarity)")
    ax.legend(ncol=2)
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(FIGS / "exp1_betas.pdf")
    print("wrote", FIGS / "exp1_betas.pdf")


if __name__ == "__main__":
    motifs, fam, ged, lab, dino, C, expo, _ = load()
    fig_scatter(motifs, fam, ged, lab, dino, C, expo)
    fig_betas()
