#!/usr/bin/env python3
"""Figure: standardized dyadic-regression betas per predictor, one panel of
grouped bars per perceptual-embedding baseline.

For each of DINOv2 / CLIP-B32 / SigLIP we show the 3-predictor race
(formal family + label prior + that perceptual similarity) on pooled
confusions; this avoids the CLIP-SigLIP collinearity (Mantel r=0.83) that
makes the joint 5-way race uninterpretable. Stars: permutation p<0.05,
dagger: p<0.10. Writes eval/analysis_out/perceptual_race.pdf/.png
"""
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/Users/stone/GEB-benchmark/eval/analysis_out")
R = json.loads((OUT / "perceptual_race.json").read_text())

INK = "#1F2421"
plt.rcParams.update({
    "font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

EMBS = ["dinov2", "clip_b32", "siglip"]
EMB_LABEL = {"dinov2": "DINOv2", "clip_b32": "CLIP ViT-B/32",
             "siglip": "SigLIP"}
BAR_KEYS = ["formal_family", "label_prior", "PERC"]
BAR_LABEL = {"formal_family": "formal family", "label_prior": "label prior",
             "PERC": "perceptual"}
COLOR = {"formal_family": "#C4612F", "label_prior": "#D9A066",
         "PERC": "#5C7A99"}


def mark(p):
    return "*" if p < 0.05 else ("$\\dagger$" if p < 0.10 else "")


races = R["race_one_perceptual_at_a_time"]

fig, ax = plt.subplots(figsize=(6.0, 3.2))
width = 0.24
xs = np.arange(len(EMBS))
for bi, key in enumerate(BAR_KEYS):
    vals, ps = [], []
    for e in EMBS:
        k = e if key == "PERC" else key
        vals.append(races[e]["beta"][k])
        ps.append(races[e]["p_perm"][k])
    pos = xs + (bi - 1) * width
    ax.bar(pos, vals, width=width * 0.92, color=COLOR[key], zorder=3,
           label=BAR_LABEL[key])
    for x, v, p in zip(pos, vals, ps):
        ax.text(x, v + (0.012 if v >= 0 else -0.012),
                f"{v:+.2f}{mark(p)}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, zorder=4)

ax.axhline(0, color="#999999", lw=0.8, zorder=2)
ax.set_xticks(xs)
ax.set_xticklabels([EMB_LABEL[e] for e in EMBS])
ax.set_xlabel("perceptual baseline entered in the race")
ax.set_ylabel(r"standardized dyadic $\beta$")
ax.set_title("Dyadic regression of pooled confusions:\n"
             "formal family + label prior + one perceptual geometry",
             fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
ax.set_ylim(min(-0.05, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.18)

fig.tight_layout()
fig.savefig(OUT / "perceptual_race.pdf")
fig.savefig(OUT / "perceptual_race.png", dpi=200)
print("wrote", OUT / "perceptual_race.pdf", "and .png")
