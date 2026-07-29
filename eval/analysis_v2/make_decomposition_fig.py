#!/usr/bin/env python3
"""Decomposition slope figure: the same 210 scenes asked four ways.

x = task (25-way id, family-of-4 id, scene->theorem, scene->story),
y = accuracy, one line per vision model. Reads eval/results_v2/*.json
(the exact source of paper/tables/main_results.tex); no other input.
Output: paper/figs/decomposition_v2.pdf
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/stone/GEB-benchmark"
sys.path.insert(0, ROOT)
import figstyle
figstyle.use()

RES = os.path.join(ROOT, "eval", "results_v2")
OUT = os.path.join(ROOT, "paper", "figs", "decomposition_v2.pdf")

MODELS = [  # vision-capable eight, frontier first
    ("or_gpt-5.5", "GPT-5.5"),
    ("or_gpt-5.2", "GPT-5.2"),
    ("or_claude-opus-4.8", "Claude Opus 4.8"),
    ("or_gemini-3.1-pro", "Gemini 3.1 Pro"),
    ("or_kimi-k2.5", "Kimi K2.5"),
    ("or_qwen2.5-vl-72b", "Qwen2.5-VL-72B"),
    ("qwen2.5-vl-32b", "Qwen2.5-VL-32B"),
    ("qwen2-vl-7b", "Qwen2-VL-7B"),
]
TASKS = ["scene2motif", "scene2motif_family4", "xv_scene_thm", "crossvoice"]
XLABELS = ["identify\n(25-way menu)", "identify\n(family of 4)",
           "match theorem\n(4-way)", "match story\n(4-way)"]


def acc(stem, task):
    d = json.load(open(os.path.join(RES, stem + ".json")))
    rec = [r for r in d["records"] if r["task"] == task]
    assert len(rec) == 210, (stem, task, len(rec))
    return 100.0 * sum(r["correct"] for r in rec) / len(rec)


def main():
    fig, ax = plt.subplots(figsize=(figstyle.FULL, 2.9))
    xs = range(len(TASKS))
    for stem, name in MODELS:
        ys = [acc(stem, t) for t in TASKS]
        ax.plot(xs, ys, marker=figstyle.MODEL_MARKERS[name],
                color=figstyle.MODEL_COLORS[name], label=name,
                lw=1.3, ms=4, alpha=0.9, clip_on=False)
    # chance floors: 4% under the 25-way column, 25% under the 4-way columns
    ax.plot([-0.15, 0.15], [4, 4], color="#999999", lw=0.9, ls=":")
    ax.plot([0.85, 3.15], [25, 25], color="#999999", lw=0.9, ls=":")
    ax.text(3.18, 25, "chance", color="#888888", fontsize=7, va="center")
    ax.text(0.18, 4, "chance", color="#888888", fontsize=7, va="center")
    ax.set_xticks(list(xs), XLABELS)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.25, 3.6)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
