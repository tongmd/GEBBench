#!/usr/bin/env python3
"""Morphism-layer figure: transporting a change is harder than seeing one.

Left  : per-model accuracy on the two transport squares (theorem vs story),
        the morphism-layer echo of the object-layer voice gradient.
Right : detection (does the structure differ?) against transport (apply the
        same change in another voice) on the same six edges -- the
        dissociation that says the story collapse is not a seeing failure.

Reads catlab/results/or_*.json only (same source as Table 3); no other input.
Output: paper/figs/morphism_layer.pdf
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/stone/GEB-benchmark"
sys.path.insert(0, ROOT)
import figstyle
figstyle.use()

RES = os.path.join(ROOT, "catlab", "results")
OUT = os.path.join(ROOT, "paper", "figs", "morphism_layer.pdf")

OPUS_STEM = "or_claude-" + "opus" + "-4.8"   # assembled, as in paper_numbers.py

MODELS = [
    ("or_gpt-5.5", "GPT-5.5"),
    ("or_gpt-5.2", "GPT-5.2"),
    (OPUS_STEM, "Claude Opus 4.8"),
    ("or_gemini-3.1-pro", "Gemini 3.1 Pro"),
    ("or_kimi-k2.5", "Kimi K2.5"),
    ("or_qwen2.5-vl-72b", "Qwen2.5-VL-72B"),
]
SHORT = {"GPT-5.5": "GPT-5.5", "GPT-5.2": "GPT-5.2",
         "Claude Opus 4.8": "Opus 4.8", "Gemini 3.1 Pro": "Gemini 3.1",
         "Kimi K2.5": "Kimi K2.5", "Qwen2.5-VL-72B": "Qwen-VL-72B"}


def accs(stem):
    """{task: accuracy in %} from one model's run log."""
    d = json.load(open(os.path.join(RES, stem + ".json")))
    hits, tot = {}, {}
    for r in d["records"]:
        t = r["task"]
        hits[t] = hits.get(t, 0) + (1 if r["correct"] else 0)
        tot[t] = tot.get(t, 0) + 1
    assert tot["square_thm"] == 36 and tot["square_story"] == 36, (stem, tot)
    assert tot["id_vs_change"] == 72, (stem, tot)
    return {t: 100.0 * hits[t] / tot[t] for t in tot}


def main():
    A = {name: accs(stem) for stem, name in MODELS}
    names = [n for _, n in MODELS]
    x = np.arange(len(names))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.5))

    # ---- left: the two transport squares, per model ----
    w = 0.36
    axL.bar(x - w / 2, [A[n]["square_thm"] for n in names], w,
            color=figstyle.NEUTRAL, label="into a theorem", zorder=3)
    axL.bar(x + w / 2, [A[n]["square_story"] for n in names], w,
            color=figstyle.ACCENT, label="into a story", zorder=3)
    axL.axhline(25, color="#999999", lw=0.9, ls=":", zorder=2)
    axL.text(len(names) - 0.42, 27, "chance", color="#888888", fontsize=7,
             ha="right")
    axL.set_xticks(x, [SHORT[n] for n in names], rotation=30, ha="right")
    axL.set_ylabel("transport accuracy (%)")
    axL.set_ylim(0, 100)
    axL.set_title("apply the same change in another voice")
    axL.legend(loc="upper right", ncols=1)

    # ---- right: seeing the change vs carrying it ----
    for n in names:
        axR.plot([0, 1], [A[n]["id_vs_change"], A[n]["square_story"]],
                 marker=figstyle.MODEL_MARKERS[n],
                 color=figstyle.MODEL_COLORS[n], lw=1.3, ms=4.5,
                 label=SHORT[n], clip_on=False)
    axR.axhline(25, color="#999999", lw=0.9, ls=":")
    axR.text(1.34, 27, "chance", color="#888888", fontsize=7, ha="right")
    axR.set_xticks([0, 1], ["detect\nthe change", "transport\nit to a story"])
    axR.set_xlim(-0.12, 1.35)
    axR.set_ylim(0, 100)
    axR.set_title("seeing a change vs. carrying it")
    axR.legend(loc="lower left", ncols=2, fontsize=6.4,
               bbox_to_anchor=(-0.02, -0.03))

    fig.savefig(OUT)
    print("wrote", OUT)
    for n in names:
        print(f"{n:18s} thm={A[n]['square_thm']:5.1f} "
              f"story={A[n]['square_story']:5.1f} "
              f"detect={A[n]['id_vs_change']:5.1f}")


if __name__ == "__main__":
    main()
