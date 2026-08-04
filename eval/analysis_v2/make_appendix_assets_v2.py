#!/usr/bin/env python3
"""V2 appendix assets: verbatim EN menu table + appendix figures
(hardest_motifs, confusion_pairs, task_breakdown, per_motif_heatmap, scaling)
regenerated from eval/results_v2 with the shared ICLR figstyle."""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import sys; sys.path.insert(0, "/Users/stone/GEB-benchmark")
import figstyle; figstyle.use()
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"
PAPER = ROOT / "paper"

LIB = json.loads((ROOT / "data" / "motif_library.json").read_text())
MOTIFS = [m["id"] for m in LIB]
LIB_MAP = {m["id"]: m for m in LIB}

MODELS = ["or_gpt-5.5", "or_gpt-5.2", "or_claude-opus-4.8",
          "or_gemini-3.1-pro", "or_kimi-k2.5", "or_qwen2.5-vl-72b",
          "or_deepseek-v3.2",
          "qwen2.5-vl-32b", "deepseek-coder-v2-lite",
          "qwen2.5-14b", "qwen2.5-7b", "qwen2-vl-7b"]
LABELS = ["GPT-5.5", "GPT-5.2", "Claude Opus 4.8",
          "Gemini 3.1 Pro", "Kimi K2.5", "Qwen2.5-VL-72B",
          "DeepSeek-V3.2",
          "Qwen2.5-VL-32B", "DeepSeek-Coder-V2-Lite",
          "Qwen2.5-14B", "Qwen2.5-7B", "Qwen2-VL-7B"]
ID_TASKS = ["scene2motif", "skeleton2motif", "story2motif", "theorem2motif"]

# colors come from the shared figstyle palette (aesthetics only)
ACCENT = figstyle.ACCENT
MUTED = figstyle.NEUTRAL
INK = "#222222"


def motif_name(mid):
    return LIB_MAP[mid]["name_en"].split("—")[0].strip()


def load(tag):
    return json.loads((EVAL / "results_v2" / f"{tag}.json").read_text())


def save(fig, stem):
    for ext, dpi in (("pdf", 300), ("png", 150)):
        fig.savefig(PAPER / "figs" / f"{stem}.{ext}", dpi=dpi,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf/png")


# ---------------------------------------------------------------- menu table
def tex_escape(s):
    s = s.replace("\\", r"\textbackslash{}")
    for c in "&%#$_{}":
        s = s.replace(c, "\\" + c)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    s = s.replace("—", "---").replace("–", "--")
    s = s.replace("<", "$<$").replace(">", "$>$")
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    return s


def menu_table():
    menu = json.loads((EVAL / "items_v2.json").read_text())["menu"]
    rows = []
    for line in menu.strip().split("\n"):
        m = re.match(r"^- ([a-z_0-9]+): (.*)$", line)
        assert m, line
        rows.append((m.group(1), m.group(2)))
    assert len(rows) == 25
    out = []
    out.append(r"\begin{small}")
    out.append(r"\begin{longtable}{@{}p{0.22\linewidth}p{0.74\linewidth}@{}}")
    out.append(r"\caption{The 25-way answer menu, verbatim as administered to"
               r" models (\texttt{data/library\_en.json}).}"
               r"\label{tab:menu}\\")
    out.append(r"\toprule")
    out.append(r"Motif & Invariant \\")
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append(r"\toprule")
    out.append(r"Motif & Invariant \\")
    out.append(r"\midrule")
    out.append(r"\endhead")
    out.append(r"\bottomrule")
    out.append(r"\endfoot")
    for mid, inv in rows:
        tid = r"\texttt{%s}" % mid.replace("_", r"\_\allowbreak ")
        out.append(f"{tid} & {tex_escape(inv)} \\\\")
    out.append(r"\end{longtable}")
    out.append(r"\end{small}")
    (PAPER / "tables" / "menu_en.tex").write_text("\n".join(out) + "\n")
    print("  tables/menu_en.tex (%d rows)" % len(rows))


# ------------------------------------------------------------- id-task pools
def per_motif_matrix():
    """model x motif recall over the four 25-way identification tasks."""
    mat = np.full((len(MODELS), len(MOTIFS)), np.nan)
    pooled = defaultdict(list)
    for i, tag in enumerate(MODELS):
        recs = [r for r in load(tag)["records"] if r["task"] in ID_TASKS]
        bym = defaultdict(list)
        for r in recs:
            if r["answer"] in LIB_MAP:
                bym[r["answer"]].append(bool(r["correct"]))
                pooled[r["answer"]].append(bool(r["correct"]))
        for j, mid in enumerate(MOTIFS):
            if bym[mid]:
                mat[i, j] = np.mean(bym[mid])
    recall = {m: float(np.mean(v)) for m, v in pooled.items()}
    return mat, recall


def hardest_fig(recall):
    ranked = sorted(recall.items(), key=lambda kv: kv[1])[:10]
    fig, ax = plt.subplots(figsize=(figstyle.FULL, 3.4))
    ids = [m for m, _ in ranked]
    accs = [a * 100 for _, a in ranked]
    ax.barh(range(len(ids)), accs, color=ACCENT,
            edgecolor="white", linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(a + 1, i, f"{a:.1f}%", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([motif_name(m) for m in ids])
    ax.set_xlabel("Pooled recall (%), all models, four identification tasks")
    ax.set_title("Ten Hardest Motifs", pad=10)
    ax.set_xlim(0, 100)
    ax.axvline(4.0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.5,
               label="Random (4%)")
    ax.legend(loc="lower right")
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "hardest_motifs")
    return ranked


def heatmap_fig(mat):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "warmdiv", ["#B3402A", "#FFFFFF", "#3E7DA6"])
    fig, ax = plt.subplots(figsize=(figstyle.FULL, 3.0))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                c = "white" if v < 0.3 or v > 0.7 else INK
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=4.2, color=c)
    ax.set_xticks(range(len(MOTIFS)))
    ax.set_xticklabels([motif_name(m) for m in MOTIFS], rotation=45,
                       ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(LABELS, fontsize=6.5)
    ax.set_xlabel("Motif (25 total)", labelpad=4)
    ax.set_ylabel("Model")
    ax.set_title("Per-Motif Recall Across Models "
                 "(pooled over the four identification tasks)",
                 pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Recall", rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    save(fig, "per_motif_heatmap")


def confusion_pairs_fig():
    conf = json.loads(
        (EVAL / "analysis_v2" / "confusion_v2.json").read_text())
    top = conf["top_confusions"][:10]
    labels = [f"{c['true'].replace('_', ' ')}  →  "
              f"{c['pred'].replace('_', ' ')}" for c in top][::-1]
    ns = [c["count"] for c in top][::-1]
    fig, ax = plt.subplots(figsize=(figstyle.FULL, 2.9))
    ax.barh(range(len(ns)), ns, color=ACCENT, height=0.62,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("pooled errors (all models, four identification tasks)")
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    fig.tight_layout()
    save(fig, "confusion_pairs")
    return top


def task_breakdown_fig():
    nums = json.loads(
        (EVAL / "analysis_v2" / "main_table_numbers.json").read_text())
    tasks = ["scene2motif", "skeleton2motif", "story2motif", "theorem2motif",
             "crossvoice", "xv_scene_thm", "xv_story_thm",
             "scene2motif_family4"]
    tlabels = ["Scene", "Skel", "Story", "Thm", "Cross", "XvSc", "XvSt",
               "Fam4"]
    colors = ["#8B4513", "#C4612F", "#D4A373", "#6B8E23",
              "#556B2F", "#4E6E8E", "#8E7CC3", "#B0503C"]
    data = np.full((len(MODELS), len(tasks)), np.nan)
    for i, tag in enumerate(MODELS):
        for j, t in enumerate(tasks):
            if t in nums[tag]["accs"]:
                data[i, j] = nums[tag]["accs"][t]
    fig, ax = plt.subplots(figsize=(figstyle.FULL, 3.0))
    x = np.arange(len(MODELS))
    width = 0.105
    for j, tl in enumerate(tlabels):
        off = (j - len(tasks) / 2 + 0.5) * width
        ax.bar(x + off, data[:, j], width, label=tl, color=colors[j],
               edgecolor="white", linewidth=0.4)
    ax.axhline(4.0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.6,
               label="Chance (25-way)")
    ax.axhline(25.0, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.6,
               label="Chance (4-way)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Task Accuracy Breakdown", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "task_breakdown")


def scaling_fig():
    nums = json.loads(
        (EVAL / "analysis_v2" / "main_table_numbers.json").read_text())
    fam = ["qwen2.5-7b", "qwen2.5-14b", "qwen2.5-vl-32b"]
    sizes = [7, 14, 32]
    thm = [nums[t]["accs"]["theorem2motif"] / 100 for t in fam]
    sto = [nums[t]["accs"]["story2motif"] / 100 for t in fam]
    STORY_BLUE = "#3E7DA6"  # palette-consistent condition color (task, not model)
    fig, ax = plt.subplots(figsize=(figstyle.HALF, 2.0))
    ax.plot(sizes, thm, "o-", color=ACCENT, label="theorem")
    ax.plot(sizes, sto, "s-", color=STORY_BLUE, label="story")
    g55t = nums["or_gpt-5.5"]["accs"]["theorem2motif"] / 100
    g55s = nums["or_gpt-5.5"]["accs"]["story2motif"] / 100
    ax.axhline(g55t, color=ACCENT, lw=1, ls="--", alpha=0.55)
    ax.axhline(g55s, color=STORY_BLUE, lw=1, ls="--", alpha=0.55)
    ax.text(32.7, g55t + 0.015, "GPT-5.5", color=ACCENT, fontsize=7,
            va="bottom")
    ax.text(32.7, g55s - 0.015, "GPT-5.5", color=STORY_BLUE, fontsize=7,
            va="top")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["7B", "14B", "32B (VL)"])
    ax.minorticks_off()
    ax.set_xlabel("Qwen2.5 parameters")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(6, 40)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "scaling")


if __name__ == "__main__":
    print("menu table:")
    menu_table()
    print("figures:")
    mat, recall = per_motif_matrix()
    ranked = hardest_fig(recall)
    heatmap_fig(mat)
    top = confusion_pairs_fig()
    task_breakdown_fig()
    scaling_fig()
    print("bottom-10 recall:")
    for m, a in ranked:
        print(f"  {m:28s} {a:.3f}")
    (EVAL / "analysis_v2" / "appendix_assets_v2.json").write_text(json.dumps({
        "bottom10_recall": {m: a for m, a in ranked},
        "top_confusions": top,
    }, indent=1))
