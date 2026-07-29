#!/usr/bin/env python3
"""Recognition-vs-mapping decomposition on v2 results.

Three analyses:
  1. Conditional decomposition (vision models): P(crossvoice correct | scene2motif
     correct on the same scene) vs P(correct | scene wrong); same for xv_scene_thm.
  2. Mapping-vs-recognition gap at matched arity: scene2motif_family4 (4-way,
     within-family recognition) minus crossvoice (4-way, scene->story mapping)
     on the same 210 scenes; accuracy and kappa gaps, paired bootstrap + McNemar.
  3. Scale-helps-recognition-not-mapping: Qwen2.5 text family 7B->14B on
     story2motif vs xv_story_thm; per-task McNemar between sizes + bootstrap CI
     on the interaction (delta_story - delta_xv).

Outputs:
  eval/analysis_v2/mapping_v2.json
  eval/analysis_v2/mapping_gap_v2.pdf / .png  (also copied to paper/figs/)
"""
import json
import os
import shutil
from collections import defaultdict

import numpy as np
from scipy import stats as sps

ROOT = "/Users/stone/GEB-benchmark"
RES = os.path.join(ROOT, "eval", "results_v2")
OUT = os.path.join(ROOT, "eval", "analysis_v2")
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(20260721)
N_BOOT = 10000

VISION_MODELS = ["or_gpt-5.5", "or_gpt-5.2", "or_claude-opus-4.8",
                 "or_gemini-3.1-pro", "or_kimi-k2.5", "or_qwen2.5-vl-72b",
                 "qwen2.5-vl-32b", "qwen2-vl-7b"]
TEXT_PANEL = ["or_gpt-5.5", "or_gpt-5.2", "or_claude-opus-4.8",
              "or_gemini-3.1-pro", "or_deepseek-v3.2",
              "qwen2.5-14b", "qwen2.5-7b",
              "deepseek-coder-v2-lite"]
LABELS = {
    "or_gpt-5.5": "GPT-5.5", "or_gpt-5.2": "GPT-5.2",
    "or_claude-opus-4.8": "Claude Opus 4.8",
    "or_gemini-3.1-pro": "Gemini 3.1 Pro", "or_kimi-k2.5": "Kimi K2.5",
    "or_qwen2.5-vl-72b": "Qwen2.5-VL-72B", "or_deepseek-v3.2": "DeepSeek-V3.2",
    "qwen2.5-vl-32b": "Qwen2.5-VL-32B", "qwen2-vl-7b": "Qwen2-VL-7B",
    "qwen2.5-14b": "Qwen2.5-14B", "qwen2.5-7b": "Qwen2.5-7B",
    "deepseek-coder-v2-lite": "DeepSeek-Coder-V2-Lite",
}
CHANCE = {"scene2motif": 1/25, "story2motif": 1/25,
          "crossvoice": 1/4, "xv_scene_thm": 1/4, "xv_story_thm": 1/4,
          "scene2motif_family4": 1/4}


def kappa(acc, chance):
    return (acc - chance) / (1.0 - chance)


def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def load_model(m):
    """Return task -> {key: 0/1}, key = item id with task prefix stripped."""
    path = os.path.join(RES, m + ".json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    by_task = defaultdict(dict)
    for r in d["records"]:
        key = r["id"].split(":", 1)[1]
        by_task[r["task"]][key] = 1 if r["correct"] else 0
    return by_task


DATA = {m: load_model(m) for m in set(VISION_MODELS + TEXT_PANEL)}
DATA = {m: v for m, v in DATA.items() if v is not None}
print("models loaded:", sorted(DATA))


# ---------------------------------------------------------------- analysis 1
def conditional_decomposition(model, map_task):
    d = DATA[model]
    sc, mp = d["scene2motif"], d[map_task]
    keys = sorted(set(sc) & set(mp))
    s = np.array([sc[k] for k in keys], dtype=float)
    m_ = np.array([mp[k] for k in keys], dtype=float)
    n = len(keys)
    assert n == 210, (model, map_task, n)

    def cond(sv, mv):
        pc = mv[sv == 1].mean() if (sv == 1).any() else np.nan
        pw = mv[sv == 0].mean() if (sv == 0).any() else np.nan
        return pc, pw

    p_c, p_w = cond(s, m_)
    # paired item bootstrap
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    S, M = s[idx], m_[idx]
    den_c, den_w = S.sum(1), (1 - S).sum(1)
    pc_b = np.where(den_c > 0, (M * S).sum(1) / np.maximum(den_c, 1), np.nan)
    pw_b = np.where(den_w > 0, (M * (1 - S)).sum(1) / np.maximum(den_w, 1), np.nan)
    diff_b = pc_b - pw_b
    return {
        "n": n,
        "n_scene_correct": int(s.sum()), "n_scene_wrong": int(n - s.sum()),
        "p_map_given_scene_correct": float(p_c),
        "p_map_given_scene_correct_ci95": ci95(pc_b),
        "p_map_given_scene_wrong": float(p_w),
        "p_map_given_scene_wrong_ci95": ci95(pw_b),
        "diff": float(p_c - p_w), "diff_ci95": ci95(diff_b),
        "p_boot_diff_le_0": float(np.nanmean(diff_b <= 0)),
    }


# ---------------------------------------------------------------- analysis 2
def matched_arity_gap(model):
    """family4 (recognition, 4-way) vs crossvoice (mapping, 4-way), same scenes."""
    d = DATA[model]
    f4, xv = d["scene2motif_family4"], d["crossvoice"]
    keys = sorted(set(f4) & set(xv))
    a = np.array([f4[k] for k in keys], dtype=float)   # recognition
    b = np.array([xv[k] for k in keys], dtype=float)   # mapping
    n = len(keys)
    assert n == 210, (model, n)
    acc_f4, acc_xv = a.mean(), b.mean()
    ch = 0.25
    k_f4, k_xv = kappa(acc_f4, ch), kappa(acc_xv, ch)
    # paired bootstrap on scenes
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    gap_b = a[idx].mean(1) - b[idx].mean(1)
    # exact McNemar (paired by scene)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    p_mcn = sps.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {
        "n": n,
        "acc_family4": float(acc_f4), "acc_crossvoice": float(acc_xv),
        "acc_gap": float(acc_f4 - acc_xv), "acc_gap_ci95": ci95(gap_b),
        "kappa_family4": float(k_f4), "kappa_crossvoice": float(k_xv),
        "kappa_gap": float(k_f4 - k_xv),
        "kappa_gap_ci95": [v / (1 - ch) for v in ci95(gap_b)],
        "mcnemar_n10_recogOnly": n10, "mcnemar_n01_mapOnly": n01,
        "mcnemar_p": float(p_mcn),
    }


# ---------------------------------------------------------------- analysis 3
def paired_model_delta(mA, mB, task):
    """acc(mB) - acc(mA) on the same items + McNemar across models."""
    a_d, b_d = DATA[mA][task], DATA[mB][task]
    keys = sorted(set(a_d) & set(b_d))
    a = np.array([a_d[k] for k in keys], dtype=float)
    b = np.array([b_d[k] for k in keys], dtype=float)
    n = len(keys)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    delta_b = b[idx].mean(1) - a[idx].mean(1)
    n10 = int(((b == 1) & (a == 0)).sum())  # B right, A wrong
    n01 = int(((b == 0) & (a == 1)).sum())
    p_mcn = sps.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {
        "n": n, "acc_" + mA: float(a.mean()), "acc_" + mB: float(b.mean()),
        "delta": float(b.mean() - a.mean()), "delta_ci95": ci95(delta_b),
        "mcnemar_B_only": n10, "mcnemar_A_only": n01, "mcnemar_p": float(p_mcn),
    }, delta_b


def scale_interaction(mA="qwen2.5-7b", mB="qwen2.5-14b"):
    story, story_b = paired_model_delta(mA, mB, "story2motif")
    xv, xv_b = paired_model_delta(mA, mB, "xv_story_thm")
    inter_b = story_b - xv_b  # independent item resamples across tasks
    return {
        "models": [mA, mB],
        "story2motif": story, "xv_story_thm": xv,
        "interaction_delta_story_minus_delta_xv": float(story["delta"] - xv["delta"]),
        "interaction_ci95": ci95(inter_b),
        "p_boot_interaction_le_0": float(np.mean(inter_b <= 0)),
    }


# ------------------------------------------------------------------- run all
results = {"n_boot": N_BOOT, "seed": 20260721,
           "conditional_decomposition": {}, "matched_arity_gap": {},
           "scale_interaction": None, "text_mapping_summary": {}}

for m in VISION_MODELS:
    if m not in DATA:
        continue
    results["conditional_decomposition"][m] = {
        "crossvoice": conditional_decomposition(m, "crossvoice"),
        "xv_scene_thm": conditional_decomposition(m, "xv_scene_thm"),
    }
    results["matched_arity_gap"][m] = matched_arity_gap(m)

results["scale_interaction"] = scale_interaction()

# text panel: story2motif vs xv_story_thm kappas per model
for m in TEXT_PANEL:
    if m not in DATA or "story2motif" not in DATA[m]:
        continue
    st = np.array(list(DATA[m]["story2motif"].values()), dtype=float)
    xv = np.array(list(DATA[m]["xv_story_thm"].values()), dtype=float)
    results["text_mapping_summary"][m] = {
        "acc_story2motif": float(st.mean()),
        "kappa_story2motif": float(kappa(st.mean(), 1/25)),
        "acc_xv_story_thm": float(xv.mean()),
        "kappa_xv_story_thm": float(kappa(xv.mean(), 1/4)),
        "kappa_gap": float(kappa(st.mean(), 1/25) - kappa(xv.mean(), 1/4)),
    }

with open(os.path.join(OUT, "mapping_v2.json"), "w") as f:
    json.dump(results, f, indent=2)
print("wrote", os.path.join(OUT, "mapping_v2.json"))


# -------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import sys; sys.path.insert(0, "/Users/stone/GEB-benchmark")
import figstyle; figstyle.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Two-condition encoding within each model row (recognition vs mapping):
# dark neutral vs the paper's terracotta accent, from the shared palette.
RECOG = figstyle.NEUTRAL
MAPPING = figstyle.ACCENT
CONNECT = "#D6D0C8"   # light neutral connector between the paired dots

fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.3), sharex=True)

# panel A: vision -- family4 (recognition) vs crossvoice (mapping), both 4-way
axA = axes[0]
vm = [m for m in VISION_MODELS if m in results["matched_arity_gap"]]
ys = np.arange(len(vm))[::-1]
for y, m in zip(ys, vm):
    g = results["matched_arity_gap"][m]
    axA.plot([g["kappa_crossvoice"], g["kappa_family4"]], [y, y],
             color=CONNECT, lw=1.8, zorder=1, solid_capstyle="round")
    axA.scatter([g["kappa_family4"]], [y], s=30, color=RECOG, zorder=3)
    axA.scatter([g["kappa_crossvoice"]], [y], s=30, color=MAPPING, zorder=3)
axA.set_yticks(ys)
axA.set_yticklabels([LABELS[m] for m in vm])
axA.set_title("Vision: identify vs. map (both 4-way)", loc="left")
axA.set_xlabel(r"chance-corrected accuracy $\kappa$")

# panel B: text -- story2motif (recognition) vs xv_story_thm (mapping)
axB = axes[1]
tm = [m for m in TEXT_PANEL if m in results["text_mapping_summary"]]
ys = np.arange(len(tm))[::-1]
for y, m in zip(ys, tm):
    g = results["text_mapping_summary"][m]
    axB.plot([g["kappa_xv_story_thm"], g["kappa_story2motif"]], [y, y],
             color=CONNECT, lw=1.8, zorder=1, solid_capstyle="round")
    axB.scatter([g["kappa_story2motif"]], [y], s=30, color=RECOG, zorder=3)
    axB.scatter([g["kappa_xv_story_thm"]], [y], s=30, color=MAPPING, zorder=3)
axB.set_yticks(ys)
axB.set_yticklabels([LABELS[m] for m in tm])
axB.set_title("Text: identify vs. map (stories)", loc="left")
axB.set_xlabel(r"chance-corrected accuracy $\kappa$")

for ax in axes:
    ax.axvline(0, color=figstyle.NEUTRAL, lw=0.7, ls=":")
    # horizontal dumbbell: vertical gridlines only
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=figstyle.GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.12, 1.0)

handles = [Line2D([], [], marker="o", ls="", color=RECOG, label="recognition"),
           Line2D([], [], marker="o", ls="", color=MAPPING, label="mapping")]
axB.legend(handles=handles, loc="lower right", frameon=False,
           handletextpad=0.1, borderaxespad=0.2)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"mapping_gap_v2.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    shutil.copy(p, os.path.join(FIGS, f"mapping_gap_v2.{ext}"))
    print("wrote", p, "and copied to paper/figs/")
