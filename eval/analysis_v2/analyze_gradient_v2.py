#!/usr/bin/env python3
"""Chance-corrected voice-gradient analysis on v2 results.

Replicates eval/analyze_gradient.py on eval/results_v2, extended with the
v2-only tasks:
  - per model x task kappa = (acc - chance)/(1 - chance), 10k bootstrap CIs
    (all tasks each model ran, incl. xv_scene_thm / xv_story_thm / family4);
  - ordering check on the five voice-gradient tasks (v2 reference ordering
    theorem > story > skeleton > scene > crossvoice) + ordering bootstrap;
  - pooled-frontier kappas (cluster bootstrap by item);
  - GLM correct ~ task + model, cluster-robust SE by item id, plain and with
    logit(chance) offset;
  - family4 vs scene2motif (same 210 scenes, menu 25 -> 4): paired contrast;
  - family4 vs crossvoice (same 210 scenes, both 4-way): paired contrast
    + exact McNemar (mapping deficit at matched chance);
  - crossvoice vs scene kappa difference (unpaired bootstrap, as in v1).

Outputs:
  eval/analysis_v2/gradient_v2.json
  eval/analysis_v2/gradient_kappa_v2.pdf/.png (copied to paper/figs/)
"""
import json
import os
import shutil
import numpy as np
from collections import defaultdict

ROOT = "/Users/stone/GEB-benchmark"
RES = os.path.join(ROOT, "eval", "results_v2")
OUT = os.path.join(ROOT, "eval", "analysis_v2")
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(20260721)
N_BOOT = 10000

VISION = ["api_gpt-5.5", "api_gpt-5.2", "api_claude-opus-4-8",
          "qwen2.5-vl-32b", "qwen2-vl-7b"]
TEXT_ONLY = ["qwen2.5-14b", "qwen2.5-7b", "deepseek-coder-v2-lite"]
MODELS = VISION + TEXT_ONLY
FRONTIER = ["api_gpt-5.5", "api_gpt-5.2", "api_claude-opus-4-8"]
LABELS = {
    "api_gpt-5.5": "GPT-5.5", "api_gpt-5.2": "GPT-5.2",
    "api_claude-opus-4-8": "Claude Opus 4.8",
    "qwen2.5-vl-32b": "Qwen2.5-VL-32B", "qwen2-vl-7b": "Qwen2-VL-7B",
    "qwen2.5-14b": "Qwen2.5-14B", "qwen2.5-7b": "Qwen2.5-7B",
    "deepseek-coder-v2-lite": "DeepSeek-Coder-V2-Lite",
}

# v2 reference gradient ordering (frontier): theorem > story > skeleton >
# scene > crossvoice. v1 claimed skeleton > story; we test both.
GRADIENT = ["theorem2motif", "story2motif", "skeleton2motif",
            "scene2motif", "crossvoice"]
V1_ORDER = ["theorem2motif", "skeleton2motif", "story2motif",
            "scene2motif", "crossvoice"]
TASK_LABELS = {"theorem2motif": "theorem", "story2motif": "story",
               "skeleton2motif": "skeleton", "scene2motif": "scene",
               "crossvoice": "cross-voice", "xv_scene_thm": "xv-scene-thm",
               "xv_story_thm": "xv-story-thm",
               "scene2motif_family4": "family-4"}
CHANCE = {"theorem2motif": 1/25, "story2motif": 1/25, "skeleton2motif": 1/25,
          "scene2motif": 1/25, "crossvoice": 1/4, "xv_scene_thm": 1/4,
          "xv_story_thm": 1/4, "scene2motif_family4": 1/4}


def kappa(acc, chance):
    return (acc - chance) / (1.0 - chance)


def bootstrap_acc(vec, n_boot=N_BOOT):
    vec = np.asarray(vec, dtype=float)
    idx = RNG.integers(0, len(vec), size=(n_boot, len(vec)))
    return vec[idx].mean(axis=1)


def ci95(arr):
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def load_records():
    data = {}
    for m in MODELS:
        d = json.load(open(os.path.join(RES, m + ".json")))
        by_task = defaultdict(list)
        for r in d["records"]:
            by_task[r["task"]].append((r["id"], 1 if r["correct"] else 0))
        data[m] = by_task
    return data


def vec(data, m, t):
    return np.array([c for _, c in data[m][t]], dtype=float)


def keyed(data, m, t):
    """item key (task prefix stripped) -> 0/1, for paired scene contrasts."""
    return {i.split(":", 1)[1]: c for i, c in data[m][t]}


def per_model_task_stats(data):
    out = {}
    for m in MODELS:
        out[m] = {"label": LABELS[m], "tasks": {}}
        for t in sorted(data[m], key=lambda t: list(CHANCE).index(t)):
            v = vec(data, m, t)
            acc = float(v.mean())
            ch = CHANCE[t]
            boot = bootstrap_acc(v)
            out[m]["tasks"][t] = {
                "n": len(v), "chance": ch,
                "acc": acc, "acc_ci95": ci95(boot),
                "kappa": kappa(acc, ch),
                "kappa_ci95": ci95(kappa(boot, ch)),
            }
    return out


def ordering_check(stats):
    res = {}
    for m in VISION:
        ks = {t: stats[m]["tasks"][t]["kappa"] for t in GRADIENT}
        order = sorted(GRADIENT, key=lambda t: -ks[t])
        res[m] = {
            "label": LABELS[m],
            "kappa_by_task": ks,
            "kappa_descending_order": [TASK_LABELS[t] for t in order],
            "matches_v2_gradient": order == GRADIENT,
            "matches_v1_gradient": order == V1_ORDER,
            "crossvoice_is_lowest_kappa": order[-1] == "crossvoice",
        }
    return res


def ordering_bootstrap(data):
    out = {}
    for m in VISION:
        boots = {t: kappa(bootstrap_acc(vec(data, m, t)), CHANCE[t])
                 for t in GRADIENT}
        arr = np.stack([boots[t] for t in GRADIENT])          # v2 order
        arr1 = np.stack([boots[t] for t in V1_ORDER])
        out[m] = {
            "label": LABELS[m],
            "prob_v2_ordering_nonstrict":
                float(np.all(arr[:-1] >= arr[1:], axis=0).mean()),
            "prob_v1_ordering_nonstrict":
                float(np.all(arr1[:-1] >= arr1[1:], axis=0).mean()),
            "prob_crossvoice_lowest_kappa":
                float(np.all(arr[:-1] > arr[-1], axis=0).mean()),
            "prob_story_above_skeleton":
                float((boots["story2motif"] >
                       boots["skeleton2motif"]).mean()),
        }
    return out


def pooled_frontier(data):
    out = {}
    boot_by_task = {}
    for t in GRADIENT + ["scene2motif_family4"]:
        ids = [i for i, _ in data[FRONTIER[0]][t]]
        mats = []
        for m in FRONTIER:
            d = dict(data[m][t])
            mats.append([d[i] for i in ids])
        mat = np.array(mats, dtype=float).T          # (n_items, 3)
        n = mat.shape[0]
        acc = float(mat.mean())
        ch = CHANCE[t]
        idx = RNG.integers(0, n, size=(N_BOOT, n))
        boot_k = kappa(mat[idx].mean(axis=(1, 2)), ch)
        boot_by_task[t] = boot_k
        out[t] = {"n_items": n, "acc": acc, "kappa": kappa(acc, ch),
                  "kappa_ci95": ci95(boot_k)}
    arr = np.stack([boot_by_task[t] for t in GRADIENT])
    out["_prob_v2_ordering"] = float(np.all(arr[:-1] >= arr[1:],
                                            axis=0).mean())
    bdiff = boot_by_task["crossvoice"] - boot_by_task["scene2motif"]
    out["_crossvoice_minus_scene"] = {
        "kappa_diff": out["crossvoice"]["kappa"] - out["scene2motif"]["kappa"],
        "ci95": ci95(bdiff),
        "prob_crossvoice_lower": float((bdiff < 0).mean()),
    }
    bss = boot_by_task["story2motif"] - boot_by_task["skeleton2motif"]
    out["_story_minus_skeleton"] = {
        "kappa_diff": out["story2motif"]["kappa"]
                      - out["skeleton2motif"]["kappa"],
        "ci95": ci95(bss),
        "prob_story_higher": float((bss > 0).mean()),
    }
    return out


def adjacent_step_diffs(data):
    steps = list(zip(GRADIENT[:-1], GRADIENT[1:]))
    res = {}
    for m in VISION:
        res[m] = {"label": LABELS[m], "steps": {}}
        for ta, tb in steps:
            va, vb = vec(data, m, ta), vec(data, m, tb)
            ka, kb = kappa(va.mean(), CHANCE[ta]), kappa(vb.mean(), CHANCE[tb])
            bdiff = kappa(bootstrap_acc(va), CHANCE[ta]) - \
                    kappa(bootstrap_acc(vb), CHANCE[tb])
            res[m]["steps"][f"{TASK_LABELS[ta]}>{TASK_LABELS[tb]}"] = {
                "kappa_diff": float(ka - kb), "ci95": ci95(bdiff),
                "robust_positive": bool(np.percentile(bdiff, 2.5) > 0),
                "prob_positive": float((bdiff > 0).mean()),
            }
    return res


def crossvoice_vs_scene(data):
    """v1 analysis: kappa(crossvoice) - kappa(scene), unpaired bootstrap."""
    res = {}
    for m in VISION:
        xv, sc = vec(data, m, "crossvoice"), vec(data, m, "scene2motif")
        k_xv = kappa(xv.mean(), CHANCE["crossvoice"])
        k_sc = kappa(sc.mean(), CHANCE["scene2motif"])
        bdiff = kappa(bootstrap_acc(xv), CHANCE["crossvoice"]) - \
                kappa(bootstrap_acc(sc), CHANCE["scene2motif"])
        lo, hi = np.percentile(bdiff, [2.5, 97.5])
        res[m] = {
            "label": LABELS[m],
            "kappa_crossvoice": float(k_xv), "kappa_scene": float(k_sc),
            "kappa_diff_xv_minus_scene": float(k_xv - k_sc),
            "diff_ci95": [float(lo), float(hi)],
            "prob_crossvoice_lower_kappa": float((bdiff < 0).mean()),
            "crossvoice_harder_ci_excludes_0": bool(hi < 0 or lo > 0),
        }
    return res


def paired_contrast(data, m, task_a, task_b):
    """Paired-by-scene contrast between two tasks on the same 210 scenes.
    Returns acc/kappa for each, paired bootstrap CIs on both gaps, and exact
    McNemar. a - b convention."""
    from scipy import stats as sps
    da, db = keyed(data, m, task_a), keyed(data, m, task_b)
    keys = sorted(set(da) & set(db))
    a = np.array([da[k] for k in keys], dtype=float)
    b = np.array([db[k] for k in keys], dtype=float)
    n = len(keys)
    assert n == 210, (m, task_a, task_b, n)
    cha, chb = CHANCE[task_a], CHANCE[task_b]
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    acc_gap_b = a[idx].mean(1) - b[idx].mean(1)
    kap_gap_b = kappa(a[idx].mean(1), cha) - kappa(b[idx].mean(1), chb)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    p_mcn = sps.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue \
        if n10 + n01 else 1.0
    return {
        "n": n,
        f"acc_{task_a}": float(a.mean()), f"acc_{task_b}": float(b.mean()),
        f"kappa_{task_a}": float(kappa(a.mean(), cha)),
        f"kappa_{task_b}": float(kappa(b.mean(), chb)),
        "acc_gap_a_minus_b": float(a.mean() - b.mean()),
        "acc_gap_ci95": ci95(acc_gap_b),
        "kappa_gap_a_minus_b": float(kappa(a.mean(), cha)
                                     - kappa(b.mean(), chb)),
        "kappa_gap_ci95": ci95(kap_gap_b),
        "mcnemar_a_only": n10, "mcnemar_b_only": n01,
        "mcnemar_p": float(p_mcn),
    }


def glm_task_effects(data, models):
    """correct ~ task + model on the 5 gradient tasks for the given models;
    binomial GLM, cluster-robust SE by item id; plain + chance-logit offset.
    Reference: theorem2motif / GPT-5.5."""
    import pandas as pd
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    rows = []
    for m in models:
        for t in GRADIENT:
            for iid, c in data[m][t]:
                rows.append({"correct": c, "task": t, "model": m, "item": iid,
                             "chance_logit": float(np.log(
                                 CHANCE[t] / (1 - CHANCE[t])))})
    df = pd.DataFrame(rows)
    df["task"] = pd.Categorical(df["task"], categories=GRADIENT)
    df["model"] = pd.Categorical(df["model"], categories=models)

    def fit(offset=None):
        mod = smf.glm("correct ~ task + model", data=df,
                      family=sm.families.Binomial(), offset=offset)
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["item"]})
        return {name: {
            "coef": float(res.params[name]), "se": float(res.bse[name]),
            "z": float(res.tvalues[name]), "p": float(res.pvalues[name]),
            "ci95": [float(res.conf_int().loc[name, 0]),
                     float(res.conf_int().loc[name, 1])],
        } for name in res.params.index}

    return {"reference": {"task": "theorem2motif", "model": models[0]},
            "models": list(models),
            "cluster_variable": "item_id", "n_obs": len(df),
            "n_clusters": df["item"].nunique(),
            "plain": fit(),
            "chance_offset": fit(offset=df["chance_logit"])}


def make_figure(stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG = "#F7F4EF"
    SURFACE = "#FBF9F5"
    MUTED = "#5C635D"
    INK = "#1F2421"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": BG,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#E7E1D7",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    styles = {"api_gpt-5.5": ("GPT-5.5", "#B3402A", "o"),
              "api_gpt-5.2": ("GPT-5.2", "#E8A03E", "s"),
              "api_claude-opus-4-8": ("Claude Opus 4.8", "#7A4B94", "D"),
              "qwen2.5-vl-32b": ("Qwen2.5-VL-32B", "#4C8C5C", "v"),
              "qwen2-vl-7b": ("Qwen2-VL-7B", "#3B4A8C", "^")}
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    xs = np.arange(len(GRADIENT))
    n_m = len(VISION)
    for i, m in enumerate(VISION):
        name, color, marker = styles[m]
        off = (i - (n_m - 1) / 2) * 0.07
        ks = [stats[m]["tasks"][t]["kappa"] for t in GRADIENT]
        los = [stats[m]["tasks"][t]["kappa_ci95"][0] for t in GRADIENT]
        his = [stats[m]["tasks"][t]["kappa_ci95"][1] for t in GRADIENT]
        yerr = [np.array(ks) - np.array(los), np.array(his) - np.array(ks)]
        ax.errorbar(xs + off, ks, yerr=yerr, marker=marker, color=color,
                    label=name, lw=1.8, ms=5.5, capsize=2.5,
                    elinewidth=1.0, alpha=0.95)
    ax.axhline(0.0, color="#999999", lw=1, ls=":")
    ax.text(4.45, 0.025, "chance ($\\kappa=0$)", color="#777777",
            fontsize=8.5, ha="right")
    ax.set_xticks(xs, [TASK_LABELS[t] for t in GRADIENT])
    ax.set_ylabel("chance-corrected accuracy $\\kappa$")
    ax.set_ylim(-0.15, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"gradient_kappa_v2.{ext}")
        fig.savefig(p, dpi=200)
        shutil.copy(p, os.path.join(FIGS, f"gradient_kappa_v2.{ext}"))
    print("wrote gradient_kappa_v2.pdf/png (copied to paper/figs/)")


def main():
    data = load_records()
    stats = per_model_task_stats(data)
    ordering = ordering_check(stats)
    ord_boot = ordering_bootstrap(data)
    pooled = pooled_frontier(data)
    steps = adjacent_step_diffs(data)
    xv_scene = crossvoice_vs_scene(data)
    fam4_scene = {m: paired_contrast(data, m, "scene2motif_family4",
                                     "scene2motif") for m in VISION}
    fam4_xv = {m: paired_contrast(data, m, "scene2motif_family4",
                                  "crossvoice") for m in VISION}
    glm = glm_task_effects(data, VISION)
    glm_frontier = glm_task_effects(data, FRONTIER)
    make_figure(stats)

    out = {
        "meta": {
            "models": MODELS, "vision_models": VISION,
            "gradient_tasks": GRADIENT, "chance": CHANCE,
            "n_boot": N_BOOT, "seed": 20260721,
            "kappa_def": "(acc - chance) / (1 - chance)",
        },
        "per_model_task": stats,
        "ordering_check": ordering,
        "ordering_bootstrap": ord_boot,
        "pooled_frontier": pooled,
        "adjacent_step_diffs": steps,
        "crossvoice_vs_scene_kappa": xv_scene,
        "family4_vs_scene_paired": fam4_scene,
        "family4_vs_crossvoice_paired": fam4_xv,
        "glm_cluster_robust": glm,
        "glm_cluster_robust_frontier_only": glm_frontier,
    }
    with open(os.path.join(OUT, "gradient_v2.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote gradient_v2.json")

    # console summary
    print("\n=== kappa by model x gradient task ===")
    print("model".ljust(24) + "".join(TASK_LABELS[t].rjust(13)
                                      for t in GRADIENT))
    for m in VISION:
        print(LABELS[m].ljust(24) + "".join(
            f"{stats[m]['tasks'][t]['kappa']:.3f}".rjust(13)
            for t in GRADIENT))
    print("\n=== ordering (kappa descending) ===")
    for m, o in ordering.items():
        print(LABELS[m].ljust(24), " > ".join(o["kappa_descending_order"]),
              "| v2:", o["matches_v2_gradient"],
              "| v1:", o["matches_v1_gradient"],
              "| xv lowest:", o["crossvoice_is_lowest_kappa"])
    print("\n=== ordering bootstrap ===")
    for m, o in ord_boot.items():
        print(LABELS[m].ljust(24),
              f"P(v2 order)={o['prob_v2_ordering_nonstrict']:.3f} "
              f"P(v1 order)={o['prob_v1_ordering_nonstrict']:.3f} "
              f"P(xv lowest)={o['prob_crossvoice_lowest_kappa']:.3f} "
              f"P(story>skel)={o['prob_story_above_skeleton']:.3f}")
    print("\n=== pooled frontier ===")
    for t in GRADIENT + ["scene2motif_family4"]:
        p = pooled[t]
        print(TASK_LABELS[t].ljust(14),
              f"acc={p['acc']:.3f} kappa={p['kappa']:.3f} "
              f"CI[{p['kappa_ci95'][0]:.3f},{p['kappa_ci95'][1]:.3f}]")
    print("P(v2 order, pooled) =", pooled["_prob_v2_ordering"])
    ss = pooled["_story_minus_skeleton"]
    print(f"pooled story-skel dk={ss['kappa_diff']:+.3f} "
          f"CI[{ss['ci95'][0]:+.3f},{ss['ci95'][1]:+.3f}] "
          f"P(story>skel)={ss['prob_story_higher']:.3f}")
    xs = pooled["_crossvoice_minus_scene"]
    print(f"pooled xv-scene dk={xs['kappa_diff']:+.3f} "
          f"CI[{xs['ci95'][0]:+.3f},{xs['ci95'][1]:+.3f}] "
          f"P(xv lower)={xs['prob_crossvoice_lower']:.4f}")
    print("\n=== family4 vs scene (paired, menu 25->4) ===")
    for m in VISION:
        g = fam4_scene[m]
        print(LABELS[m].ljust(24),
              f"acc {g['acc_scene2motif']*100:.1f}->"
              f"{g['acc_scene2motif_family4']*100:.1f} "
              f"kappa {g['kappa_scene2motif']:.3f}->"
              f"{g['kappa_scene2motif_family4']:.3f} "
              f"dk={g['kappa_gap_a_minus_b']:+.3f} "
              f"CI[{g['kappa_gap_ci95'][0]:+.3f},"
              f"{g['kappa_gap_ci95'][1]:+.3f}]")
    print("\n=== family4 vs crossvoice (paired, both 4-way) ===")
    for m in VISION:
        g = fam4_xv[m]
        print(LABELS[m].ljust(24),
              f"acc {g['acc_scene2motif_family4']*100:.1f} vs "
              f"{g['acc_crossvoice']*100:.1f} "
              f"gap={g['acc_gap_a_minus_b']*100:+.1f} "
              f"CI[{g['acc_gap_ci95'][0]*100:+.1f},"
              f"{g['acc_gap_ci95'][1]*100:+.1f}] "
              f"McNemar p={g['mcnemar_p']:.2e}")
    print("\n=== GLM (plain) task coefs vs theorem ===")
    for k, v in glm["plain"].items():
        if k.startswith("task"):
            print(k.ljust(32), f"b={v['coef']:+.3f} se={v['se']:.3f} "
                  f"p={v['p']:.2e}")
    print("=== GLM (chance offset) task coefs vs theorem ===")
    for k, v in glm["chance_offset"].items():
        if k.startswith("task"):
            print(k.ljust(32), f"b={v['coef']:+.3f} se={v['se']:.3f} "
                  f"p={v['p']:.2e}")
    print("=== GLM frontier-only (plain) task coefs vs theorem ===")
    for k, v in glm_frontier["plain"].items():
        if k.startswith("task"):
            print(k.ljust(32), f"b={v['coef']:+.3f} se={v['se']:.3f} "
                  f"p={v['p']:.2e}")
    print("=== GLM frontier-only (chance offset) task coefs vs theorem ===")
    for k, v in glm_frontier["chance_offset"].items():
        if k.startswith("task"):
            print(k.ljust(32), f"b={v['coef']:+.3f} se={v['se']:.3f} "
                  f"p={v['p']:.2e}")


if __name__ == "__main__":
    main()
