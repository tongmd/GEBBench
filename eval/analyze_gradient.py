#!/usr/bin/env python3
"""Chance-corrected voice-gradient analysis.

Addresses reviewer point: five task accuracies cannot be directly ordered into
a gradient because arities differ (25-way, 4% chance vs 4-way, 25% chance).

Outputs:
  eval/analysis_out/gradient_stats.json
  eval/analysis_out/gradient_kappa.pdf / .png
"""
import json
import os
import numpy as np
from collections import defaultdict

ROOT = "/Users/stone/GEB-benchmark"
RES = os.path.join(ROOT, "eval", "results")
OUT = os.path.join(ROOT, "eval", "analysis_out")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(20260714)
N_BOOT = 10000

MODELS = ["api_gpt-5.5", "api_gpt-5.2", "api_claude-opus-4-8",
          "qwen2.5-vl-32b", "qwen2-vl-7b"]
MODEL_LABELS = {
    "api_gpt-5.5": "GPT-5.5", "api_gpt-5.2": "GPT-5.2",
    "api_claude-opus-4-8": "Claude Opus 4.8",
    "qwen2.5-vl-32b": "Qwen2.5-VL-32B", "qwen2-vl-7b": "Qwen2-VL-7B",
}
FRONTIER = ["api_gpt-5.5", "api_gpt-5.2", "api_claude-opus-4-8"]

TASKS = ["theorem2motif", "skeleton2motif", "story2motif",
         "scene2motif", "crossvoice"]
TASK_LABELS = {"theorem2motif": "theorem", "skeleton2motif": "skeleton",
               "story2motif": "story", "scene2motif": "scene",
               "crossvoice": "cross-voice"}
# chance level per task (arity)
CHANCE = {"theorem2motif": 1/25, "skeleton2motif": 1/25, "story2motif": 1/25,
          "scene2motif": 1/25, "crossvoice": 1/4}


def load_records():
    data = {}
    for m in MODELS:
        d = json.load(open(os.path.join(RES, m + ".json")))
        # per task -> list of (item_id, correct_bool)
        by_task = defaultdict(list)
        for r in d["records"]:
            by_task[r["task"]].append((r["id"], 1 if r["correct"] else 0))
        data[m] = by_task
    return data


def kappa(acc, chance):
    """Chance-corrected agreement: (acc - chance)/(1 - chance)."""
    return (acc - chance) / (1.0 - chance)


def bits_above_chance(acc, chance):
    """log2(acc/chance), capped at 0 below (undefined for acc=0)."""
    a = max(acc, 1e-9)
    return float(np.log2(a / chance))


def bootstrap_acc(vec, n_boot=N_BOOT):
    """Resample items; return array of bootstrap accuracies."""
    vec = np.asarray(vec, dtype=float)
    n = len(vec)
    idx = RNG.integers(0, n, size=(n_boot, n))
    return vec[idx].mean(axis=1)


def ci95(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def per_model_task_stats(data):
    """Part 1+2: acc, kappa, bits, and bootstrap CIs per model x task."""
    out = {}
    for m in MODELS:
        out[m] = {"label": MODEL_LABELS[m], "tasks": {}}
        for t in TASKS:
            recs = data[m][t]
            vec = np.array([c for _, c in recs], dtype=float)
            n = len(vec)
            acc = float(vec.mean())
            ch = CHANCE[t]
            k = kappa(acc, ch)
            bits = bits_above_chance(acc, ch)
            # bootstrap
            boot_acc = bootstrap_acc(vec)
            boot_kappa = kappa(boot_acc, ch)
            boot_bits = np.array([bits_above_chance(a, ch) for a in boot_acc])
            acc_lo, acc_hi = ci95(boot_acc)
            k_lo, k_hi = ci95(boot_kappa)
            b_lo, b_hi = ci95(boot_bits)
            out[m]["tasks"][t] = {
                "n": n, "chance": ch,
                "acc": acc, "acc_ci95": [acc_lo, acc_hi],
                "kappa": k, "kappa_ci95": [k_lo, k_hi],
                "bits_above_chance": bits, "bits_ci95": [b_lo, b_hi],
            }
    return out


def ordering_check(stats):
    """Part 4: does theorem>skeleton>story>scene>crossvoice survive kappa?
    Report per-model kappa ordering + whether it matches the raw-acc ordering."""
    order_ref = TASKS  # already in claimed gradient order
    result = {}
    for m in MODELS:
        ks = {t: stats[m]["tasks"][t]["kappa"] for t in TASKS}
        accs = {t: stats[m]["tasks"][t]["acc"] for t in TASKS}
        # order tasks by descending kappa
        kappa_order = sorted(TASKS, key=lambda t: -ks[t])
        acc_order = sorted(TASKS, key=lambda t: -accs[t])
        result[m] = {
            "label": MODEL_LABELS[m],
            "kappa_by_task": ks,
            "kappa_descending_order": kappa_order,
            "acc_descending_order": acc_order,
            "matches_claimed_gradient": kappa_order == order_ref,
            "crossvoice_is_hardest_by_kappa": kappa_order[-1] == "crossvoice",
            "crossvoice_is_hardest_by_acc": acc_order[-1] == "crossvoice",
        }
    return result


def crossvoice_vs_scene(data):
    """Part 4 key test: kappa(crossvoice) - kappa(scene) per frontier model,
    with paired item bootstrap CI on the difference. Items differ between the
    two tasks, so resample within each task independently (unpaired bootstrap
    of a difference of two independent statistics)."""
    result = {}
    for m in FRONTIER + ["qwen2.5-vl-32b", "qwen2-vl-7b"]:
        xv = np.array([c for _, c in data[m]["crossvoice"]], dtype=float)
        sc = np.array([c for _, c in data[m]["scene2motif"]], dtype=float)
        ch_xv, ch_sc = CHANCE["crossvoice"], CHANCE["scene2motif"]
        k_xv = kappa(xv.mean(), ch_xv)
        k_sc = kappa(sc.mean(), ch_sc)
        diff = k_xv - k_sc  # negative => crossvoice harder (lower kappa)
        bx = kappa(bootstrap_acc(xv), ch_xv)
        bs = kappa(bootstrap_acc(sc), ch_sc)
        bdiff = bx - bs
        lo, hi = ci95(bdiff)
        # fraction of bootstrap reps where crossvoice kappa < scene kappa
        p_xv_harder = float((bdiff < 0).mean())
        result[m] = {
            "label": MODEL_LABELS[m],
            "kappa_crossvoice": k_xv, "kappa_scene": k_sc,
            "kappa_diff_xv_minus_scene": diff,
            "diff_ci95": [lo, hi],
            "prob_crossvoice_lower_kappa": p_xv_harder,
            "crossvoice_harder_ci_excludes_0": hi < 0 or lo > 0,
        }
    return result


def adjacent_step_diffs(data):
    """Bootstrap CIs on each adjacent kappa step of the claimed gradient,
    per model. Steps: theorem-skeleton, skeleton-story, story-scene,
    scene-crossvoice (positive = earlier task easier, gradient direction)."""
    steps = list(zip(TASKS[:-1], TASKS[1:]))
    result = {}
    for m in MODELS:
        result[m] = {"label": MODEL_LABELS[m], "steps": {}}
        for ta, tb in steps:
            va = np.array([c for _, c in data[m][ta]], dtype=float)
            vb = np.array([c for _, c in data[m][tb]], dtype=float)
            ka = kappa(va.mean(), CHANCE[ta])
            kb = kappa(vb.mean(), CHANCE[tb])
            bdiff = kappa(bootstrap_acc(va), CHANCE[ta]) - \
                    kappa(bootstrap_acc(vb), CHANCE[tb])
            lo, hi = ci95(bdiff)
            result[m]["steps"][f"{TASK_LABELS[ta]}>{TASK_LABELS[tb]}"] = {
                "kappa_diff": ka - kb, "ci95": [lo, hi],
                "robust_positive": lo > 0,
                "prob_positive": float((bdiff > 0).mean()),
            }
    return result


def pooled_frontier(data):
    """Pool the three frontier models per task; cluster bootstrap by item
    (resample items, keep all three model responses per item)."""
    out = {}
    boot_by_task = {}
    for t in TASKS:
        ids = [i for i, _ in data[FRONTIER[0]][t]]
        # matrix: items x 3 models
        mats = []
        for m in FRONTIER:
            d = dict(data[m][t])
            mats.append([d[i] for i in ids])
        mat = np.array(mats, dtype=float).T  # (n_items, 3)
        n = mat.shape[0]
        acc = float(mat.mean())
        ch = CHANCE[t]
        idx = RNG.integers(0, n, size=(N_BOOT, n))
        boot_acc = mat[idx].mean(axis=(1, 2))
        boot_k = kappa(boot_acc, ch)
        boot_by_task[t] = boot_k
        out[t] = {
            "n_items": n, "acc": acc, "kappa": kappa(acc, ch),
            "kappa_ci95": list(ci95(boot_k)),
        }
    # pooled crossvoice - scene kappa difference
    bdiff = boot_by_task["crossvoice"] - boot_by_task["scene2motif"]
    out["_crossvoice_minus_scene"] = {
        "kappa_diff": out["crossvoice"]["kappa"] - out["scene2motif"]["kappa"],
        "ci95": list(ci95(bdiff)),
        "prob_crossvoice_lower": float((bdiff < 0).mean()),
    }
    return out


def ordering_bootstrap(data):
    """Per model: fraction of bootstrap replicates in which (a) the full
    claimed kappa ordering holds, (b) crossvoice has the lowest kappa."""
    out = {}
    for m in MODELS:
        boots = {}
        for t in TASKS:
            vec = np.array([c for _, c in data[m][t]], dtype=float)
            boots[t] = kappa(bootstrap_acc(vec), CHANCE[t])
        arr = np.stack([boots[t] for t in TASKS])  # (5, N_BOOT) in order
        full_order = np.all(arr[:-1] >= arr[1:], axis=0)  # monotone nonincr.
        strict = np.all(arr[:-1] > arr[1:], axis=0)
        xv_lowest = np.all(arr[:-1] > arr[-1], axis=0)
        out[m] = {
            "label": MODEL_LABELS[m],
            "prob_full_ordering_nonstrict": float(full_order.mean()),
            "prob_full_ordering_strict": float(strict.mean()),
            "prob_crossvoice_lowest_kappa": float(xv_lowest.mean()),
        }
    return out


def glm_task_effects(data):
    """Part 3: correct ~ task + model, binomial GLM, cluster-robust SE by
    item id. theorem2motif is the reference task, GPT-5.5 reference model.
    Also an offset variant with logit(chance) as offset, so task coefficients
    are on the above-chance log-odds scale (arity-corrected)."""
    import pandas as pd
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    rows = []
    for m in MODELS:
        for t in TASKS:
            for iid, c in data[m][t]:
                rows.append({"correct": c, "task": t, "model": m,
                             "item": iid,
                             "chance_logit": float(np.log(CHANCE[t] /
                                                          (1 - CHANCE[t])))})
    df = pd.DataFrame(rows)
    df["task"] = pd.Categorical(df["task"], categories=TASKS)  # theorem ref
    df["model"] = pd.Categorical(df["model"], categories=MODELS)  # 5.5 ref

    def fit(formula, offset=None):
        mod = smf.glm(formula, data=df, family=sm.families.Binomial(),
                      offset=offset)
        res = mod.fit(cov_type="cluster",
                      cov_kwds={"groups": df["item"]})
        coefs = {}
        for name in res.params.index:
            coefs[name] = {
                "coef": float(res.params[name]),
                "se": float(res.bse[name]),
                "z": float(res.tvalues[name]),
                "p": float(res.pvalues[name]),
                "ci95": [float(res.conf_int().loc[name, 0]),
                         float(res.conf_int().loc[name, 1])],
            }
        return coefs, res

    coefs_plain, _ = fit("correct ~ task + model")
    coefs_offset, _ = fit("correct ~ task + model",
                          offset=df["chance_logit"])
    return {"reference": {"task": "theorem2motif", "model": "api_gpt-5.5"},
            "cluster_variable": "item_id", "n_obs": len(df),
            "n_clusters": df["item"].nunique(),
            "plain": coefs_plain,
            "chance_offset": coefs_offset}


def make_figure(stats):
    """Part 5: kappa by task per model with CIs, warm palette."""
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
    xs = np.arange(len(TASKS))
    n_m = len(MODELS)
    for i, m in enumerate(MODELS):
        name, color, marker = styles[m]
        off = (i - (n_m - 1) / 2) * 0.07
        ks = [stats[m]["tasks"][t]["kappa"] for t in TASKS]
        los = [stats[m]["tasks"][t]["kappa_ci95"][0] for t in TASKS]
        his = [stats[m]["tasks"][t]["kappa_ci95"][1] for t in TASKS]
        yerr = [np.array(ks) - np.array(los), np.array(his) - np.array(ks)]
        ax.errorbar(xs + off, ks, yerr=yerr, marker=marker, color=color,
                    label=name, lw=1.8, ms=5.5, capsize=2.5,
                    elinewidth=1.0, alpha=0.95)
    ax.axhline(0.0, color="#999999", lw=1, ls=":")
    ax.text(4.45, 0.025, "chance ($\\kappa=0$)", color="#777777",
            fontsize=8.5, ha="right")
    ax.set_xticks(xs, [TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("chance-corrected accuracy $\\kappa$")
    ax.set_ylim(-0.15, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gradient_kappa.pdf"))
    fig.savefig(os.path.join(OUT, "gradient_kappa.png"), dpi=200)
    print("wrote gradient_kappa.pdf/png")


def main():
    data = load_records()
    stats = per_model_task_stats(data)
    ordering = ordering_check(stats)
    xv_vs_scene = crossvoice_vs_scene(data)
    steps = adjacent_step_diffs(data)
    pooled = pooled_frontier(data)
    ord_boot = ordering_bootstrap(data)
    glm = glm_task_effects(data)
    make_figure(stats)

    out = {
        "meta": {
            "models": MODELS, "tasks": TASKS, "chance": CHANCE,
            "n_boot": N_BOOT, "seed": 20260714,
            "kappa_def": "(acc - chance) / (1 - chance)",
            "bits_def": "log2(acc / chance), acc floored at 1e-9",
        },
        "per_model_task": stats,
        "ordering_check": ordering,
        "crossvoice_vs_scene_kappa": xv_vs_scene,
        "adjacent_step_diffs": steps,
        "pooled_frontier": pooled,
        "ordering_bootstrap": ord_boot,
        "glm_cluster_robust": glm,
    }
    with open(os.path.join(OUT, "gradient_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote gradient_stats.json")

    # console summary
    print("\n=== kappa by model x task ===")
    hdr = "model".ljust(18) + "".join(TASK_LABELS[t].rjust(12) for t in TASKS)
    print(hdr)
    for m in MODELS:
        row = MODEL_LABELS[m].ljust(18)
        for t in TASKS:
            s = stats[m]["tasks"][t]
            row += f"{s['kappa']:.3f}".rjust(12)
        print(row)
    print("\n=== ordering (kappa descending) ===")
    for m in MODELS:
        o = ordering[m]
        print(MODEL_LABELS[m].ljust(18),
              " > ".join(TASK_LABELS[t] for t in o["kappa_descending_order"]),
              "| matches claimed:", o["matches_claimed_gradient"],
              "| xv hardest:", o["crossvoice_is_hardest_by_kappa"])
    print("\n=== crossvoice vs scene (kappa diff, xv - scene) ===")
    for m, r in xv_vs_scene.items():
        print(MODEL_LABELS[m].ljust(18),
              f"dk={r['kappa_diff_xv_minus_scene']:+.3f} "
              f"CI[{r['diff_ci95'][0]:+.3f},{r['diff_ci95'][1]:+.3f}] "
              f"P(xv harder)={r['prob_crossvoice_lower_kappa']:.4f}")
    print("\n=== adjacent steps robust? (CI lower bound > 0) ===")
    for m in MODELS:
        for k, v in steps[m]["steps"].items():
            print(MODEL_LABELS[m].ljust(18), k.ljust(20),
                  f"d={v['kappa_diff']:+.3f} CI[{v['ci95'][0]:+.3f},"
                  f"{v['ci95'][1]:+.3f}] robust={v['robust_positive']}")
    print("\n=== GLM task coefs (plain, theorem ref, cluster-robust) ===")
    for k, v in glm["plain"].items():
        if k.startswith("task"):
            print(k.ljust(30), f"b={v['coef']:+.3f} se={v['se']:.3f} "
                  f"p={v['p']:.2e}")
    print("\n=== GLM task coefs (chance-logit offset) ===")
    for k, v in glm["chance_offset"].items():
        if k.startswith("task"):
            print(k.ljust(30), f"b={v['coef']:+.3f} se={v['se']:.3f} "
                  f"p={v['p']:.2e}")
    print("\n=== pooled frontier kappa ===")
    for t in TASKS:
        p = pooled[t]
        print(TASK_LABELS[t].ljust(14),
              f"acc={p['acc']:.3f} kappa={p['kappa']:.3f} "
              f"CI[{p['kappa_ci95'][0]:.3f},{p['kappa_ci95'][1]:.3f}]")
    pd_ = pooled["_crossvoice_minus_scene"]
    print("pooled xv - scene:",
          f"dk={pd_['kappa_diff']:+.3f} "
          f"CI[{pd_['ci95'][0]:+.3f},{pd_['ci95'][1]:+.3f}] "
          f"P(xv harder)={pd_['prob_crossvoice_lower']:.4f}")
    print("\n=== ordering bootstrap ===")
    for m in MODELS:
        o = ord_boot[m]
        print(MODEL_LABELS[m].ljust(18),
              f"P(full order)={o['prob_full_ordering_nonstrict']:.3f} "
              f"P(xv lowest)={o['prob_crossvoice_lowest_kappa']:.3f}")


if __name__ == "__main__":
    main()
