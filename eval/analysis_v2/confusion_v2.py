#!/usr/bin/env python3
"""Confusion geometry + independent-distance replication on v2 results.

Replicates the v1 methods (eval/analyze_logs.py analysis_a/b,
eval/analysis_out/analyze_independent_distance.py) on eval/results_v2:

 1. Pooled identification errors (scene2motif, skeleton2motif, story2motif,
    theorem2motif; valid wrong preds among the 25 motif ids) across 8 models:
    (a) share at formal distance d=1/2/3 vs chance, permutation p (10k);
    (b) Mantel confusion-vs-family r (10k);
    (c) GED percentile-rank concentration (pooled / frontier / text / vision);
    (d) dyadic Poisson regression confusion ~ family + GED-sim (QAP 10k).
 2. Cross-vendor error identity + binary-correctness kappas (frontier),
    plus frontier-vs-smallest-open-vision kappas.
 3. Pooled per-motif recall, bottom 5.
 4. Figure distance_confusion_v2.pdf/png.

Outputs: eval/analysis_v2/confusion_v2.json,
         eval/analysis_v2/distance_confusion_v2.{pdf,png}
         (also copied to paper/figs/).
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import sys
sys.path.insert(0, "/Users/stone/GEB-benchmark")
import figstyle
figstyle.use()
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/stone/GEB-benchmark")
RES = ROOT / "eval" / "results_v2"
OUT = ROOT / "eval" / "analysis_v2"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260721)
N_PERM = 10_000

MODELS = ["or_gpt-5.5", "or_gpt-5.2", "or_claude-opus-4.8",
          "or_gemini-3.1-pro", "or_kimi-k2.5", "or_qwen2.5-vl-72b",
          "or_deepseek-v3.2",
          "qwen2.5-vl-32b", "qwen2-vl-7b",
          "qwen2.5-14b", "qwen2.5-7b", "deepseek-coder-v2-lite"]
FRONTIER = ["or_gpt-5.5", "or_gpt-5.2", "or_claude-opus-4.8",
            "or_gemini-3.1-pro"]
ID_TASKS = ("scene2motif", "skeleton2motif", "story2motif", "theorem2motif")
VISION_TASKS = ("scene2motif", "skeleton2motif")
TEXT_TASKS = ("story2motif", "theorem2motif")

FAMILIES = [  # verbatim from eval/build_items.py:34
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

# ---------------------------------------------------------------- distances
def same_family(a, b):
    return any(a in f and b in f for f in FAMILIES)


def shares_member(a, b):
    fa = set().union(*[set(f) for f in FAMILIES if a in f], set())
    fb = set().union(*[set(f) for f in FAMILIES if b in f], set())
    return len((fa - {a}) & (fb - {b})) > 0


def fam_distance(a, b):
    if same_family(a, b):
        return 1
    if shares_member(a, b):
        return 2
    return 3


def family_comembership(motifs):
    idx = {m: i for i, m in enumerate(motifs)}
    S = np.zeros((len(motifs), len(motifs)))
    for fam in FAMILIES:
        for a in fam:
            for b in fam:
                if a != b:
                    S[idx[a], idx[b]] = 1.0
    return S


# ---------------------------------------------------------------- loading
def load_all(motifs):
    idx = {m: i for i, m in enumerate(motifs)}
    per_model = {}
    errors = []          # (model, task, true_idx, pred_idx)
    recall_num = Counter()   # per-motif correct over ID tasks, all models
    recall_den = Counter()
    id_records = {}      # model -> {item_id: (correct, pred, task)}
    n_id_records = 0
    n_wrong = 0
    n_invalid = 0
    for name in MODELS:
        d = json.loads((RES / f"{name}.json").read_text())
        m = per_model.setdefault(name, {
            "C": np.zeros((len(motifs), len(motifs))),
            "expo": np.zeros(len(motifs))})
        id_records[name] = {}
        for r in d["records"]:
            if r["task"] not in ID_TASKS:
                continue
            n_id_records += 1
            t = r["answer"]
            m["expo"][idx[t]] += 1
            recall_den[t] += 1
            if r["correct"]:
                recall_num[t] += 1
            id_records[name][r["id"]] = (bool(r["correct"]), r.get("pred"),
                                         r["task"])
            p = r.get("pred")
            if not r["correct"]:
                n_wrong += 1
                if p in idx and p != t:
                    m["C"][idx[t], idx[p]] += 1
                    errors.append((name, r["task"], idx[t], idx[p]))
                else:
                    n_invalid += 1
    recall = {t: recall_num[t] / recall_den[t] for t in recall_den}
    meta = {"n_id_records": n_id_records, "n_wrong_total": n_wrong,
            "n_invalid_pred": n_invalid, "n_valid_errors": len(errors)}
    return per_model, errors, recall, recall_den, id_records, meta


# ---------------------------------------------------------------- 1a. d-share
def distance_share(errors, motifs):
    obs = Counter(fam_distance(motifs[i], motifs[j]) for _, _, i, j in errors)
    n = len(errors)
    exp = np.zeros(4)
    for _, _, i, _ in errors:
        a = motifs[i]
        for b in motifs:
            if b != a:
                exp[fam_distance(a, b)] += 1 / 24
    # permutation on d=1 mass: pred uniform over 24 wrong options
    wrong_opts = {a: [b for b in motifs if b != a] for a in motifs}
    d1_flags = {a: np.array([fam_distance(a, b) == 1 for b in wrong_opts[a]])
                for a in motifs}
    trues = [motifs[i] for _, _, i, _ in errors]
    obs_d1 = obs[1]
    count_ge = 0
    for _ in range(N_PERM):
        sim = 0
        for a in trues:
            sim += d1_flags[a][RNG.integers(0, 24)]
        if sim >= obs_d1:
            count_ge += 1
    p = (count_ge + 1) / (N_PERM + 1)
    return {
        "n_errors": n,
        "counts": {d: int(obs[d]) for d in (1, 2, 3)},
        "share": {d: obs[d] / n for d in (1, 2, 3)},
        "chance_share": {d: float(exp[d] / n) for d in (1, 2, 3)},
        "d1_over_chance": float(obs[1] / exp[1]),
        "perm_p_d1": p,
    }


# ---------------------------------------------------------------- Mantel
def mantel(A, B, n_perm=N_PERM):
    iu = np.triu_indices_from(A, k=1)
    a, b = A[iu].astype(float), B[iu].astype(float)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    r_obs = float(np.mean(a * b))
    n = A.shape[0]
    count = 0
    for _ in range(n_perm):
        p = RNG.permutation(n)
        bp = B[p][:, p][iu]
        bp = (bp - bp.mean()) / bp.std()
        if abs(np.mean(a * bp)) >= abs(r_obs):
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


# ------------------------------------------------------- GED percentile rank
def percentile_ranks(errors, D):
    n = D.shape[0]
    ranks = []
    for _, _, i, j in errors:
        row = np.array([D[i, k] for k in range(n) if k != i])
        d = D[i, j]
        less = (row < d).sum()
        eq = (row == d).sum()
        ranks.append((less + (eq - 1) / 2.0) / (len(row) - 1))
    return np.array(ranks)


def perm_test_mean_rank(errors, D, n_perm=N_PERM):
    n = D.shape[0]
    obs = percentile_ranks(errors, D)
    obs_mean = float(obs.mean())
    rank_rows = {}
    for i in {e[2] for e in errors}:
        opts = [k for k in range(n) if k != i]
        row = np.array([D[i, k] for k in opts])
        rr = np.empty(len(opts))
        for a, k in enumerate(opts):
            less = (row < row[a]).sum()
            eq = (row == row[a]).sum()
            rr[a] = (less + (eq - 1) / 2.0) / (len(opts) - 1)
        rank_rows[i] = rr
    trues = np.array([e[2] for e in errors])
    count = 0
    null_means = np.empty(n_perm)
    for b in range(n_perm):
        s = 0.0
        for i in trues:
            s += rank_rows[i][RNG.integers(0, 24)]
        null_means[b] = s / len(trues)
        if null_means[b] <= obs_mean:
            count += 1
    return {
        "n_errors": len(errors),
        "mean_percentile_rank": obs_mean,
        "median_percentile_rank": float(np.median(obs)),
        "null_mean": float(null_means.mean()),
        "null_sd": float(null_means.std()),
        "p_perm_lower": (count + 1) / (n_perm + 1),
        "share_in_nearest_quartile": float((obs <= 0.25).mean()),
    }


# --------------------------------------------------- dyadic Poisson (QAP)
def poisson_fit(X, y, offset, max_iter=100, tol=1e-8):
    Xd = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(Xd.shape[1])
    for _ in range(max_iter):
        eta = Xd @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        W = mu
        z = eta - offset + (y - mu) / np.maximum(mu, 1e-10)
        XtW = Xd.T * W
        try:
            new = np.linalg.solve(XtW @ Xd + 1e-8 * np.eye(Xd.shape[1]),
                                  XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def dyadic_regression(counts, expo, preds, n_perm=N_PERM):
    n = counts.shape[0]
    mask = ~np.eye(n, dtype=bool)
    keep = np.repeat(expo > 0, n).reshape(n, n) & mask
    y = counts[keep]
    off = np.log(np.repeat(expo, n).reshape(n, n)[keep])
    names = list(preds)

    def design(perm=None):
        cols = []
        for nm in names:
            M = preds[nm]
            if perm is not None:
                M = M[perm][:, perm]
            v = M[keep]
            cols.append((v - v.mean()) / (v.std() + 1e-12))
        return np.column_stack(cols)

    b_obs = poisson_fit(design(), y, off)[1:]
    exceed = np.zeros(len(names))
    for _ in range(n_perm):
        p = RNG.permutation(n)
        b = poisson_fit(design(p), y, off)[1:]
        exceed += np.abs(b) >= np.abs(b_obs)
    pvals = (exceed + 1) / (n_perm + 1)
    return dict(zip(names, map(float, b_obs))), \
        dict(zip(names, map(float, pvals))), int(y.sum())


# --------------------------------------------------- cross-model agreement
def pair_stats(recs_a, recs_b):
    shared = sorted(set(recs_a) & set(recs_b))
    v1 = np.array([recs_a[i][0] for i in shared])
    v2 = np.array([recs_b[i][0] for i in shared])
    po = float((v1 == v2).mean())
    p1, p2 = v1.mean(), v2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    both_wrong = [i for i in shared if not recs_a[i][0] and not recs_b[i][0]]
    same_wrong = sum(1 for i in both_wrong
                     if recs_a[i][1] is not None
                     and recs_a[i][1] == recs_b[i][1])
    return {
        "n_shared": len(shared),
        "kappa": float(kappa),
        "n_both_wrong": len(both_wrong),
        "n_same_wrong_pred": int(same_wrong),
        "same_wrong_share": same_wrong / len(both_wrong) if both_wrong
        else None,
    }


# ---------------------------------------------------------------- figure
def make_figure(ds):
    # figure is included at ~0.46\linewidth in the paper -> half-width
    fig, ax = plt.subplots(figsize=(figstyle.HALF, 1.9))
    dists = [1, 2, 3]
    obs_frac = [ds["share"][d] * 100 for d in dists]
    exp_frac = [ds["chance_share"][d] * 100 for d in dists]
    x = np.arange(3)
    ax.bar(x - 0.18, obs_frac, 0.36, label="Observed errors",
           color=figstyle.ACCENT)
    ax.bar(x + 0.18, exp_frac, 0.36,
           label="Chance (uniform over wrong options)", color="#BBBBBB")
    ax.set_xticks(x)
    ax.set_xticklabels(["Same family\n(d=1)", "Shares a member\n(d=2)",
                        "Unrelated\n(d=3)"])
    ax.set_ylabel("Share of all errors (%)")
    ax.set_title("Errors concentrate at minimal formal distance", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        kw = {"dpi": 150} if ext == "png" else {}
        plt.savefig(OUT / f"distance_confusion_v2.{ext}",
                    bbox_inches="tight", **kw)
        plt.savefig(ROOT / "paper" / "figs" / f"distance_confusion_v2.{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote distance_confusion_v2.pdf/png")


def main():
    ind = json.loads(
        (ROOT / "eval/analysis_out/independent_distance.json").read_text())
    motifs = ind["motifs"]
    GED = np.array(ind["ged_normalized"])
    FAM = family_comembership(motifs)

    per_model, errors, recall, recall_den, id_records, meta = load_all(motifs)
    print("meta:", meta)

    res = {"seed": 20260721, "n_perm": N_PERM, "models": MODELS,
           "id_tasks": list(ID_TASKS), "meta": meta}

    # ---- 1a. formal-distance shares + permutation ----------------------
    ds = distance_share(errors, motifs)
    res["distance_share"] = ds
    print("d-share:", {d: f"{ds['share'][d]:.3f}" for d in (1, 2, 3)},
          "chance:", {d: f"{ds['chance_share'][d]:.3f}" for d in (1, 2, 3)},
          f"ratio={ds['d1_over_chance']:.2f} p={ds['perm_p_d1']:.4f}")

    # ---- 1b. Mantel confusion vs family --------------------------------
    pooledC = sum(m["C"] for m in per_model.values())
    pooledE = sum(m["expo"] for m in per_model.values())
    conf_sim = pooledC + pooledC.T
    r, p = mantel(conf_sim, FAM)
    res["mantel_confusion_vs_family"] = {"r": r, "p": p}
    print(f"Mantel confusion~family r={r:+.3f} p={p:.4f}")

    # scene+skeleton pooled subset (sec:geometry numbers)
    C_vis = np.zeros_like(pooledC)
    idx = {m: i for i, m in enumerate(motifs)}
    n_vis_err = 0
    n_vis_fam = 0
    for name, task, i, j in errors:
        if task in VISION_TASKS:
            C_vis[i, j] += 1
            n_vis_err += 1
            if fam_distance(motifs[i], motifs[j]) == 1:
                n_vis_fam += 1
    r_vis, p_vis = mantel(C_vis + C_vis.T, FAM)
    # inside/outside symmetrized-count contrast (fig:exp1-style number)
    rate = (C_vis + C_vis.T)
    iu = np.triu_indices(len(motifs), k=1)
    inside = rate[iu][FAM[iu] == 1]
    outside = rate[iu][FAM[iu] == 0]
    res["scene_skeleton_subset"] = {
        "n_errors": n_vis_err, "n_same_family": n_vis_fam,
        "same_family_share": n_vis_fam / n_vis_err,
        "mantel_r": r_vis, "mantel_p": p_vis,
        "mean_count_inside": float(inside.mean()),
        "mean_count_outside": float(outside.mean()),
        "inside_over_outside": float(inside.mean() / outside.mean()),
        "family_dyad_share": float(FAM[iu].mean()),
    }
    print(f"scene+skeleton: {n_vis_fam}/{n_vis_err} same-family "
          f"(dyad share {FAM[iu].mean():.3f}), Mantel r={r_vis:+.3f} "
          f"p={p_vis:.4f}, inside/outside={inside.mean()/outside.mean():.2f}")

    # ---- 1c. GED percentile-rank concentration -------------------------
    conc = {"pooled": perm_test_mean_rank(errors, GED)}
    conc["frontier"] = perm_test_mean_rank(
        [e for e in errors if e[0] in FRONTIER], GED)
    conc["vision_tasks"] = perm_test_mean_rank(
        [e for e in errors if e[1] in VISION_TASKS], GED)
    conc["text_tasks"] = perm_test_mean_rank(
        [e for e in errors if e[1] in TEXT_TASKS], GED)
    res["ged_concentration"] = conc
    for k, v in conc.items():
        print(f"GED conc {k:12s} n={v['n_errors']:4d} "
              f"mean rank {v['mean_percentile_rank']:.3f} "
              f"(null {v['null_mean']:.3f}) p={v['p_perm_lower']:.4f}")

    # ---- 1d. dyadic regression family + GED-sim ------------------------
    b, pq, ne = dyadic_regression(pooledC, pooledE,
                                  {"family": FAM, "ged_sim": -GED})
    res["dyadic_family_ged"] = {"beta": b, "p_perm": pq, "n_errors": ne}
    print("dyadic family+ged:", {k: round(v, 3) for k, v in b.items()},
          {k: round(v, 4) for k, v in pq.items()})

    # ---- 2. cross-vendor error identity + kappas -----------------------
    xm = {}
    for a, bn in itertools.combinations(FRONTIER, 2):
        xm[f"{a}__{bn}"] = pair_stats(id_records[a], id_records[bn])
    for fm in FRONTIER:
        xm[f"{fm}__qwen2-vl-7b"] = pair_stats(id_records[fm],
                                              id_records["qwen2-vl-7b"])
    res["cross_model"] = xm
    for k, v in xm.items():
        sw = ("n/a" if v["same_wrong_share"] is None
              else f"{v['same_wrong_share']:.2f}")
        print(f"{k}: kappa={v['kappa']:.2f} both-wrong={v['n_both_wrong']} "
              f"same-wrong={sw}")

    # ---- 3. hardest motifs ---------------------------------------------
    ranked = sorted(recall.items(), key=lambda kv: kv[1])
    res["per_motif_recall"] = {m: {"recall": recall[m],
                                   "n": recall_den[m]} for m in motifs}
    res["hardest_bottom5"] = [
        {"motif": m, "recall": r_, "n": recall_den[m]} for m, r_ in ranked[:5]]
    print("bottom 5:", [(m, f"{r_:.3f}") for m, r_ in ranked[:5]])

    # top pooled confusions (prose check)
    pairs = []
    for i in range(len(motifs)):
        for j in range(len(motifs)):
            if pooledC[i, j] > 0:
                pairs.append((motifs[i], motifs[j], int(pooledC[i, j])))
    pairs.sort(key=lambda t: -t[2])
    res["top_confusions"] = [
        {"true": a, "pred": b, "count": c} for a, b, c in pairs[:15]]
    print("top confusions:", pairs[:8])

    # ---- 4. figure ------------------------------------------------------
    make_figure(ds)

    (OUT / "confusion_v2.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "confusion_v2.json")


if __name__ == "__main__":
    main()
