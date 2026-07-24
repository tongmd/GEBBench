#!/usr/bin/env python3
"""Independent formal distance, steps 3-5: validation + error re-analysis.

  3. Mantel tests: independent GED / eq-token distance vs hand-designed
     family co-membership (and vs each other, and vs pooled confusion).
  4. Error concentration: percentile rank of each pooled identification
     error's GED among the 24 wrong options; permutation test vs uniform.
  5. Dyadic Poisson regression: confusion ~ family + GED-similarity
     (+ eq-similarity), standardized, QAP permutation p-values.

Inputs : eval/analysis_out/independent_distance.json
         eval/results/*.json  (identification tasks, crossvoice skipped)
Output : eval/analysis_out/independent_distance_results.json
         eval/analysis_out/independent_distance.pdf/.png
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/stone/GEB-benchmark")
OUT = ROOT / "eval" / "analysis_out"
RNG = np.random.default_rng(20260719)
N_PERM = 10_000

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

ID_TASKS = ("scene2motif", "skeleton2motif", "story2motif", "theorem2motif")


def family_comembership(motifs):
    idx = {m: i for i, m in enumerate(motifs)}
    S = np.zeros((len(motifs), len(motifs)))
    for fam in FAMILIES:
        for a in fam:
            for b in fam:
                if a != b:
                    S[idx[a], idx[b]] = 1.0
    return S  # similarity: 1 = same family


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


def load_errors(motifs):
    """Pooled and per-model confusion counts + per-error list."""
    idx = {m: i for i, m in enumerate(motifs)}
    per_model = {}
    errors = []  # (model, task, true_idx, pred_idx)
    for f in sorted(glob.glob(str(ROOT / "eval/results/*.json"))):
        if "_aggregate" in f:
            continue
        d = json.loads(Path(f).read_text())
        if "records" not in d:
            continue
        name = Path(f).stem
        m = per_model.setdefault(name, {
            "C": np.zeros((len(motifs), len(motifs))),
            "expo": np.zeros(len(motifs))})
        for r in d["records"]:
            if r["task"] not in ID_TASKS:
                continue
            t = r["answer"]
            m["expo"][idx[t]] += 1
            p = r.get("pred")
            if not r["correct"] and p in idx and p != t:
                m["C"][idx[t], idx[p]] += 1
                errors.append((name, r["task"], idx[t], idx[p]))
    return per_model, errors


# ---- error concentration: percentile ranks --------------------------------
def percentile_ranks(errors, D):
    """For each error (true i, pred j): rank of D[i,j] among D[i,k] for the
    24 wrong options k != i, as a fraction in [0,1]; 0 = nearest. Mid-rank
    ties."""
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
    """Observed mean percentile rank vs null where each error's pred is
    uniform over the 24 wrong options of its true motif."""
    n = D.shape[0]
    obs = percentile_ranks(errors, D)
    obs_mean = float(obs.mean())
    # precompute per true-motif rank vectors
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
        "chance_share_nearest_quartile": 0.25,
    }


# ---- dyadic Poisson regression (QAP) --------------------------------------
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


def main():
    ind = json.loads((OUT / "independent_distance.json").read_text())
    motifs = ind["motifs"]
    GED = np.array(ind["ged_normalized"])
    EQ = np.array(ind["eq_token_distance"])
    FAM = family_comembership(motifs)          # similarity
    FAMD = 1.0 - FAM                            # distance
    np.fill_diagonal(FAMD, 0.0)

    per_model, errors = load_errors(motifs)
    pooledC = sum(m["C"] for m in per_model.values())
    pooledE = sum(m["expo"] for m in per_model.values())
    conf_sim = pooledC + pooledC.T             # symmetrized confusion counts

    res = {"n_motifs": len(motifs), "n_perm": N_PERM, "seed": 20260719,
           "n_models": len(per_model), "models": sorted(per_model)}

    # ---- 3. Mantel validation ------------------------------------------
    man = {}
    for nm, (A, B) in {
        "ged_vs_family": (GED, FAMD),
        "eq_vs_family": (EQ, FAMD),
        "ged_vs_eq": (GED, EQ),
        "confusion_vs_ged": (conf_sim, -GED),
        "confusion_vs_eq": (conf_sim, -EQ),
        "confusion_vs_family": (conf_sim, FAM),
    }.items():
        r, p = mantel(A, B)
        man[nm] = {"mantel_r": round(r, 4), "p": round(p, 5)}
        print(f"Mantel {nm:22s} r={r:+.3f} p={p:.4f}", flush=True)
    res["mantel"] = man

    # family pairs: GED lower inside than outside?
    iu = np.triu_indices(len(motifs), k=1)
    inside = GED[iu][FAM[iu] == 1]
    outside = GED[iu][FAM[iu] == 0]
    res["ged_by_family"] = {
        "n_pairs_inside": int(len(inside)), "n_pairs_outside": int(len(outside)),
        "mean_ged_inside": float(inside.mean()),
        "mean_ged_outside": float(outside.mean()),
    }
    ineq = EQ[iu][FAM[iu] == 1]
    outeq = EQ[iu][FAM[iu] == 0]
    res["eq_by_family"] = {
        "mean_eq_inside": float(ineq.mean()),
        "mean_eq_outside": float(outeq.mean()),
    }

    # ---- 4. error concentration ----------------------------------------
    res["error_concentration"] = {
        "ged": perm_test_mean_rank(errors, GED),
        "eq_token": perm_test_mean_rank(errors, EQ),
    }
    for k, v in res["error_concentration"].items():
        print(f"errors@{k}: mean rank {v['mean_percentile_rank']:.3f} "
              f"(null {v['null_mean']:.3f}) p={v['p_perm_lower']:.4f} "
              f"nearest-quartile share {v['share_in_nearest_quartile']:.3f}",
              flush=True)

    # split by vision vs text tasks (robustness)
    vis = [e for e in errors if e[1] in ("scene2motif", "skeleton2motif")]
    txt = [e for e in errors if e[1] in ("story2motif", "theorem2motif")]
    res["error_concentration_by_modality"] = {
        "vision_ged": perm_test_mean_rank(vis, GED),
        "text_ged": perm_test_mean_rank(txt, GED),
    }

    # frontier-only robustness
    fr = [e for e in errors if e[0].startswith("api_")]
    res["error_concentration_frontier_ged"] = perm_test_mean_rank(fr, GED)

    # ---- 5. dyadic regressions -----------------------------------------
    def std_sim(D):
        return -D
    reg = {}
    preds_full = {"family": FAM, "ged_sim": std_sim(GED),
                  "eq_sim": std_sim(EQ)}
    b, p, ne = dyadic_regression(pooledC, pooledE, preds_full)
    reg["pooled_family_ged_eq"] = {"beta": b, "p_perm": p, "n_errors": ne}
    b, p, ne = dyadic_regression(pooledC, pooledE,
                                 {"family": FAM, "ged_sim": std_sim(GED)})
    reg["pooled_family_ged"] = {"beta": b, "p_perm": p, "n_errors": ne}
    for nm, P1 in [("family", {"family": FAM}),
                   ("ged_sim", {"ged_sim": std_sim(GED)}),
                   ("eq_sim", {"eq_sim": std_sim(EQ)})]:
        b, p, ne = dyadic_regression(pooledC, pooledE, P1)
        reg[f"pooled_uni_{nm}"] = {"beta": b, "p_perm": p, "n_errors": ne}
    res["dyadic_regression"] = reg
    for k, v in reg.items():
        print(k, {kk: round(vv, 3) for kk, vv in v["beta"].items()},
              {kk: round(vv, 4) for kk, vv in v["p_perm"].items()}, flush=True)

    (OUT / "independent_distance_results.json").write_text(
        json.dumps(res, indent=1))
    print("wrote", OUT / "independent_distance_results.json")


if __name__ == "__main__":
    main()
