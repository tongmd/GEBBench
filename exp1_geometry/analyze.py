#!/usr/bin/env python3
"""Exp1 — analysis: decoupling pre-check, dyadic regression race, kill-shot.

Inputs
  exp1_geometry/embeddings.npz        (dino/clip embeddings of all renders)
  exp1_geometry/matrices_formal.json  (formal_family, formal_ged, label_prior)
  eval/results/*.json                 (scene2motif + skeleton2motif records)
  exp1_geometry/results/*.json        (nuisance-resample scene2motif runs)

Pipeline
  1. PERCEPTUAL matrices: per-motif centroid of L2-normalized embeddings,
     pairwise cosine distance (DINOv2 primary, CLIP robustness).
  2. Pre-check: Mantel r(PERCEPTUAL, FORMAL-family) and (PERCEPTUAL, GED).
     If r > 0.2 vs the primary formal predictor -> DECOUPLING FAILURE, stop.
  3. Dyadic dataset per model: ordered pairs (i,j), i!=j; y = # errors with
     true=i, pred=j pooled over scene2motif + skeleton2motif; exposure =
     # items with answer=i.  Poisson GLM with log-exposure offset,
     standardized predictors: same_family (1=same), -GED... predictors are
     entered as SIMILARITIES (negated distances) so beta>0 = "closer in this
     space -> more confusions" for every predictor uniformly.
  4. Significance: 10k node (QAP) permutations of the 25 motif labels applied
     jointly to all predictor matrices; permutation p per coefficient.
  5. Kill-shot control: per scene item, correctness vs cosine distance to own
     motif centroid (leave-one-out); permutation within motif; should be flat.

Output: exp1_geometry/results.json (+ printed report)
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "exp1_geometry"
RNG = np.random.default_rng(20260717)
N_PERM = 10_000

VISION_TASKS = ("scene2motif", "skeleton2motif")


# ---------------------------------------------------------------- perceptual
def load_perceptual(motifs):
    z = np.load(EXP / "embeddings.npz", allow_pickle=True)
    paths = [str(p) for p in z["paths"]]
    out = {}
    for key in ("dino", "clip"):
        E = z[key]
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        cent = np.zeros((len(motifs), E.shape[1]))
        for i, m in enumerate(motifs):
            rows = [k for k, p in enumerate(paths) if p.split("/")[0] == m]
            assert len(rows) >= 5, f"{m}: only {len(rows)} renders"
            c = E[rows].mean(0)
            cent[i] = c / np.linalg.norm(c)
        out[key] = 1.0 - cent @ cent.T
        np.fill_diagonal(out[key], 0.0)
    # per-item info for kill-shot
    item_info = {"paths": paths,
                 "dino": z["dino"] / np.linalg.norm(z["dino"], axis=1,
                                                    keepdims=True)}
    return out["dino"], out["clip"], item_info


# ------------------------------------------------------------------- mantel
def mantel(A, B, n_perm=N_PERM):
    iu = np.triu_indices_from(A, k=1)
    a, b = A[iu], B[iu]
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    r_obs = float(np.mean(a * b))
    n = A.shape[0]
    count = 0
    for _ in range(n_perm):
        p = RNG.permutation(n)
        bp = B[p][:, p][iu]
        bp = (bp - bp.mean()) / bp.std()
        r = np.mean(a * bp)
        if abs(r) >= abs(r_obs):
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


# ------------------------------------------------------------ response data
def load_confusions(motifs):
    """per model: counts[i,j] = #errors true=i pred=j; exposure[i] = #items."""
    idx = {m: i for i, m in enumerate(motifs)}
    models = {}
    # main-eval v2 vision errors (210 scenes + 49 skeletons per vision model)
    # + the controlled nuisance-resample render pool; both OpenRouter/local
    # provenance.  (The v1 pilot pool is superseded: its scenes are a subset
    # of the v2 scene set, so pooling both would double-count stimuli.)
    files = (sorted(glob.glob(str(ROOT / "eval/results_v2/*.json")))
             + sorted(glob.glob(str(EXP / "results/*.json"))))
    for f in files:
        d = json.loads(Path(f).read_text())
        if "records" not in d:
            continue
        recs = [r for r in d["records"] if r["task"] in VISION_TASKS]
        if not recs:
            continue
        m = models.setdefault(d["model"], {
            "C": np.zeros((len(motifs), len(motifs))),
            "expo": np.zeros(len(motifs)), "records": []})
        for r in recs:
            t = r["answer"]
            m["expo"][idx[t]] += 1
            m["records"].append(r)
            p = r.get("pred")
            if not r["correct"] and p in idx and p != t:
                m["C"][idx[t], idx[p]] += 1
    for m in models.values():
        m["n_err"] = int(m["C"].sum())
    return models


# --------------------------------------------------------- poisson IRLS GLM
def poisson_fit(X, y, offset, max_iter=100, tol=1e-8):
    """Poisson GLM with log link + offset. Returns beta (incl. intercept)."""
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
    """preds: dict name -> 25x25 SIMILARITY matrix (higher = closer).
    Returns betas + permutation p-values (node/QAP permutation)."""
    n = counts.shape[0]
    mask = ~np.eye(n, dtype=bool)
    keep = np.repeat(expo > 0, n).reshape(n, n) & mask  # rows with exposure
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
    return dict(zip(names, b_obs)), dict(zip(names, pvals)), int(y.sum())


# ------------------------------------------------------------- kill-shot
def kill_shot(models, motifs, item_info, n_perm=N_PERM):
    """within-motif: item correctness vs distance to own-motif centroid
    (leave-one-out).  Point-biserial r, permutation within motif."""
    paths = item_info["paths"]
    E = item_info["dino"]
    # map scene item id -> embedding row: id 'scene:scene_xxx:motif'
    row_of = {p: k for k, p in enumerate(paths)}
    out = {}
    for mdl, d in models.items():
        xs, ys, groups = [], [], []
        for r in d["records"]:
            if r["task"] != "scene2motif":
                continue
            mid = r["answer"]
            stem = r["id"].split(":")[1]  # scene_xxx
            key = f"{mid}/{stem}.png"
            if key not in row_of:
                continue
            k = row_of[key]
            rows = [j for j, p in enumerate(paths)
                    if p.split("/")[0] == mid and j != k]
            c = E[rows].mean(0)
            c /= np.linalg.norm(c)
            xs.append(1.0 - float(E[k] @ c))
            ys.append(1.0 if r["correct"] else 0.0)
            groups.append(mid)
        xs, ys = np.array(xs), np.array(ys)
        groups = np.array(groups)
        # within-motif centering to remove motif difficulty
        xc, yc = xs.copy(), ys.copy()
        for g in set(groups):
            s = groups == g
            xc[s] -= xs[s].mean()
            yc[s] -= ys[s].mean()
        denom = xc.std() * yc.std()
        r_obs = float(np.mean(xc * yc) / denom) if denom > 0 else 0.0
        cnt = 0
        for _ in range(n_perm):
            yp = yc.copy()
            for g in set(groups):
                s = np.where(groups == g)[0]
                yp[s] = yp[RNG.permutation(s)]
            d2 = xc.std() * yp.std()
            r = float(np.mean(xc * yp) / d2) if d2 > 0 else 0.0
            if abs(r) >= abs(r_obs):
                cnt += 1
        out[mdl] = {"r": r_obs, "p": (cnt + 1) / (n_perm + 1), "n": len(xs)}
    return out


# --------------------------------------------------------------------- main
def main():
    F = json.loads((EXP / "matrices_formal.json").read_text())
    motifs = F["motifs"]
    fam = np.array(F["formal_family"])   # distance: 0 same-family, 1 not
    ged = np.array(F["formal_ged"])      # distance
    lab = np.array(F["label_prior"])     # distance
    dino, clip, item_info = load_perceptual(motifs)

    results = {"n_motifs": len(motifs), "n_perm": N_PERM}

    # ---- decoupling pre-check ------------------------------------------
    checks = {}
    for nm, P in (("dino_vs_family", (dino, fam)),
                  ("dino_vs_ged", (dino, ged)),
                  ("clip_vs_family", (clip, fam)),
                  ("dino_vs_label", (dino, lab)),
                  ("family_vs_ged", (fam, ged))):
        r, p = mantel(*P)
        checks[nm] = {"mantel_r": round(r, 4), "p": round(p, 5)}
        print(f"Mantel {nm:18s} r={r:+.3f} p={p:.4f}", flush=True)
    results["decoupling"] = checks
    r_primary = checks["dino_vs_family"]["mantel_r"]
    if r_primary > 0.2:
        print("\n" + "!" * 70)
        print(f"!!! DECOUPLING FAILURE: Mantel r(PERCEPTUAL, FORMAL-family) = "
              f"{r_primary:.3f} > 0.2 — the race cannot be run. STOPPING.")
        print("!" * 70)
        results["verdict"] = "DECOUPLING FAILURE — regression not run"
        (EXP / "results.json").write_text(json.dumps(results, indent=1))
        sys.exit(2)
    print(f"decoupling OK (r={r_primary:+.3f} <= 0.2)\n", flush=True)

    # ---- regression race ------------------------------------------------
    # predictors as similarities: closer => larger => beta>0 means "closeness
    # in this space drives confusion"
    preds = {"formal_family": 1.0 - fam,
             "formal_ged": -ged,
             "perceptual_dino": -dino,
             "label_prior": -lab}
    models = load_confusions(motifs)
    reg = {}
    pooledC = np.zeros_like(fam)
    pooledE = np.zeros(len(motifs))
    for mdl, d in models.items():
        betas, ps, nerr = dyadic_regression(d["C"], d["expo"], preds)
        reg[mdl] = {"beta": {k: round(v, 4) for k, v in betas.items()},
                    "p_perm": {k: round(v, 5) for k, v in ps.items()},
                    "n_errors": nerr}
        pooledC += d["C"]
        pooledE += d["expo"]
        print(f"{mdl}: n_err={nerr}")
        for k in preds:
            print(f"   {k:16s} beta={betas[k]:+.3f} p={ps[k]:.4f}")
    betas, ps, nerr = dyadic_regression(pooledC, pooledE, preds)
    reg["POOLED"] = {"beta": {k: round(v, 4) for k, v in betas.items()},
                     "p_perm": {k: round(v, 5) for k, v in ps.items()},
                     "n_errors": nerr}
    print(f"POOLED: n_err={nerr}")
    for k in preds:
        print(f"   {k:16s} beta={betas[k]:+.3f} p={ps[k]:.4f}")
    results["regression"] = reg

    # CLIP robustness on pooled
    preds_clip = dict(preds, perceptual_dino=-clip)
    b2, p2, _ = dyadic_regression(pooledC, pooledE, preds_clip)
    results["regression_clip_robustness"] = {
        "beta": {k: round(v, 4) for k, v in b2.items()},
        "p_perm": {k: round(v, 5) for k, v in p2.items()}}

    # univariate races on pooled
    uni = {}
    for k in preds:
        bu, pu, _ = dyadic_regression(pooledC, pooledE, {k: preds[k]})
        uni[k] = {"beta": round(bu[k], 4), "p_perm": round(pu[k], 5)}
        print(f"UNIVARIATE pooled {k:16s} beta={bu[k]:+.3f} p={pu[k]:.4f}")
    results["regression_univariate_pooled"] = uni

    # same-family enrichment
    same = (fam == 0) & ~np.eye(len(motifs), dtype=bool)
    mask = ~np.eye(len(motifs), dtype=bool)
    enrich = {"errors_same_family": int(pooledC[same].sum()),
              "errors_total": int(pooledC[mask].sum()),
              "same_family_dyad_share": round(same.sum() / mask.sum(), 4),
              "per_model": {m: {"same_family": int(d["C"][same].sum()),
                                "total": int(d["C"][mask].sum())}
                            for m, d in models.items()}}
    results["same_family_enrichment"] = enrich
    print(f"same-family enrichment: {enrich['errors_same_family']}/"
          f"{enrich['errors_total']} vs base rate "
          f"{enrich['same_family_dyad_share']:.1%}")

    # ---- kill-shot -------------------------------------------------------
    ks = kill_shot(models, motifs, item_info)
    results["kill_shot"] = {m: {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in d.items()} for m, d in ks.items()}
    for m, d in ks.items():
        print(f"kill-shot {m}: r={d['r']:+.3f} p={d['p']:.3f} n={d['n']}")

    # ---- verdict ---------------------------------------------------------
    bp = reg["POOLED"]["beta"]
    pp = reg["POOLED"]["p_perm"]
    formal_wins = (bp["formal_family"] > bp["perceptual_dino"]
                   and pp["formal_family"] < 0.05)
    results["verdict"] = (
        f"FORMAL WINS: pooled beta_formal_family={bp['formal_family']:+.3f} "
        f"(p={pp['formal_family']:.4f}) vs beta_perceptual="
        f"{bp['perceptual_dino']:+.3f} (p={pp['perceptual_dino']:.4f})"
        if formal_wins else
        f"FORMAL DOES NOT WIN: beta_formal_family={bp['formal_family']:+.3f} "
        f"(p={pp['formal_family']:.4f}) vs beta_perceptual="
        f"{bp['perceptual_dino']:+.3f} (p={pp['perceptual_dino']:.4f})")
    print("\nVERDICT:", results["verdict"])

    (EXP / "results.json").write_text(json.dumps(results, indent=1))
    print("wrote", EXP / "results.json")


if __name__ == "__main__":
    main()
