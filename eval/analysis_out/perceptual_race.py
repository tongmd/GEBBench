#!/usr/bin/env python3
"""Perceptual-race robustness analysis for the reviewer point:
"conceptual-not-perceptual is overclaimed — add CLIP, SigLIP baselines".

Steps
  1. Motif-centroid pairwise cosine-distance matrices for three perceptual
     embeddings: DINOv2 (existing exp1_geometry/embeddings.npz), CLIP
     ViT-B/32 and SigLIP base (freshly computed, eval/analysis_out/*.npz).
  2. Mantel tests of each perceptual geometry against (a) formal family
     co-membership, (b) the pooled model confusion matrix (rebuilt from
     eval/results/*.json + exp1_geometry/results/*.json identification
     errors on scene2motif + skeleton2motif).
  3. Dyadic Poisson regression race on pooled confusions:
     confusion ~ formal family + label prior + DINOv2 + CLIP + SigLIP
     (standardized similarities, 10k QAP node-permutation p-values).
     Univariate races per predictor + deviance/variance explained alone.
  4. Within-motif control replicated with each embedding: render's
     leave-one-out distance to own centroid vs correctness (per model,
     permutation within motif; plus pooled across models).

Outputs: eval/analysis_out/perceptual_race.json, perceptual_race.pdf/png
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path("/Users/stone/GEB-benchmark")
EXP = ROOT / "exp1_geometry"
OUT = ROOT / "eval" / "analysis_out"
RNG = np.random.default_rng(20260719)
N_PERM = 10_000
VISION_TASKS = ("scene2motif", "skeleton2motif")

EMB_NAMES = ["dinov2", "clip_b32", "siglip"]


# ---------------------------------------------------------------- embeddings
def load_embeddings():
    """Return dict name -> (paths list, L2-normalized (125,D) array)."""
    out = {}
    z = np.load(EXP / "embeddings.npz", allow_pickle=True)
    paths = [str(p) for p in z["paths"]]
    E = np.asarray(z["dino"], dtype=np.float64)
    out["dinov2"] = (paths, E / np.linalg.norm(E, axis=1, keepdims=True))
    for name, fn in (("clip_b32", "percep_clip.npz"),
                     ("siglip", "percep_siglip.npz")):
        z2 = np.load(OUT / fn, allow_pickle=True)
        p2 = [str(p) for p in z2["paths"]]
        assert p2 == paths, f"{fn}: path ordering mismatch with embeddings.npz"
        E2 = np.asarray(z2["emb"], dtype=np.float64)
        out[name] = (paths, E2 / np.linalg.norm(E2, axis=1, keepdims=True))
    return out


def centroid_dist(paths, E, motifs):
    cent = np.zeros((len(motifs), E.shape[1]))
    for i, m in enumerate(motifs):
        rows = [k for k, p in enumerate(paths) if p.split("/")[0] == m]
        assert len(rows) >= 5, f"{m}: only {len(rows)} renders"
        c = E[rows].mean(0)
        cent[i] = c / np.linalg.norm(c)
    D = 1.0 - cent @ cent.T
    np.fill_diagonal(D, 0.0)
    return D


# ------------------------------------------------------------------- mantel
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


# ------------------------------------------------------------ response data
def load_confusions(motifs):
    idx = {m: i for i, m in enumerate(motifs)}
    models = {}
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
    return models


# --------------------------------------------------------- poisson IRLS GLM
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


def poisson_deviance(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * float(np.sum(t - (y - mu)))


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

    X = design()
    b_full = poisson_fit(X, y, off)
    b_obs = b_full[1:]
    # deviance explained (McFadden vs intercept-only-with-offset null)
    mu_full = np.exp(np.clip(np.column_stack([np.ones(len(y)), X]) @ b_full
                             + off, -30, 30))
    b0 = poisson_fit(np.zeros((len(y), 0)), y, off)
    mu0 = np.exp(np.clip(b0[0] + off, -30, 30))
    dev_expl = 1.0 - poisson_deviance(y, mu_full) / poisson_deviance(y, mu0)
    exceed = np.zeros(len(names))
    for _ in range(n_perm):
        p = RNG.permutation(n)
        b = poisson_fit(design(p), y, off)[1:]
        exceed += np.abs(b) >= np.abs(b_obs)
    pvals = (exceed + 1) / (n_perm + 1)
    return (dict(zip(names, b_obs)), dict(zip(names, pvals)),
            int(y.sum()), float(dev_expl))


def ols_r2(counts, expo, sim):
    """Plain variance in dyadic confusion *rates* explained by one predictor
    alone (OLS R^2) — the 'max variance explained' number."""
    n = counts.shape[0]
    mask = ~np.eye(n, dtype=bool)
    keep = np.repeat(expo > 0, n).reshape(n, n) & mask
    rate = counts[keep] / np.repeat(expo, n).reshape(n, n)[keep]
    x = sim[keep]
    r = np.corrcoef(x, rate)[0, 1]
    return float(r * r)


# ------------------------------------------------------------- kill-shot
def kill_shot(records, paths, E, n_perm=N_PERM):
    row_of = {p: k for k, p in enumerate(paths)}
    xs, ys, groups = [], [], []
    for r in records:
        if r["task"] != "scene2motif":
            continue
        mid = r["answer"]
        stem = r["id"].split(":")[1]
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
    xs, ys, groups = np.array(xs), np.array(ys), np.array(groups)
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
    return {"r": round(r_obs, 4), "p": round((cnt + 1) / (n_perm + 1), 5),
            "n": int(len(xs))}


# --------------------------------------------------------------------- main
def main():
    F = json.loads((EXP / "matrices_formal.json").read_text())
    motifs = F["motifs"]
    fam = np.array(F["formal_family"])   # distance: 0 same-family, 1 not
    lab = np.array(F["label_prior"])     # distance

    embs = load_embeddings()
    D = {}   # perceptual distance matrices
    for name in EMB_NAMES:
        paths, E = embs[name]
        D[name] = centroid_dist(paths, E, motifs)

    models = load_confusions(motifs)
    pooledC = np.zeros_like(fam)
    pooledE = np.zeros(len(motifs))
    pooled_records = []
    for d in models.values():
        pooledC += d["C"]
        pooledE += d["expo"]
        pooled_records += d["records"]

    # symmetric pooled confusion-rate matrix for Mantel
    conf_rate = np.zeros_like(fam)
    n = len(motifs)
    for i in range(n):
        for j in range(n):
            if i != j and (pooledE[i] + pooledE[j]) > 0:
                conf_rate[i, j] = ((pooledC[i, j] + pooledC[j, i])
                                   / (pooledE[i] + pooledE[j]))
    conf_sym = (conf_rate + conf_rate.T) / 2.0

    results = {"n_motifs": n, "n_perm": N_PERM,
               "n_models_pooled": len(models),
               "embeddings": {
                   "dinov2": "facebook/dinov2-base (CLS), precomputed",
                   "clip_b32": "openai/clip-vit-base-patch32 image features",
                   "siglip": "google/siglip-base-patch16-224 image features"}}

    # ---- Mantel tests ---------------------------------------------------
    mantels = {}
    for name in EMB_NAMES:
        r1, p1 = mantel(D[name], fam)
        r2, p2 = mantel(D[name], conf_sym)
        mantels[name] = {
            "vs_family": {"mantel_r": round(r1, 4), "p": round(p1, 5)},
            "vs_pooled_confusion": {"mantel_r": round(r2, 4),
                                    "p": round(p2, 5)}}
        print(f"Mantel {name:9s} vs family    r={r1:+.3f} p={p1:.4f}")
        print(f"Mantel {name:9s} vs confusion r={r2:+.3f} p={p2:.4f}")
    # reference: formal family vs confusion
    rf, pf = mantel(1.0 - fam, conf_sym)
    mantels["formal_family_ref"] = {
        "vs_pooled_confusion": {"mantel_r": round(rf, 4), "p": round(pf, 5)}}
    print(f"Mantel family-sim vs confusion r={rf:+.3f} p={pf:.4f}")
    # inter-embedding agreement
    inter = {}
    for a, b in (("dinov2", "clip_b32"), ("dinov2", "siglip"),
                 ("clip_b32", "siglip")):
        r, p = mantel(D[a], D[b])
        inter[f"{a}_vs_{b}"] = {"mantel_r": round(r, 4), "p": round(p, 5)}
        print(f"Mantel {a} vs {b}: r={r:+.3f} p={p:.4f}")
    results["mantel"] = mantels
    results["mantel_inter_embedding"] = inter

    # ---- dyadic regression race -----------------------------------------
    preds = {"formal_family": 1.0 - fam,
             "label_prior": -lab,
             "dinov2": -D["dinov2"],
             "clip_b32": -D["clip_b32"],
             "siglip": -D["siglip"]}
    betas, ps, nerr, dev = dyadic_regression(pooledC, pooledE, preds)
    results["race_pooled"] = {
        "beta": {k: round(v, 4) for k, v in betas.items()},
        "p_perm": {k: round(v, 5) for k, v in ps.items()},
        "n_errors": nerr, "deviance_explained": round(dev, 4)}
    print(f"\nRACE (pooled, n_err={nerr}, dev_expl={dev:.3f}):")
    for k in preds:
        print(f"   {k:14s} beta={betas[k]:+.3f} p={ps[k]:.4f}")

    # ---- univariate: each predictor alone --------------------------------
    uni = {}
    for k in preds:
        bu, pu, _, du = dyadic_regression(pooledC, pooledE, {k: preds[k]})
        r2 = ols_r2(pooledC, pooledE, preds[k])
        uni[k] = {"beta": round(bu[k], 4), "p_perm": round(pu[k], 5),
                  "deviance_explained": round(du, 4),
                  "ols_r2_rate": round(r2, 4)}
        print(f"UNIVARIATE {k:14s} beta={bu[k]:+.3f} p={pu[k]:.4f} "
              f"dev={du:.4f} olsR2={r2:.4f}")
    results["univariate_pooled"] = uni
    max_perc = max((uni[k]["ols_r2_rate"], k)
                   for k in ("dinov2", "clip_b32", "siglip"))
    max_perc_dev = max((uni[k]["deviance_explained"], k)
                       for k in ("dinov2", "clip_b32", "siglip"))
    results["max_perceptual_alone"] = {
        "ols_r2": {"value": round(max_perc[0], 4), "embedding": max_perc[1]},
        "deviance_explained": {"value": round(max_perc_dev[0], 4),
                               "embedding": max_perc_dev[1]}}
    print(f"MAX perceptual alone: OLS R2={max_perc[0]:.4f} ({max_perc[1]}), "
          f"deviance={max_perc_dev[0]:.4f} ({max_perc_dev[1]})")

    # ---- per-model race (compact: betas only, 2k perms) ------------------
    per_model = {}
    for mdl, d in models.items():
        b, p, ne, dv = dyadic_regression(d["C"], d["expo"], preds,
                                         n_perm=2000)
        per_model[mdl] = {"beta": {k: round(v, 4) for k, v in b.items()},
                          "p_perm": {k: round(v, 5) for k, v in p.items()},
                          "n_errors": ne}
    results["race_per_model"] = per_model

    # ---- within-motif kill-shot per embedding -----------------------------
    ks = {}
    for name in EMB_NAMES:
        paths, E = embs[name]
        ks[name] = {"POOLED": kill_shot(pooled_records, paths, E)}
        for mdl, d in models.items():
            if not any(r["task"] == "scene2motif" for r in d["records"]):
                continue
            ks[name][mdl] = kill_shot(d["records"], paths, E, n_perm=2000)
        print(f"kill-shot {name} POOLED: r={ks[name]['POOLED']['r']:+.3f} "
              f"p={ks[name]['POOLED']['p']:.3f} n={ks[name]['POOLED']['n']}")
    results["within_motif_control"] = ks

    (OUT / "perceptual_race.json").write_text(json.dumps(results, indent=1))
    print("\nwrote", OUT / "perceptual_race.json")


if __name__ == "__main__":
    main()
