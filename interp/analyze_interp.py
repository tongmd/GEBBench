#!/usr/bin/env python3
"""Analysis + figures for the GemmaScope GEB-Bench experiment.

Consumes results/acts_{tag}.npz + meta_{tag}.json from run_gemmascope.py.
Produces (in results/figures/):

  fig_overlap_{tag}.png     : same-motif vs different-motif cross-voice
                              similarity (cosine + top-k Jaccard) per layer,
                              with permutation-test p-values
  fig_geometry_{tag}.png    : story-voice SAE similarity vs (a) behavioral
                              confusion (pooled over eval/results/*.json) and
                              (b) family co-membership; Mantel tests
  fig_layersweep_{tag}.png  : voice-invariance gap + retrieval@1 across layers
  fig_probe_{tag}.png       : token-level logistic probe, trained on story
                              tokens, tested on theorem documents (cross-voice)
  metrics_{tag}.json        : all numbers used in the figures

Runs on CPU, no GPU needed. Usage:
  python analyze_interp.py --tag zh
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = Path(__file__).resolve().parent / "results"

# keep in sync with eval/build_items.py FAMILIES
sys.path.insert(0, str(ROOT / "eval"))
try:
    from build_items import FAMILIES  # type: ignore
except Exception:  # pragma: no cover — fallback snapshot (2026-07)
    FAMILIES = [
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

VOICE_IDX = {"story": 0, "theorem": 1, "invariant": 2}


# ---------------------------------------------------------------- helpers

def cos_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return An @ Bn.T


def jaccard_topk(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    ta = np.argsort(-A, axis=1)[:, :k]
    tb = np.argsort(-B, axis=1)[:, :k]
    n, m = A.shape[0], B.shape[0]
    J = np.zeros((n, m))
    sa = [set(r.tolist()) for r in ta]
    sb = [set(r.tolist()) for r in tb]
    for i in range(n):
        for j in range(m):
            inter = len(sa[i] & sb[j])
            J[i, j] = inter / (2 * k - inter)
    return J


def diag_gap_perm(S: np.ndarray, n_perm: int = 10000, seed: int = 0):
    """Statistic: mean(diag) - mean(offdiag). Null: permute column labels."""
    n = S.shape[0]
    diag = float(np.mean(np.diag(S)))
    off = float((S.sum() - np.trace(S)) / (n * n - n))
    stat = diag - off
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        Sp = S[:, perm]
        null[p] = np.mean(np.diag(Sp)) - (Sp.sum() - np.trace(Sp)) / (n * n - n)
    pval = float((np.sum(null >= stat) + 1) / (n_perm + 1))
    return {"diag": diag, "offdiag": off, "gap": stat, "p": pval,
            "retrieval_top1": float(np.mean(S.argmax(1) == np.arange(n)))}


def mantel(D1: np.ndarray, D2: np.ndarray, n_perm: int = 10000, seed: int = 0):
    """Mantel test, Spearman correlation on upper-triangle entries."""
    from scipy.stats import spearmanr
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    r = float(spearmanr(D1[iu], D2[iu]).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        null[p] = spearmanr(D1[perm][:, perm][iu], D2[iu]).statistic
    pval = float((np.sum(np.abs(null) >= abs(r)) + 1) / (n_perm + 1))
    return r, pval


# ---------------------------------------------------------------- external matrices

def family_matrix(ids: list[str]) -> np.ndarray:
    """M[i,j] = number of confusable families containing both i and j."""
    idx = {m: i for i, m in enumerate(ids)}
    M = np.zeros((len(ids), len(ids)))
    for fam in FAMILIES:
        fam = [m for m in fam if m in idx]
        for a in fam:
            for b in fam:
                if a != b:
                    M[idx[a], idx[b]] += 1
    return M


def confusion_matrix(ids: list[str], tasks=("story2motif", "theorem2motif")):
    """Pooled error counts over eval/results/*.json → symmetrized rates."""
    idx = {m: i for i, m in enumerate(ids)}
    C = np.zeros((len(ids), len(ids)))
    used = []
    for f in sorted((ROOT / "eval" / "results").glob("*.json")):
        if f.name.startswith("_"):
            continue
        d = json.loads(f.read_text())
        recs = d.get("records", [])
        hit = 0
        for r in recs:
            if r.get("task") in tasks and r.get("answer") in idx \
                    and r.get("pred") in idx:
                C[idx[r["answer"]], idx[r["pred"]]] += 1
                hit += 1
        if hit:
            used.append(f.name)
    row = C.sum(axis=1, keepdims=True)
    R = C / np.maximum(row, 1)          # row-normalized rates
    S = (R + R.T) / 2                   # symmetrize for Mantel
    np.fill_diagonal(S, 0)
    return S, used, C


# ---------------------------------------------------------------- probe

def crossvoice_probe(npz, meta, layer: int, seed: int = 0):
    """Token-level 25-way logistic probe. Train: story tokens. Test: theorem
    docs (mean of per-token class probabilities). Sparse top-k token features
    are densified per token."""
    from sklearn.linear_model import LogisticRegression
    ids = meta["motif_ids"]
    d_sae = meta["d_sae"]

    def tokens(i, vi):
        idx = npz[f"tokidx_L{layer}_{i}_{vi}"]
        val = npz[f"tokval_L{layer}_{i}_{vi}"].astype(np.float32)
        X = np.zeros((idx.shape[0], d_sae), dtype=np.float32)
        rows = np.arange(idx.shape[0])[:, None]
        X[rows, idx] = val
        return X

    Xtr, ytr = [], []
    for i in range(len(ids)):
        X = tokens(i, VOICE_IDX["story"])
        Xtr.append(X)
        ytr += [i] * X.shape[0]
    Xtr = np.vstack(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=seed)
    clf.fit(Xtr, np.array(ytr))

    correct, preds = 0, []
    for i in range(len(ids)):
        P = clf.predict_proba(tokens(i, VOICE_IDX["theorem"]))
        pred = int(P.mean(0).argmax())
        preds.append(pred)
        correct += int(pred == i)
    return {"acc": correct / len(ids), "chance": 1 / len(ids), "preds": preds}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="zh", help="matches acts_{tag}.npz")
    ap.add_argument("--pool", choices=["mean", "max"], default="max",
                    help="document pooling used for main metrics")
    ap.add_argument("--topk", type=int, default=50, help="k for Jaccard overlap")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--probe", action="store_true", help="run the token probe (slow-ish)")
    args = ap.parse_args()

    meta = json.loads((RES / f"meta_{args.tag}.json").read_text())
    npz = np.load(RES / f"acts_{args.tag}.npz")
    ids, layers = meta["motif_ids"], meta["layers"]
    n = len(ids)
    fig_dir = RES / "figures"
    fig_dir.mkdir(exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {"tag": args.tag, "pool": args.pool, "topk": args.topk, "layers": {}}

    # ---- (a)+(c) per-layer cross-voice overlap ---------------------------
    for L in layers:
        V = npz[f"{args.pool}_L{L}"].astype(np.float32)  # (n, 3, d_sae)
        s, t, inv = V[:, 0], V[:, 1], V[:, 2]
        res = {
            "cos_story_thm": diag_gap_perm(cos_matrix(s, t), args.n_perm),
            "jac_story_thm": diag_gap_perm(jaccard_topk(s, t, args.topk), args.n_perm),
            "cos_story_inv": diag_gap_perm(cos_matrix(s, inv), args.n_perm),
            "cos_thm_inv": diag_gap_perm(cos_matrix(t, inv), args.n_perm),
        }
        out["layers"][str(L)] = res
        print(f"layer {L}: cos gap={res['cos_story_thm']['gap']:.4f} "
              f"(p={res['cos_story_thm']['p']:.4f}) "
              f"jac gap={res['jac_story_thm']['gap']:.4f} "
              f"(p={res['jac_story_thm']['p']:.4f}) "
              f"retr@1={res['cos_story_thm']['retrieval_top1']:.2f}")

    # best layer = max cosine gap
    best_L = max(layers, key=lambda L: out["layers"][str(L)]["cos_story_thm"]["gap"])
    out["best_layer"] = best_L

    # fig 1: overlap bars at best layer
    V = npz[f"{args.pool}_L{best_L}"].astype(np.float32)
    pairs = [("story↔theorem", cos_matrix(V[:, 0], V[:, 1])),
             ("story↔invariant", cos_matrix(V[:, 0], V[:, 2])),
             ("theorem↔invariant", cos_matrix(V[:, 1], V[:, 2]))]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(pairs))
    same = [np.mean(np.diag(S)) for _, S in pairs]
    diff = [(S.sum() - np.trace(S)) / (n * n - n) for _, S in pairs]
    ax.bar(x - 0.18, same, 0.36, label="same motif")
    ax.bar(x + 0.18, diff, 0.36, label="different motif")
    for xi, (name, S) in zip(x, pairs):
        p = diag_gap_perm(S, args.n_perm)["p"]
        ax.text(xi, max(same[xi], diff[xi]) + 0.01, f"p={p:.3g}", ha="center", fontsize=8)
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.set_ylabel(f"SAE feature cosine ({args.pool}-pool, layer {best_L})")
    ax.set_title("Voice-invariant motif features (GemmaScope)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_overlap_{args.tag}.png", dpi=200)

    # ---- (b) geometry: SAE similarity vs confusion vs families -----------
    story_sim = cos_matrix(V[:, 0], V[:, 0])
    np.fill_diagonal(story_sim, 0)
    fam = family_matrix(ids)
    conf, used_files, _ = confusion_matrix(ids)
    r_conf, p_conf = mantel(story_sim, conf, args.n_perm)
    r_fam, p_fam = mantel(story_sim, fam, args.n_perm)
    r_fc, p_fc = mantel(fam, conf, args.n_perm)  # sanity: families predict confusion
    out["geometry"] = {"layer": best_L,
                       "mantel_sim_vs_confusion": {"r": r_conf, "p": p_conf},
                       "mantel_sim_vs_family": {"r": r_fam, "p": p_fam},
                       "mantel_family_vs_confusion": {"r": r_fc, "p": p_fc},
                       "confusion_pooled_from": used_files}
    print(f"[geometry] SAE-sim vs confusion: r={r_conf:.3f} p={p_conf:.4f} | "
          f"vs family: r={r_fam:.3f} p={p_fam:.4f} | "
          f"family vs confusion: r={r_fc:.3f} p={p_fc:.4f}")

    iu = np.triu_indices(n, k=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fam_pair = fam[iu] > 0
    axes[0].scatter(story_sim[iu][~fam_pair], conf[iu][~fam_pair],
                    s=12, alpha=0.5, label="non-family pair")
    axes[0].scatter(story_sim[iu][fam_pair], conf[iu][fam_pair],
                    s=16, alpha=0.8, color="crimson", label="confusable-family pair")
    axes[0].set_xlabel(f"SAE story-feature cosine (layer {best_L})")
    axes[0].set_ylabel("behavioral confusion rate (pooled, symmetrized)")
    axes[0].set_title(f"Mantel r={r_conf:.2f}, p={p_conf:.3g}")
    axes[0].legend(fontsize=8)
    axes[1].boxplot([story_sim[iu][~fam_pair], story_sim[iu][fam_pair]],
                    tick_labels=["non-family", "family"])
    axes[1].set_ylabel("SAE story-feature cosine")
    axes[1].set_title(f"family pairs closer? Mantel r={r_fam:.2f}, p={p_fam:.3g}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_geometry_{args.tag}.png", dpi=200)

    # ---- (c) layer sweep figure -------------------------------------------
    fig, ax1 = plt.subplots(figsize=(6, 4))
    gaps = [out["layers"][str(L)]["cos_story_thm"]["gap"] for L in layers]
    retr = [out["layers"][str(L)]["cos_story_thm"]["retrieval_top1"] for L in layers]
    ax1.plot(layers, gaps, "o-", color="tab:blue", label="cosine gap (same−diff)")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("same−different cosine gap", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(layers, retr, "s--", color="tab:orange", label="retrieval@1")
    ax2.axhline(1 / n, color="gray", lw=0.8, ls=":")
    ax2.set_ylabel("story→theorem retrieval@1", color="tab:orange")
    ax1.set_title("Where does the voice-invariant signal live?")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_layersweep_{args.tag}.png", dpi=200)

    # ---- (d) probe ---------------------------------------------------------
    if args.probe:
        accs = {}
        for L in layers:
            r = crossvoice_probe(npz, meta, L)
            accs[L] = r["acc"]
            print(f"[probe] layer {L}: story-tokens→theorem-doc acc={r['acc']:.2f} "
                  f"(chance {r['chance']:.2f})")
        out["probe"] = {str(L): a for L, a in accs.items()}
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(accs), list(accs.values()), "o-")
        ax.axhline(1 / n, color="gray", ls=":", label="chance")
        ax.set_xlabel("layer")
        ax.set_ylabel("cross-voice probe accuracy")
        ax.set_title("Linear probe: train on story tokens, test on theorems")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig_probe_{args.tag}.png", dpi=200)

    (RES / f"metrics_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[save] metrics_{args.tag}.json + figures/ in {RES}")


if __name__ == "__main__":
    sys.exit(main())
