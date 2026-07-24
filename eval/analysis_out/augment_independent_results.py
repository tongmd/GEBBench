#!/usr/bin/env python3
"""Augment independent_distance_results.json:
  - graded family distance (d=1/2/3) percentile ranks on the same splits,
    for apples-to-apples comparison with the independent GED;
  - dyadic regression variant with row-percentile GED similarity
    (row-conditioned, removes per-graph scale);
  - per-model mean GED percentile ranks.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("/Users/stone/GEB-benchmark/eval/analysis_out")
sys.path.insert(0, str(OUT))
from analyze_independent_distance import (FAMILIES, family_comembership,
                                          load_errors, percentile_ranks,
                                          perm_test_mean_rank,
                                          dyadic_regression)


def graded_family_distance(motifs):
    """d=1 same family; d=2 families share a member; d=3 otherwise (paper)."""
    idx = {m: i for i, m in enumerate(motifs)}
    n = len(motifs)
    fam_of = {m: set() for m in motifs}
    for k, fam in enumerate(FAMILIES):
        for m in fam:
            fam_of[m].add(k)
    D = np.full((n, n), 3.0)
    np.fill_diagonal(D, 0.0)
    members = {k: set(fam) for k, fam in enumerate(FAMILIES)}
    for a in motifs:
        for b in motifs:
            if a == b:
                continue
            if fam_of[a] & fam_of[b]:
                D[idx[a], idx[b]] = 1.0
            else:
                # families share a member?
                na = set().union(*(members[k] for k in fam_of[a])) if fam_of[a] else set()
                nb = set().union(*(members[k] for k in fam_of[b])) if fam_of[b] else set()
                if na & nb:
                    D[idx[a], idx[b]] = 2.0
    return D


def main():
    ind = json.loads((OUT / "independent_distance.json").read_text())
    res = json.loads((OUT / "independent_distance_results.json").read_text())
    motifs = ind["motifs"]
    GED = np.array(ind["ged_normalized"])
    FAM = family_comembership(motifs)
    FAMG = graded_family_distance(motifs)

    per_model, errors = load_errors(motifs)
    vis = [e for e in errors if e[1] in ("scene2motif", "skeleton2motif")]
    txt = [e for e in errors if e[1] in ("story2motif", "theorem2motif")]
    fr = [e for e in errors if e[0].startswith("api_")]

    # family-metric percentile ranks on identical splits (heavy ties -> midrank)
    res["error_concentration_family_metric"] = {
        "pooled": perm_test_mean_rank(errors, FAMG, n_perm=10_000),
        "vision": perm_test_mean_rank(vis, FAMG, n_perm=10_000),
        "text": perm_test_mean_rank(txt, FAMG, n_perm=10_000),
        "frontier": perm_test_mean_rank(fr, FAMG, n_perm=10_000),
    }
    for k, v in res["error_concentration_family_metric"].items():
        print(f"family-metric {k}: mean rank {v['mean_percentile_rank']:.3f} "
              f"p={v['p_perm_lower']:.4f}")

    # GED split table (already there for pooled/modality/frontier); add per model
    per_model_rank = {}
    for name in sorted(per_model):
        es = [e for e in errors if e[0] == name]
        if len(es) >= 10:
            r = percentile_ranks(es, GED)
            per_model_rank[name] = {"n": len(es),
                                    "mean_rank": float(r.mean())}
    res["error_concentration_ged_per_model"] = per_model_rank
    print("per-model GED mean ranks:",
          {k: round(v["mean_rank"], 3) for k, v in per_model_rank.items()})

    # row-percentile GED similarity: sim[i,j] = 1 - rank of j in row i
    n = len(motifs)
    R = np.zeros((n, n))
    for i in range(n):
        opts = [k for k in range(n) if k != i]
        row = GED[i, opts]
        for a, k in enumerate(opts):
            less = (row < row[a]).sum()
            eq = (row == row[a]).sum()
            R[i, k] = 1.0 - (less + (eq - 1) / 2.0) / (len(opts) - 1)
    pooledC = sum(m["C"] for m in per_model.values())
    pooledE = sum(m["expo"] for m in per_model.values())
    b, p, ne = dyadic_regression(pooledC, pooledE,
                                 {"family": FAM, "ged_rank_sim": R},
                                 n_perm=10_000)
    res["dyadic_regression"]["pooled_family_gedrank"] = {
        "beta": b, "p_perm": p, "n_errors": ne,
        "note": "ged_rank_sim = 1 - within-row percentile rank of the "
                "normalized typed GED (row-conditioned)"}
    print("family+gedrank:", {k: round(v, 3) for k, v in b.items()},
          {k: round(v, 4) for k, v in p.items()})
    b, p, ne = dyadic_regression(pooledC, pooledE, {"ged_rank_sim": R},
                                 n_perm=10_000)
    res["dyadic_regression"]["pooled_uni_gedrank"] = {
        "beta": b, "p_perm": p, "n_errors": ne}
    print("uni gedrank:", {k: round(v, 3) for k, v in b.items()},
          {k: round(v, 4) for k, v in p.items()})

    (OUT / "independent_distance_results.json").write_text(
        json.dumps(res, indent=1))
    print("updated", OUT / "independent_distance_results.json")


if __name__ == "__main__":
    main()
