#!/usr/bin/env python3
"""Supplement: 3-predictor races (formal family + label prior + ONE perceptual
embedding at a time) to avoid CLIP-SigLIP collinearity (Mantel r=0.83), plus
the reverse test: does formal family survive when entered after each
perceptual predictor? Appends to perceptual_race.json.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

import importlib.util
spec = importlib.util.spec_from_file_location(
    "pr", "/Users/stone/GEB-benchmark/eval/analysis_out/perceptual_race.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)  # __name__ == "pr", so main() does not run

ROOT = Path("/Users/stone/GEB-benchmark")
EXP = ROOT / "exp1_geometry"
OUT = ROOT / "eval" / "analysis_out"

F = json.loads((EXP / "matrices_formal.json").read_text())
motifs = F["motifs"]
fam = np.array(F["formal_family"])
lab = np.array(F["label_prior"])

embs = pr.load_embeddings()
D = {n: pr.centroid_dist(*embs[n], motifs) for n in pr.EMB_NAMES}

models = pr.load_confusions(motifs)
pooledC = np.zeros_like(fam)
pooledE = np.zeros(len(motifs))
for d in models.values():
    pooledC += d["C"]
    pooledE += d["expo"]

res = json.loads((OUT / "perceptual_race.json").read_text())
res["race_one_perceptual_at_a_time"] = {}
for name in pr.EMB_NAMES:
    preds = {"formal_family": 1.0 - fam,
             "label_prior": -lab,
             name: -D[name]}
    b, p, ne, dv = pr.dyadic_regression(pooledC, pooledE, preds)
    res["race_one_perceptual_at_a_time"][name] = {
        "beta": {k: round(v, 4) for k, v in b.items()},
        "p_perm": {k: round(v, 5) for k, v in p.items()},
        "deviance_explained": round(dv, 4)}
    print(f"3-way race w/ {name}:")
    for k in preds:
        print(f"   {k:14s} beta={b[k]:+.3f} p={p[k]:.4f}")

# pairwise: formal family vs each perceptual head-to-head (2 predictors)
res["race_family_vs_each_perceptual"] = {}
for name in pr.EMB_NAMES:
    preds = {"formal_family": 1.0 - fam, name: -D[name]}
    b, p, ne, dv = pr.dyadic_regression(pooledC, pooledE, preds)
    res["race_family_vs_each_perceptual"][name] = {
        "beta": {k: round(v, 4) for k, v in b.items()},
        "p_perm": {k: round(v, 5) for k, v in p.items()},
        "deviance_explained": round(dv, 4)}
    print(f"head-to-head family vs {name}: "
          f"fam={b['formal_family']:+.3f}(p={p['formal_family']:.4f}) "
          f"{name}={b[name]:+.3f}(p={p[name]:.4f})")

(OUT / "perceptual_race.json").write_text(json.dumps(res, indent=1))
print("updated", OUT / "perceptual_race.json")
