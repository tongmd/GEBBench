# Exp1 — Formal Geometry, Not Pixel Geometry

Regression race between **formal distance** and **perceptual distance** as
predictors of VLM confusion errors on GEB-Bench scenes
(plan: `docs/iclr_experiment_plan.md` §①).

## TL;DR

- **Decoupling pre-check PASSED**: Mantel r(DINOv2-perceptual, formal-family)
  = **+0.022** (p = 0.70) — far below the 0.2 kill threshold. The GEB render
  pipeline experimentally decouples formal structure from pixel appearance.
- **Univariate race (pooled, n = 86 errors)**: formal same-family
  β = **+0.362 (p = .034)** beats perceptual DINOv2 β = +0.221 (p = .28,
  n.s.). Label-prior β = +0.439 (p = .019) is comparably strong.
- **Multivariate race is underpowered** at n = 86 pooled errors (plan calls
  for n ≥ 400 via nuisance-resample eval runs): formal-family β = +0.253
  (p = .21) vs perceptual β = +0.234 (p = .25); neither clears α alone.
- **Kill-shot control is flat** for all 3 models (|r| ≤ .15, all p > .24):
  within a motif, an item's perceptual distance to its own motif centroid
  does NOT predict per-item accuracy — errors are not driven by "weird
  renders".
- **Caveat**: the secondary hand-coded skeleton-GED predictor comes out
  *negative* (univariate β = −0.429, p = .06) — errors do not track this
  particular graph reduction; treat GED as a failed operationalization, not
  evidence against the formal story (the pre-registered primary predictor is
  binary same-family).
- 40.7% of all confusion errors (35/86) land inside a confusable family,
  vs a 21.3% same-family dyad base rate (~2.3× enrichment; frontier models
  ≈ 45%).

**One-sentence verdict:** with the perceptual and formal spaces successfully
decoupled (Mantel r = .02), pooled VLM confusions are organized by formal
family (β = +.36, p = .03) and label-space priors (β = +.44, p = .02) but not
by DINOv2 perceptual distance (β = +.22, n.s.) — formal wins the univariate
race, and the multivariate verdict awaits the n ≥ 400 error events from the
nuisance-resample eval runs.

## Files

| file | role |
|---|---|
| `gen_renders.py` | nuisance-resample generation: brings every motif to ≥5 renders via gpt-image-2 `/images/edits` (skeleton PNG as reference, prompts = `scene_ideas` × varied media/palette), 3 retries, PIL-validated |
| `renders/<motif>/resample_XX.png` | 80 new renders (7 motifs ×4, 16 ×3, 2 ×2) |
| `embed_remote.py`, `run_embeddings.sh` | DINOv2-base (CLS) + CLIP ViT-L/14 (projected pooler) embeddings of all 125 renders, run on kiti GPU 0 (`~/micromamba/envs/interp`, proxy env needed non-interactively; torchvision 0.21 installed for transformers 5.x) |
| `embeddings.npz` | `paths` (125), `dino` (125×768), `clip` (125×768) |
| `formal_graphs.py` | hand-coded reduction of each motif's `skeleton_spec` to a small DiGraph (≤10 nodes; node attr `c` encodes state/role; repetition counts minimized — reductions documented inline) |
| `build_formal_matrices.py` | 25×25 matrices → `matrices_formal.json`: `formal_family` (binary, verbatim from `eval/build_items.py:34` FAMILIES — fixed by skeleton specs before any render existed), `formal_ged` (networkx GED, 30 s timeout, normalized by combined graph size), `label_prior` (text-embedding-3-small over `name_en + invariant_zh`, cosine distance) |
| `analyze.py` | Mantel pre-check (10k node perms) → dyadic Poisson regression with log-exposure offset, 4 standardized similarity predictors, 10k QAP node-permutation p-values, per-model + pooled + CLIP robustness → kill-shot control → `results.json` |
| `figures.py` | `paper/figs/exp1_confusion_vs_distance.pdf`, `paper/figs/exp1_betas.pdf` |
| `results.json` | all numbers below |

## Protocol details & choices

- **Stimuli**: 45 existing `motifs/*/scene_*.png` + 80 new resamples = 125
  images, exactly ≥5 per motif (min = 5). New prompts recombine the motif's
  three `scene_ideas` with 5 media/palette axes (photo-daylight,
  golden-hour, ink-wash, studio-dark, woodblock), skeleton as reference.
- **Perceptual matrix**: per-motif centroid of L2-normalized embeddings,
  cosine distance between centroids. DINOv2 primary, CLIP for robustness
  (CLIP-substituted pooled regression in
  `results.json:regression_clip_robustness`; same qualitative pattern).
- **Formal matrices**: primary = binary same-family (pre-registered);
  secondary = GED between hand-coded graphs. GED reductions are approximate
  by design; the family↔GED Mantel r is only +0.146, i.e. the hand-coded GED
  is a poor proxy for the design families — see caveat above.
- **Label-prior matrix**: separates "mis-seeing structure" from
  "label-space prior": it is the strongest single predictor for
  qwen2-vl-7b (β = +0.548, p = .03), consistent with a near-chance model
  answering from name priors.
- **Response data**: pooled `scene2motif` + `skeleton2motif` errors from
  `eval/results/*.json`. Vision records exist for gpt-5.2 (18 errors),
  gpt-5.5 (15), qwen2-vl-7b (53). qwen2.5-vl-32b had not landed at run
  time; `analyze.py` picks up any new results file automatically — rerun
  `python3 exp1_geometry/analyze.py` when it does.
- **Dyadic model**: 600 ordered motif pairs, y = error count(i→j),
  offset = log(#items of motif i), predictors standardized similarities
  (family-same, −GED, −DINOv2, −label). β > 0 ⇒ closeness in that space
  drives confusion. p-values from 10k joint node permutations (QAP).

## Key numbers (results.json)

Decoupling (Mantel, 10k perms):

| pair | r | p |
|---|---|---|
| DINOv2 vs formal-family | **+0.022** | 0.704 |
| DINOv2 vs formal-GED | +0.058 | 0.582 |
| CLIP vs formal-family | +0.069 | 0.230 |
| DINOv2 vs label-prior | +0.220 | 0.002 |

Multivariate dyadic Poisson β (10k-perm p):

| model | n_err | formal-family | formal-GED | perceptual-DINOv2 | label-prior |
|---|---|---|---|---|---|
| gpt-5.2 | 18 | +0.33 (.34) | −0.18 (.51) | +0.31 (.27) | +0.26 (.43) |
| gpt-5.5 | 15 | +0.43 (.22) | −0.09 (.74) | +0.23 (.41) | +0.10 (.76) |
| qwen2-vl-7b | 53 | +0.18 (.40) | −0.78 (.02) | +0.21 (.38) | +0.55 (.03) |
| **pooled** | 86 | **+0.25 (.21)** | −0.57 (.03) | +0.23 (.25) | +0.42 (.05) |

Univariate pooled (10k-perm p): family **+0.362 (.034)**, GED −0.429 (.060),
DINOv2 +0.221 (.279), label +0.439 (.019).

Kill-shot (within-motif correctness vs distance-to-own-centroid, LOO):
gpt-5.2 r = +0.04 (p = 1.0), gpt-5.5 r = −0.15 (p = .25),
qwen2-vl-7b r = −0.02 (p = 1.0) — flat, as required.

## Reproduce

```bash
python3 exp1_geometry/gen_renders.py          # ~1 h, ~80 gpt-image-2 calls
bash    exp1_geometry/run_embeddings.sh       # kiti GPU 0, ~2 min
python3 exp1_geometry/build_formal_matrices.py
python3 exp1_geometry/analyze.py              # ~5 min (10k perms)
python3 exp1_geometry/figures.py
```
