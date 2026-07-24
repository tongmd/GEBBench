# Experiment ③ — The Psychophysical Function of Abstraction

A 7-level skeleton→photoreal **naturalization ladder**: the same formal motif
program rendered at continuously increasing surface entropy. Claim under
test: abstraction collapse is one mechanism with model-specific operating
points — psychometric curves share a slope, only the threshold moves.

## Ladder

| level | rendering | seeds |
|---|---|---|
| L0 | programmatic skeleton PNG (`motifs/<id>/skeleton_*.png`) | 1 |
| L1 | hand-media redraw of the diagram (pencil / ink / charcoal) | 3 (medium) |
| L2 | stylized flat-color infographic (3 style guides) | 3 (style) |
| L3 | minimal flat vector illustration, structure dominant | 3 (scene idea) |
| L4 | illustration embedded in a simple scene | 3 (scene idea) |
| L5 | naturalistic photo-like, moderate clutter | 3 (scene idea) |
| L6 | photoreal cluttered, structure must be found | 3 (scene idea) |

25 motifs × 6 generated levels × 3 seeds = 450 images (gpt-image-2
`/images/edits`, skeleton as reference image). Seeds vary the medium (L1–L2)
or the motif's `scene_ideas` entry from `data/motif_library.json` (L3–L6).
One prompt template per level (`gen_ladder.py::TEMPLATES`).

## Pipeline

```bash
python3 exp3_ladder/gen_ladder.py --workers 4        # generate (hours)
python3 exp3_ladder/check_structure.py               # blind describe+judge L5/L6
python3 exp3_ladder/check_structure.py --regen       # regenerate flagged once
python3 exp3_ladder/check_structure.py               # re-judge regenerated
python3 exp3_ladder/build_items.py                   # -> items.json
python3 exp3_ladder/run_ladder.py --model gpt-5.5    # resumable, merges results
python3 exp3_ladder/run_ladder.py --model gpt-5.2
python3 exp3_ladder/analyze_ladder.py                # stats + figure
```

`run_ladder.py` works against any OpenAI-compatible endpoint
(`--base URL --model NAME --key KEY`) — open VLMs run on kiti the same way.
The scene2motif prompt is byte-identical to `eval/run_eval.py`.

## Files

- `gen_ladder.py` — generation (concurrency 4, 3 retries, PNG validation;
  `manifest.json` all jobs, `failures.json` retryable failures — re-run the
  script to retry, it skips valid images)
- `check_structure.py` — independent structure check for L5/L6: a blind
  vision pass describes the composition in one sentence, a text-only judge
  compares to the motif invariant; `flags.json` records verdicts; flagged
  images are regenerated at most once
- `build_items.py` → `items.json` — same schema as `eval/items.json`
  (scene2motif, 25-way menu), one item per existing image incl. L0 skeletons
- `run_ladder.py` — resumable evaluator → `results/<model>.json`
- `analyze_ladder.py` — per-model logistic GLM (motif fixed intercepts),
  2-param psychometric sigmoid (threshold θ, slope β; guess floor 1/25),
  motif-cluster bootstrap θ CIs, slope×model LRT, per-motif θ vs
  confusable-family degree → `results.json`,
  `paper/figs/psychophysics.pdf`
