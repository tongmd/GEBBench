# Experiment ② — Do VLMs Compare Spontaneously?

Comparison-facilitates-abstraction test (Gentner & Namy near-analog signature),
GEB-Bench ICLR plan §②. Within-item design over the 25 motifs; same 25-way
menu and JSON-answer protocol as `eval/run_eval.py`.

## Conditions (5 per motif, 125 items per model)

| cond | stimulus | prompt |
|---|---|---|
| C1 single | target scene alone | verbatim `scene2motif` prompt |
| C2 near pair | target + nuisance-resample of the SAME motif (similar surface) | pair prompt (identical C2–C4) |
| C3 far pair | target + same-motif scene with maximally different medium/domain | pair prompt |
| C4 foil pair | target + surface-similar scene of a DIFFERENT motif from its confusable family (`eval/build_items.py FAMILIES`) | pair prompt |
| C5 instructed | identical stimulus to C3 | pair prompt + explicit "compare the two panels, find the shared structure" step |

Pairs are composited side by side with PIL (equal heights, 24 px white
gutter), target always the LEFT panel. All pair prompts ask for the motif of
the LEFT panel only (统一指令、单图计分), so the foil condition is scored
without ambiguity and any use of the right panel in C2–C4 is *spontaneous*.
Scoring is always a single `motif_id` == the target's motif.

## Files

- `design.py` — the full per-motif design table (target / far / foil choices,
  generation prompts for missing near/far scenes)
- `gen_stimuli.py` — generates 25 near renders + 7 far renders (gpt-image-2,
  skeleton-reference pattern from `gen_motifs.py`) and composes the pair
  images into `stimuli/`; idempotent
- `build_items.py` → `items.json` (25 motifs × 5 conditions)
- `run_comparison.py` — OpenAI-compatible runner
  (`--base/--model/--conditions/--limit/--workers/--tag`); works with the
  yunwu proxy now and a kiti vLLM tunnel for open VLMs later
- `analyze_comparison.py` → `results.json` +
  `paper/figs/comparison_effect.pdf` (Δaccuracy dot-and-CI ladder, exact
  within-item McNemar tests, money contrasts, verdict)

## Reproduce

```bash
python3 exp2_comparison/gen_stimuli.py        # only regenerates missing files
python3 exp2_comparison/build_items.py
python3 exp2_comparison/run_comparison.py --base "$OPENAI_BASE_URL" --model gpt-5.5 --tag api_gpt-5.5
python3 exp2_comparison/run_comparison.py --base "$OPENAI_BASE_URL" --model gpt-5.2 --tag api_gpt-5.2
python3 exp2_comparison/analyze_comparison.py
```

## Statistics

- per-condition accuracy (n=25)
- exact McNemar (binomial on discordant pairs) for each condition vs single
- money contrasts: near vs far (== the (near−single)−(far−single) interaction,
  since both contrasts share the single baseline), foil vs single (does bad
  comparison HURT?), instructed vs far (machinery present but not spontaneous?)
- Δaccuracy paired bootstrap 95% CIs (10k resamples over motifs)

n=25/cell is thin (pre-registered risk in the plan); treat p-values as
directional evidence and extend with a second item-set + open VLMs
(Qwen2.5-VL-72B/7B, InternVL2.5-8B via kiti) before the paper claims.

## Results

_(filled by `analyze_comparison.py`; see `results.json`)_

## Known limitations

- Target is always the left panel (position not counterbalanced).
- C1 wording differs minimally from C2–C4 (single image vs pair framing);
  all pair-vs-pair contrasts are under literally identical wording.
- Far pairs for 18 motifs reuse existing scenes; 7 were newly generated —
  surface distance is designed, not measured (DINOv2 check possible via
  experiment ① machinery).
