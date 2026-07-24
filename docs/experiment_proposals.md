# GEB-Bench: Additional Experiments, Ranked by Impact/Effort

Context I'm working from: 165 items, 25 motifs, 5 tasks, 6 models; scenes are gpt-image renders conditioned on skeleton reference; all per-item logs (`pred`, `answer`, `raw`, `option_motifs`) already sit in `eval/results/*.json`. A full frontier-model run is ~165 calls ≈ **<$5**, so almost everything below is cheap in API terms — the ranking is dominated by engineering effort and reviewer weight.

---

## Tier 1 — Do these regardless (near-zero cost, direct reviewer defenses)

### 1. Menu ablation: ids-only vs full menu (and invariant-only)
**Reviewer attack:** "The 25 one-line invariants are a cheat sheet; the model just matches the item against the menu text. You're benchmarking menu-reading, not structure recognition."
**Hypothesis:** Removing invariants (ids-only menu) drops accuracy substantially on scene/story (where the model must supply the abstraction itself) but less on theorem (where the item text already is the invariant). If true, this *quantifies* how much of the task is abstraction vs. menu lookup — a finding, not just a defense. Third arm: invariants-only with anonymized ids (`M01…M25`) checks whether the English motif *names* (`strange_loop`, `mobius`) leak information.
**Protocol:** Add `--menu-variant {full,ids,anon}` to `run_eval.py` — it only changes the `MENU` string, items untouched. Run GPT-5.5 + GPT-5.2 + Qwen2.5-32B on the four recognition tasks (120 items × 2 variants × 3 models).
```bash
python3 eval/run_eval.py --base $YUNWU --model gpt-5.5 --vision \
  --tasks scene2motif,skeleton2motif,story2motif,theorem2motif \
  --menu-variant ids --tag gpt-5.5_menu-ids
```
**Exists / needs:** all data exists; ~30 lines of code.
**Cost:** ~$15 total. **Figure:** grouped bars, accuracy by task × menu-variant ("How much does the menu do?"). Goes in main paper.

### 2. Theorem-masking ablation (unmasked)
**Reviewer attack:** "Is the □-masking actually doing anything, or is `mask_names` (a 26-word blocklist, `build_items.py:69`) security theater?"
**Hypothesis:** Unmasked theorems ≈ ceiling (≥96%) even for Qwen2.5-7B, because name-matching against the menu is trivial. The masked–unmasked *gap per model* is then a clean measure of "reads names vs reads math," and the fact that the gap shrinks with scale (7B: 56%→~95%?) is itself a scaling result.
**Protocol:** In `build_items.py`, emit a parallel `theorem2motif_unmasked` task (skip `mask_names`); run all 6 models (25 items each).
**Exists / needs:** theorems exist in `motif_library.json`; ~10 lines.
**Cost:** <$3. **Figure:** paired dumbbell plot, masked vs unmasked per model. Also flag: run a quick n-gram overlap audit between masked theorems and menu invariants to preempt "residual leakage" — free.

### 3. Zero-API analyses on existing logs (error consistency + formal-distance correlation)
No new calls; pure analysis of the 6 result JSONs. Three sub-analyses:

- **(a) Formal distance vs confusion rate.** Define distance 1 = same confusable family (`FAMILIES` in `build_items.py`), 2 = share a family member, 3 = otherwise. Pool all 25-way errors, plot confusion count vs distance. You already claim "confusion geometry" qualitatively; this makes it a quantitative curve (predict: monotone, most mass at d=1). Add a permutation test: shuffle family assignments 10k×, show observed within-family error mass is >99th percentile. **This upgrades §4.2 from anecdote to statistic — the single highest-value/effort item in this list.**
- **(b) Error consistency across models.** Per-item correctness vectors → pairwise Cohen's κ above chance-agreement, plus "prediction overlap on shared errors" (when two models are both wrong, do they pick the *same* wrong motif?). If GPT-5.5 and Qwen2.5-32B make the same specific confusions, the boundaries are properties of the *items* (real structural ambiguity), not the models — a strong validity argument.
- **(c) Crossvoice position bias.** From `option_motifs` + `pred`: answer-letter distribution per model vs the balanced ground truth. Frees you from the "VLMs pick A" artifact question. Also compute crossvoice accuracy conditioned on whether the model got the same scene right in `scene2motif` — is crossvoice failure a *recognition* failure or a *mapping* failure? (If P(cross correct | scene correct) ≫ P(cross correct | scene wrong), crossvoice difficulty reduces to scene difficulty; if not, mapping is a separate skill. Either answer is a headline sentence.)

**Cost:** $0. **Figures:** distance-vs-confusion scatter/bar with permutation band; model-agreement κ heatmap. **Effort:** one afternoon, one script `eval/analyze_logs.py`.

### 4. Human baseline (small-N pilot)
**Reviewer attack:** guaranteed. Benchmark papers without a human row get asked for one in every review.
**Protocol:** N=5 humans (grad-student level, no GEB priming), each does a stratified 60-item subset (12 per task, covering all 25 motifs across annotators) under the *identical* prompt/menu shown to models, untimed, no lookup. You already have `build_site.py` — extend it to emit a static self-scoring quiz page (answers hashed), collect JSON responses. Report mean±range and per-task human accuracy as a table row + dashed line on the voice-gradient figure.
**Hypothesis worth stating in advance:** humans show the *same* voice gradient but a smaller scene↔skeleton gap — i.e., the abstraction gap is model-specific, not item-intrinsic. If humans also fail `self_reference` scenes, that's an item-quality signal you want *before* the camera-ready.
**Cost:** $0 API; ~1 day engineering + recruiting. **Figure:** human row in Table 1 + line on fig:gradient.

---

## Tier 2 — High value, modest new inference

### 5. New crossvoice pairings: scene↔theorem and story↔skeleton (voice-pair matrix)
**Why:** Currently "cross-voice" = one pairing. A skeptic asks whether crossvoice hardness is about *crossing voices* in general or about the specific scene–story pair. **All artifacts already exist** — masked theorems, stories, skeleton PNGs, scenes — so building `scene2theorem` (scene image + 4 masked theorems, family negatives) and `skeleton2story` (skeleton image + 4 stories) is a pure `build_items.py` extension reusing `negatives()`. That yields a 2×2 (or richer) *voice-pair difficulty matrix*.
**Hypothesis:** difficulty ≈ f(least-symbolic voice in the pair): scene↔theorem lands between crossvoice and theorem2motif; skeleton↔story easier than scene↔story. This turns the voice gradient into a *compositional* claim — a genuinely new figure, not just an ablation.
**Protocol:** 45 + 25 new items × 3 vision models.
**Cost:** ~$10. **Figure:** voice-pair heatmap (rows = image voice, cols = text voice, cell = GPT-5.5 accuracy). Main-paper candidate.

### 6. CoT / reasoning-effort ablation on the two hard tasks
**Hypothesis:** Explicit reasoning helps crossvoice (a two-step abstract-then-match problem) more than scene2motif (arguably perception-bound). A dissociation — thinking closes the mapping gap but not the abstraction gap — is a clean, quotable result.
**Protocol:** three arms on scene2motif + crossvoice (90 items), GPT-5.5 + GPT-5.2:
1. current direct-JSON prompt (exists),
2. "先用两三句话分析图像的抽象结构，再输出 JSON" (CoT-in-prompt; relax the parser to take the *last* JSON object),
3. API `reasoning_effort` low/high if the yunwu proxy exposes it.
**Cost:** ~$20 (CoT tokens are the expense). **Figure:** Δaccuracy from thinking, per task per model.

### 7. Distractor-menu hardness: 25-way vs 5-way family menu
**Hypothesis:** A 5-way menu restricted to the true motif's confusable family (via `negatives(mid, 4, rng)`) is *not much easier* than the 25-way menu despite the 5× chance floor — because §3 already shows errors concentrate in-family. If confirmed: "the 25-way menu is effectively a 4-5 way discrimination among formal neighbors" — beautiful validation of the design, and it converts analysis 3(a) into a causal test.
**Protocol:** rebuild the four recognition tasks with 5-way family menus (fixed seed), run GPT-5.5 + Qwen2.5-32B + Qwen2-VL-7B.
**Cost:** ~$8. **Figure:** 25-way vs 5-way accuracy, with chance-corrected (Cohen's κ vs floor) comparison so the two menu sizes are commensurable.

### 8. One-shot in-context example per voice
**Hypothesis:** One worked example (item + correct motif + one-line justification, using a motif *held out from the query*... except you only have 25 motifs — so use a *synthetic 26th demo motif* or demo on the item's own voice with a far-family motif and never score that motif's items in this arm) helps small open models a lot (format + task-framing) and frontier models little. This addresses "maybe Qwen fails at instruction-following, not abstraction" — important because Qwen2-VL-7B at 20% on skeleton2motif smells like format/parse failure. **First check the logs (free):** what fraction of Qwen2-VL-7B errors are parse failures (`pred=None`) vs wrong-but-valid ids? Report that number regardless.
**Cost:** ~$10. **Figure:** 0-shot vs 1-shot per model, ordered by scale.

---

## Tier 3 — Strong but real generation effort

### 9. Nuisance-variable robustness: resampled scene surfaces
**Reviewer attack:** "45 scene images, 1-2 per motif — per-motif scene accuracy is noise, and results could hinge on idiosyncratic renders."
**Protocol:** For 10 motifs (the 6 hardest + 4 easiest), regenerate 3 additional scenes each with resampled nuisance parameters (node counts/depths/angles — `gen_motifs.py` already treats these as recorded nuisance variables) *and* a different scene-idea from `scene_ideas`. Rerun scene2motif on GPT-5.5. Report per-motif accuracy mean±sd across surface variants and the intraclass correlation (is motif identity or surface draw the bigger variance component?).
**Exists / needs:** pipeline exists (`gen_motifs.py`); needs ~30 gpt-image generations + light QC that the structure survived rendering.
**Cost:** ~$5-15 image gen + $3 eval, plus half a day QC. **Figure:** per-motif accuracy with error bars over surface resamples — also retroactively justifies every per-motif claim in fig:hardest/fig:heatmap.

### 10. Broaden the model set (Claude, Gemini, Qwen2.5-VL-72B)
Not an ablation, but the #2 most predictable reviewer request after humans ("two proprietary models from one vendor + one 7B VLM"). Qwen2.5-VL-72B in particular fills the gaping hole in your vision scaling story — right now the "images unread at 7B" claim has no larger open-VLM counterpoint. Runs unchanged through the OpenAI-compatible path.
**Cost:** ~$10 API + GPU hours on kiti (72B may need quantization on 8×3090). **Figure:** rows in Table 1; extends fig:scaling to vision.

---

## Suggested paper packaging

| Item | Where |
|---|---|
| 1, 2, 6, 7, 8 | one "Ablations" section + one 4-panel figure |
| 3a (distance-vs-confusion) | promote into §4.2 as the quantitative backbone |
| 3b, 3c | short "Error analysis" subsection or appendix |
| 4 (human) | Table 1 row + gradient figure line |
| 5 (voice-pair matrix) | new results subsection — this is the one that adds a *claim*, not just a defense |
| 9 | appendix robustness figure |

**Execution order for a 2-week window:** 3 (today, free) → 1+2 (one day) → 5 (two days) → 4 launch recruiting in parallel → 6+7+8 (batchable in one day) → 9/10 as time allows.

One infrastructure note: items 1, 2, 5, 7, 8 all want variant item sets — I'd refactor `build_items.py` to take `--variant` and write `items_<variant>.json`, and `run_eval.py` to take `--items`, rather than forking scripts. Keeps every ablation reproducible from two flags.