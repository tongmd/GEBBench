# GEB-Bench × GemmaScope: sparse-feature interpretability experiment

**Claim to test:** the "voice gradient" and "confusion geometry" measured
behaviorally in GEB-Bench are visible *inside* an open model's sparse
features — before any answer token is produced.

## Setup

- **Model:** `google/gemma-2-2b` (base, bf16 ≈ 5 GB — trivial on the RTX 3090).
  Chosen over gemma-2-9b because GemmaScope covers *every* residual layer of
  2b with canonical 16k-width JumpReLU SAEs, the whole run is 75 short
  documents (minutes either way), and 2b leaves headroom to hold 6+ SAEs
  resident for a single-pass layer sweep. If the 2b signal is null, rerun
  with `--model google/gemma-2-9b --sae-release gemma-scope-9b-pt-res-canonical`
  (42 layers, bf16 ≈ 18 GB, still fits; adjust `--layers`).
- **SAEs:** `gemma-scope-2b-pt-res-canonical`, `layer_{L}/width_16k/canonical`
  (JumpReLU, trained on the post-block residual stream; "canonical" = the
  L0≈100 variant). Loaded via `sae_lens`; residuals taken from HF
  `hidden_states[L+1]` (= TransformerLens `resid_post[L]`), BOS dropped.
- **Inputs:** 25 motifs × 3 voices = 75 documents.
  - story — `eval/items.json` story2motif texts (byte-identical to the
    behavioral eval)
  - theorem — `eval/items.json` theorem2motif texts, i.e. the **masked**
    theorems (motif names → □), so lexical name overlap cannot fake
    voice-invariance
  - invariant — `data/motif_library.json` `invariant_zh`
- **Language: Chinese (primary).** Three reasons: (1) the voice contrast must
  hold language constant, and zh is the only language in which *all three*
  voices exist — translating theorems ourselves would inject a wording
  confound; (2) zh texts are exactly the behavioral eval items, which is what
  licenses the RQ2 comparison against the confusion matrix; (3) Gemma-2's
  256k SentencePiece vocab and multilingual pretraining handle CJK, and a
  voice-invariance result *in Chinese* on English-heavy SAEs is the stronger
  finding. `--lang en` swaps in `story_en` as a robustness check (note: en
  story vs zh theorem confounds voice with language — report it only as
  supporting, never primary).
- **Aggregation:** per-token SAE activations → document vector by max-pool
  (primary; motif signal is expected to be sparse/localized) and mean-pool
  (both saved). Top-30 features per token saved sparsely for the probe.

## Metrics ↔ research questions

| RQ | Metric | Test |
|----|--------|------|
| 1. voice-invariant motif features? | 25×25 story×theorem feature-cosine (and top-50 Jaccard): diagonal (same motif) vs off-diagonal (different motif); also story→theorem retrieval@1 (chance 4%) | permutation test, 10k label shuffles |
| 2. confusion geometry in feature space? | story-voice 25×25 cosine matrix vs (a) pooled behavioral confusion (story2motif + theorem2motif records from `eval/results/*.json`, row-normalized, symmetrized) and (b) family co-membership counts from `eval/build_items.py` FAMILIES | Mantel test (Spearman, 10k perms) |
| 3. which layer? | RQ1 gap + retrieval across layers 4,8,12,16,20,24 | sweep figure |
| 4. (optional) decodable? | 25-way logistic probe on token-level SAE features: train on story tokens, test on theorem documents | cross-voice accuracy vs 4% chance |

## Predictions

- **Supports the voice-invariant-representation claim:** diagonal cosine/Jaccard
  significantly above off-diagonal (p<0.01), retrieval@1 well above 4%
  (≥20% would be striking), peaking in mid-to-late layers (~12–20 of 26,
  where abstract/semantic features typically live), and cross-voice probe
  above chance.
- **Supports the confusion-geometry claim:** Mantel r>0 (p<0.05) between SAE
  story-feature similarity and behavioral confusion, with confusable-family
  pairs visibly right-shifted in the similarity distribution. Note the
  behavioral confusion comes from *other* models (Qwen, GPT), so a positive
  result says the geometry is a property of the *stimulus structure as
  encoded by LMs generally*, not one model's quirk.
- **Refutes / bounds the claim:** diagonal ≈ off-diagonal at every layer
  (models match voices only via surface statistics elsewhere); or overlap
  driven entirely by generic topic/register features — check by inspecting
  the top shared features on Neuronpedia
  (`neuronpedia.org/gemma-2-2b/{L}-gemmascope-res-16k/{feature}`); or Mantel
  r≈0 while family→confusion Mantel is positive (geometry exists behaviorally
  but not in this model's features).
- **Confound to report:** invariant↔theorem overlap may be inflated by shared
  mathematical register; story↔theorem is the load-bearing contrast precisely
  because register, vocabulary, and genre are maximally different.

## Files

- `run_gemmascope.py` — GPU extraction; saves `results/acts_{tag}.npz`,
  `meta_{tag}.json`, `quickmetrics_{tag}.json` (immediate readout).
- `analyze_interp.py` — CPU analysis; figures + `metrics_{tag}.json`.
- `requirements.txt`

## Running

```bash
pip install -r requirements.txt
huggingface-cli login          # gemma-2 is gated

# smoke test (~2 min incl. downloads of 1 SAE): 2 motifs, 1 layer
python run_gemmascope.py --smoke

# full run: 75 docs × 6 layers — ~10–20 min, dominated by SAE downloads (~2 GB)
python run_gemmascope.py --layers 4,8,12,16,20,24

# analysis (CPU, anywhere)
python analyze_interp.py --tag zh --probe

# robustness: English stories
python run_gemmascope.py --lang en --layers 12,16,20 && python analyze_interp.py --tag en
```

`--dry-run` checks data loading without torch (works on the Mac).

## Paper section this becomes

*"The confusion geometry is visible in the model's sparse features before any
answer is produced."* Figure panel: (a) overlap bars — same-motif cross-voice
feature overlap beats different-motif, story↔theorem included; (b) scatter —
SAE feature similarity between motif stories predicts which motifs *other*
models behaviorally confuse (Mantel r, p); (c) layer sweep — the
voice-invariant signal emerges in middle layers. This upgrades GEB-Bench from
a behavioral benchmark to one with a mechanistic correlate: the abstract
motifs are not just gradeable, they are *linearly present as sparse features*,
and the benchmark's hand-designed confusable families recover the model's own
internal similarity structure.
