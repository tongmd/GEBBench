# catlab — functorial-abstraction pilot (naturality squares)

Implements the pilot from `proposal/functorial-abstraction.md`: does a model
that *recognizes* two structures also *transport the change* between them?

> Object recognition does not imply morphism preservation — testable claim.

## The g-library (6 edges, matched pairs)

Each edge `g: m -> m'` is a **program edit on one parametric base**
(`gen_pairs.py`): the two panels share style and, where the motif allows,
node counts and ratios; the only difference is the structural change.
Structural facts are asserted in code (e.g. the aperiodic strip is a
Fibonacci word, checked to admit **no** translation period; the broken
mirror has **exactly one** asymmetric node).

| edge | src → dst | program edit |
|---|---|---|
| sym_to_symbreak | symmetry → symmetry_breaking | displace exactly 1 mirrored node |
| cycle_to_strangeloop | cycle → strange_loop | re-lay same n nodes across n tiers + back-edge |
| nesting_to_recursion | nesting → recursion | branching factor 1 → 2, same rule/depth |
| tiling_to_aperiodic | tiling → aperiodic_tiling | (LS)^k → Fibonacci word, same tiles/rule |
| iso_to_duality | isomorphism → projective_duality | two systems + dictionary → one system, involutive swap |
| recursion_to_contraction | recursion → contraction_fixed_point | branching 2 → 1, orbit marks fixed point |

`nesting → recursion → contraction_fixed_point` composes (used by T2).

## Tasks (222 items, exact-match, judge-free)

- **endpoint_id (72)** — 25-way ID of each panel. The *denominator*: enables
  the conditional analysis and the shortcut ceiling.
- **square_thm (36)** — reference pair (BEFORE→AFTER image) + masked theorem
  of the BEFORE structure; pick the theorem of the AFTER structure among 4
  formally near motifs. No identity trap here (each motif has one theorem;
  a duplicate would be eliminable as text).
- **square_story (36)** — same, story voice; **includes the identity trap**:
  one option is a *different story of the unchanged BEFORE motif* (surface
  change, no structural change).
- **id_vs_change (72)** — skeleton pair; 4-way: which change happened?
  One option is always "no structural change, only rendering parameters";
  half the items are true nuisance pairs. Invariance vs equivariance.
- **composition (6)** — STEP 1 + STEP 2 shown; apply both to a theorem.
  Distractor set includes the "stopped after STEP 1" answer (recursion).
  Deliberately: STEP 1's AFTER and STEP 2's BEFORE use *different renderings*
  of recursion, so pixel-chaining cannot solve it.

## Pre-registered readouts (analyze_catlab.py)

1. `P(square | both endpoints identified)` vs `P(square | endpoint missed)`
   — recognition vs transport dissociation.
2. **Shortcut ceiling** from the model's own endpoint_id answers (pick the
   option whose motif = model's read of the AFTER panel). K3: if
   `square_acc <= ceiling`, transport reduces to recognition.
3. **Identity pull** (square_story): share of errors choosing the unchanged
   motif, vs 1/3 chance.
4. `id_vs_change`: nuisance false-alarm rate (calls a nuisance pair a
   structural change) vs structural hit rate.
5. Composition: `P(composite | both steps right)` + the stopped-after-g1
   error share (FVR reading at n=6, exploratory only).

Run: `run_catlab.py` (OpenRouter, same protocol as the main benchmark:
greedy, JSON answer, one retry pass for unparsed). Fleet: `run_catlab_all.sh`
(6 vision models). Results → `catlab/results/`, analysis → `analysis.json`.
