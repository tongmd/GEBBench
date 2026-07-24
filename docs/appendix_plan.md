I've read the paper, library JSON, motif assets, items, results, and eval code. Key findings that shape the plan: the canonical stories live in `data/stories_v2.json` (with `story_zh`, `story_en`, `why_shape`, and `judge.device_checks` — perfect for the appendix; `motifs/*/story.txt` exists for only 4 motifs), the prompts are hardcoded in `eval/run_eval.py:35-54`, crossvoice items carry `option_motifs` maps (so wrong letters resolve to motif ids), and GPT-5.5's 33 errors include a spectacular case — **self_reference fails in all four voices**, each time to a different formal neighbor.

# GEB-Bench Appendix Expansion Plan

Current appendix: §A Reproducibility, §B Catalog (~2.5 pp). Plan adds §C–§H, ~14–18 new pages. All new content compiles under the existing xeCJK setup; per the paper's no-Chinese-in-body convention, appendix defaults to **English translations with Chinese shown only where the form device is character-level and untranslatable** (with explicit caveat that devices are verified on the Chinese text).

---

## §C — Three Motifs, Four Voices: Full Worked Examples (~4.5 pp)

One full-page spread per motif, showing every voice of the same motif side by side. Choose motifs to span the arc: one self-referential, one iterative, one combinatorial — and all three reappear in §D/§E, so the appendix cross-references itself.

**Motifs and assets:**

1. **`self_reference`** (hardest motif, §E's star)
   - Skeleton: `motifs/self_reference/skeleton_frames.png`
   - Scenes: `motifs/self_reference/scene_droste_study.png`, `scene_mountain_temple.png`
   - Story: `data/stories_v2.json` entry `motif_id=self_reference` — use `story_en`; show the acrostic in a Chinese-character strip (see §D)
   - Math: `motifs/self_reference/math.md` (diagonal lemma + Kleene SRT) — translate to English for the appendix rendering
2. **`recursion`** (already the teaser motif; 3 scenes demonstrate the nuisance contract)
   - Skeleton: `motifs/recursion/skeleton_tree.png`
   - Scenes: `motifs/recursion/scene_river_delta.png`, `scene_lightning.png`, `scene_winter_tree.png`
   - Story + math: `data/stories_v2.json` (`recursion`), `motifs/recursion/math.md`
3. **`interference`** (body §3.3 already describes its lcm(3,4) device — pay it off here)
   - Skeleton: `motifs/interference/skeleton_ripples.png`
   - Scenes: `motifs/interference/scene_two_ripples.png`, `scene_moire_curtains.png`
   - Story + math: `data/stories_v2.json` (`interference`), `motifs/interference/math.md`

**LaTeX structure:** one `figure*`-style full page per motif built with two `minipage` rows: top row = skeleton + scenes (`\includegraphics`, ~0.3\linewidth each); middle = a `tcolorbox`/framed parbox with the English story (footnote: "device verified on the Chinese original; see Appendix D"); bottom = math voice as a compact `description` block (Signature equation / Definition / Home / Theorem, from `math.md`, translated). Add a small "invariant" banner line at top (`invariant_zh` translated to English) so the reader knows what all four voices share.

---

## §D — Form-Device Verification Walkthroughs (~3 pp)

The "verifiable structured text" contribution (§1 contribution 3) currently has zero worked evidence. Show **three devices** with the mechanical check executed on the page:

1. **Acrostic fixed point (`self_reference`)** — the crown jewel. Reproduce `story_zh` from `data/stories_v2.json` with each sentence numbered and its **first character highlighted** (`\textcolor{red}` + bold), then a one-line strip: the 13 extracted characters 本篇每句首字连读恰为此标题 aligned character-by-character against the title, each with an English gloss ("this / piece's / every / sentence's / first / character / read-in-order / spells / exactly / this / title"). Then step (2): the title asserts exactly the fact step (1) verified → single-layer fixed point. Close by quoting the machine judge output verbatim from the JSON `judge.device_checks` field in a `verbatim` block. This is the one place Chinese is irreducible — say so explicitly.
2. **Crab canon (`symmetry`)** — from `data/stories_v2.json` (`motif_id=symmetry`). Two-column `tabular`: sentence *k+1−j* | sentence *k+1+j* (English translation), with the name-swap σ(A)=B marked; center sentence highlighted as the fold axis.
3. **lcm beat (`interference`)** — a 24-row compact table (or a 1×24 tick-strip TikZ/table): columns = sentence index, "opens with fixed word?" (period 3), "closes with 7-char clause?" (period 4); rows 12 and 24 highlighted as the double hits. English translations of the marked sentences beneath.

For each: end with the dual-screen record (`judge.device_passed`, `judge.blind_motif`) quoted from the JSON. **LaTeX:** numbered story via `enumerate` with tight spacing; checker clauses as an itemized "Checker specification" box; judge output in `\small\begin{verbatim}`.

---

## §E — Error Case Studies: Confusion Geometry Up Close (~4 pp)

Turn the paper's central claim — errors track *formal* proximity — into visible evidence. All predictions from `eval/results/api_gpt-5.5.json` (33 wrong of 165).

**Case study 1 (full page): self-reference fails in every voice, each to a different neighbor.**
| Voice | Item | GPT-5.5 pred |
|---|---|---|
| scene | `scene:scene_droste_study:self_reference` (`motifs/self_reference/scene_droste_study.png`) | `recursion` |
| scene | `scene:scene_mountain_temple:self_reference` | `recursion` |
| skeleton | `skel:self_reference:skeleton_frames` (`skeleton_frames.png`) | `nesting` |
| story | `story:self_reference` | `convergence` |
| crossvoice | `xv:scene_droste_study:self_reference` | picked C |

Layout: 2×2 image/text grid, each cell stamped with ground truth vs. prediction (red). Analysis paragraph: every wrong answer is in the designed confusable family {self_reference, strange_loop, recursion, nesting} except the story→convergence one — and the story error is exactly the failure mode §3.3 predicts (the model reads the *plot* — clues converging on the lost child — instead of the *telling* — the acrostic). This single case carries the voice-gradient and confusion-geometry claims simultaneously.

**Case study 2: projective duality ↔ isomorphism (the boundary the blind judge also stumbled on).**
`scene:scene_spiderweb_dewfall:projective_duality → isomorphism`, and `xv:scene_spiderweb_dewfall:projective_duality → C` where `option_motifs[C] = symmetry` (family member). Show `motifs/projective_duality/scene_spiderweb_dewfall.png`, quote the two motifs' menu invariants side by side (English), one paragraph on "role swap within one system vs. dictionary between two."

**Case study 3: fixed point read as arrival.** `scene:scene_you_are_here:contraction_fixed_point → convergence` (`motifs/contraction_fixed_point/scene_you_are_here.png`). The map-with-you-are-here-dot *is* Banach's theorem; the model sees "things ending up at a point."

**Case study 4: the global invariant.** `skel:aperiodic_tiling:skeleton_penrose → symmetry` (`motifs/aperiodic_tiling/skeleton_penrose.png`) — even on the explicit skeleton, local 5-fold patches read as symmetry; aperiodicity is unverifiable from any patch (ties to §4.4 hardest-motifs).

**Case study 5 (half page, table only): crossvoice errors resolve to family members.** Small table of all 12 GPT-5.5 crossvoice errors: item id → predicted letter → `option_motifs[letter]` → in-family? (from `eval/items.json`). Punchline: count how many wrong picks land inside the confusable family — the distractor design is what the errors chose.

**LaTeX:** per case, a `figure` with image(s) + a shaded `tcolorbox` "Model output" containing the raw JSON reply from the `raw` field (verbatim), then analysis text. Cross-reference `Figure~\ref{fig:confusion}`.

---

## §F — Exact Evaluation Prompts and Option Menu (~1.5 pp)

Verbatim reproduction of `PROMPTS` from `eval/run_eval.py:35-54` — all five templates, shown as **original Chinese in `verbatim` + English translation immediately below each** (the translation is essential since the body never shows the prompts). Plus:
- The 25-way option menu (from `eval/items.json` → `menu`): full listing, **English translations of all 25 invariant lines** in a two-column `longtable` (`motif id | invariant (EN)`), with 2–3 original Chinese lines shown as a formatting sample. This doubles as the English rendering of the invariants, which the catalog table lacks.
- Answer parsing + failure handling: 3–4 sentences citing `parse()` (`run_eval.py:80-90`) — JSON field regex, A–D letter fallback, unparseable = wrong; temperature 0, max_tokens 4096.

**LaTeX:** `\small\begin{verbatim}` blocks per template, `longtable` for the menu.

---

## §G — A Complete Cross-Voice Item (~1.5 pp)

Reproduce **one full crossvoice item end-to-end** — the task the body calls hardest but never shows. Use `xv:scene_moire_curtains:interference` (reuses §C's motif; GPT-5.5 got it wrong → picked D): the image `motifs/interference/scene_moire_curtains.png`, then all four candidate stories as English translations (from `data/stories_v2.json`, matched via the item's `option_motifs`), each labeled A–D with its motif id revealed in the caption *after* the stories, then GPT-5.5's raw reply and the correct answer. Reader gets to play the item themselves before seeing the key. **LaTeX:** figure at top, four framed parboxes, answer key + model output in a final `tcolorbox`.

---

## §H — Skeleton and Scene Galleries (~2 pp)

- **All 25 skeletons** in a 5×5 grid (`motifs/<id>/skeleton_*.png` — one per motif dir), each labeled with motif id. Complements Fig. 2 (scenes) in the body: the reader sees literally what the "recognition ceiling" condition shows the model. 1 page.
- **Nuisance-variable demonstration**: rows of same-motif/different-surface scenes — recursion's 3 scenes (`river_delta`, `lightning`, `winter_tree`), self_reference's 2, figure_ground's 2 (`scene_day_night_birds.png`, `scene_papercut.png`), interference's 2 — with a caption enumerating exactly which parameters differ (branch count, depth, medium) and that none are scored. If body Fig. `teaser_recursion` already covers recursion, drop that row and keep the other three. 1 page. **LaTeX:** `tabular` of `\includegraphics` cells.

---

## Build notes

- One small script (extend `eval/make_additional_figures.py` or new `paper/make_appendix.py`) should **generate the story/menu/error LaTeX from the JSONs** (`data/stories_v2.json`, `eval/items.json`, `eval/results/api_gpt-5.5.json`) rather than hand-copying — keeps the "every table reproducible from `analyze.py`" claim true for the appendix too.
- English invariant/math translations don't exist yet in the data — they're the only genuinely new writing (≈25 invariant lines + 3 math.md entries + prompt translations). Consider adding `invariant_en` to `motif_library.json` so the artifact matches the paper.
- Update the Limitations paragraph pointer: "the released tales are not yet multilingual" → note that reference English translations now appear in Appendix D/G (translations are for readers, not eval items).
- Ordering: keep A (repro) and B (catalog) first, then C worked examples → D device walkthroughs → E error cases → F prompts → G crossvoice item → H galleries.

**Total new: ~16 pages**, dominated by figures/verbatim (fast to produce), with §C+§E as the priority pair if space is cut.