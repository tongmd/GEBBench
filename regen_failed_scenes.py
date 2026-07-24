#!/usr/bin/env python3
"""Regeneration round for failed scenes.

For every 'fail' verdict in data/scene_screen_v2.json:
  1. gpt-5.5:stable rewrites the scene prompt, given the original prompt AND
     the screener's failure reason, with a motif-specific structural mandate.
  2. the failed png is deleted; the new prompt replaces the old tuple in
     data/scene_prompts_v2.json (same name, so gen_scenes_expand regenerates).

Then run:  python3 gen_scenes_expand.py   (regenerates deleted files)
Then:      python3 verify_scenes_v2.py    (screens only new files)

Usage: python3 regen_failed_scenes.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gen_stories as G  # chat + parse_json

G.WRITER_MODEL = "gpt-5.5:stable"

ROOT = Path(__file__).parent
LIB = {m["id"]: m for m in
       json.loads((ROOT / "data" / "motif_library.json").read_text())}
SCREEN = json.loads((ROOT / "data" / "scene_screen_v2.json").read_text())
PROMPTS_PATH = ROOT / "data" / "scene_prompts_v2.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text())

# structural mandates distilled from the screening failure patterns
MANDATES = {
    "strange_loop": "The composition MUST close the loop: a chain of levels "
        "that visibly ascends (stairs, frames, balconies, screens) whose "
        "TOPMOST level connects back to the BOTTOM one by a continuous, "
        "traceable element (a staircase that lands where it began, a screen "
        "at the top showing the bottom room, an Escher-style impossible "
        "return). The closed circuit must be the dominant compositional line.",
    "mobius": "The band MUST show one clearly visible HALF-TWIST: at one "
        "point of the loop the strip's front face turns into its back face "
        "(show both faces with different color/texture so the swap is "
        "unmistakable), like a true Mobius strip. Not a flat closed ring.",
    "figure_ground": "The negative space MUST itself form recognizable "
        "shapes: dark forms and light forms share the same contour and "
        "tessellate (Escher-style), e.g. dark birds whose gaps are light "
        "fish. Both readings must work; neither is mere background.",
    "isomorphism": "Show TWO panels of completely different materials whose "
        "underlying connection skeletons are EXACTLY the same graph: same "
        "number of nodes, same links, same layout (mirrored or translated). "
        "A viewer tracing either half must recover the identical diagram.",
    "alternation": "The sequence must be a STRICT two-state alternation with "
        "NO exception: A-B-A-B... never two alike adjacent, only two states "
        "total, at least 8 elements, evenly spaced along one path.",
    "aperiodic_tiling": "The pattern must FILL THE ENTIRE FRAME edge to edge "
        "as a Penrose-like tiling: two rhomb/kite-dart shapes, local 5-fold "
        "stars appearing here and there but NO global center, NO radial "
        "symmetry, NO repeating translation unit. Follow the reference "
        "diagram's tile layout exactly.",
    "pigeonhole": "Show MORE items than containers, mid-action: n+1 or more "
        "movers approaching n visible slots, with every slot already claimed "
        "so at least one incoming pair is FORCED to share -- but do not "
        "single out which pair: several are equally close. The count "
        "imbalance must be readable at a glance.",
    "infinite_descent": "Show a descending chain of the SAME kind of object, "
        "each strictly smaller, whose earliest/largest member is CUT OFF by "
        "the frame or hidden -- the chain visibly continues beyond the edge "
        "with no first element. The absence of a starting point is the point.",
    "diagonalization": "Show a clear GRID (rows = candidates, columns = "
        "features) where one diagonal path crosses the table and at EVERY "
        "diagonal cell the local state is visibly flipped relative to its "
        "row (lit vs dark, open vs closed). One extra row/element outside "
        "the table differs from row k exactly at position k.",
    "contraction_fixed_point": "Show ONE path folding toward a single point "
        "through visibly geometric steps: each successive segment a FIXED "
        "RATIO shorter than the last (about half), at least 5 steps, ending "
        "at a marked stationary point where the last steps pile up.",
    "bottleneck": "Two dense, internally well-connected regions with EXACTLY "
        "ONE narrow crossing between them -- every possible route funnels "
        "through that single passage; no second bridge, gap or bypass "
        "anywhere.",
    "convergence": "Show MANY (5+) separate, independent paths starting far "
        "apart and visibly merging pairwise until exactly ONE endpoint "
        "receives them all; the endpoint is terminal -- nothing continues "
        "past it.",
    "self_reference": "Somewhere in the scene (a framed picture, screen, "
        "embroidery hoop, window reflection) appears a RECOGNIZABLE copy of "
        "THIS VERY SCENE -- same room, same objects, same composition, "
        "clearly identifiable as the whole image itself, exactly one level "
        "deep (the copy does not contain a further copy).",
    "coprime_orbit": "N points evenly spaced on a circle, connected by ONE "
        "continuous line that always jumps a fixed step k (skipping points), "
        "forming a STAR polygon that visits every point exactly once before "
        "closing. The star-shaped chords are the composition.",
    "spiral": "One curve winding outward where EACH full turn is visibly "
        "larger than the previous by the SAME ratio (logarithmic growth); "
        "at least 3 turns; the scale growth must be obvious, not a "
        "same-width coil.",
    "cycle": "A closed directed circuit of at least 4 DISTINCT stations "
        "linked head-to-tail around a loop, with a readable sense of flow "
        "(one-way) and no center hub; the circuit must visibly close.",
    "tiling": "A periodic grid of same-shape cells filling the frame; every "
        "cell equivalent -- NO unique anomalous cell, no extraneous "
        "off-grid object breaking the repetition.",
    "braiding": "AT LEAST THREE strands crossing in a visible over-under "
        "weave order; each crossing clearly shows which strand passes over.",
    "interference": "TWO separate wave/periodic sources whose patterns "
        "overlap and visibly modulate each other (moire bands, beat "
        "patterns, brightened crossings) in the overlap zone.",
}

REWRITE_PROMPT = """You are fixing a scene-generation prompt for a structural
benchmark. The image generated from the previous prompt FAILED a structural
screen.

Motif invariant: {invariant}
STRUCTURAL MANDATE (the fix must enforce this):
{mandate}

Previous prompt (middle part):
{old_body}

Screener's failure reason (translated context): {reason}

Rewrite the prompt body (40-100 words, English) so the structural mandate is
IMPOSSIBLE for the image model to miss: state the structural requirement
first, concretely, in terms of scene content; keep a natural everyday scene
(this domain or a close one); no text/labels/diagram looks; do not name the
structure abstractly (no words like symmetric/recursive/loop/fractal).

Output ONLY JSON (no code fences):
{{"prompt_body": "..."}}"""


def rewrite_one(job):
    mid, name, cand, reason = job
    reply = G.chat(G.WRITER_MODEL, "", REWRITE_PROMPT.format(
        invariant=LIB[mid]["invariant_zh"],
        mandate=MANDATES.get(mid, "Make the drawn structure the unmistakable "
                                  "compositional skeleton."),
        old_body=cand["prompt_body"], reason=reason[:300]))
    data = G.parse_json(reply) if reply else None
    if data and data.get("prompt_body"):
        cand["prompt_body"] = data["prompt_body"]
        cand["regen"] = cand.get("regen", 0) + 1
        print(f"  rewrote {mid}/{name}")
        return mid, name, True
    print(f"  FAILED rewrite {mid}/{name}")
    return mid, name, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []
    for fname, rec in SCREEN.items():
        if rec["verdict"] != "fail":
            continue
        mid = rec["motif"]
        name = fname.replace("scene_", "").replace(".png", "")
        cand = next((c for c in PROMPTS.get(mid, []) if c["name"] == name),
                    None)
        if cand is None:
            print(f"  no prompt tuple for {mid}/{name} (pilot scene?) — skip")
            continue
        reason = ((rec.get("verify") or {}).get("problems") or "")
        jobs.append((mid, name, cand, reason))
    print(f"{len(jobs)} failed scenes to rewrite + regenerate")
    if args.dry_run:
        return

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(rewrite_one, jobs))

    ok = [(m, n) for m, n, s in results if s]
    # persist rewritten prompts, delete failed images, clear screen records
    PROMPTS_PATH.write_text(json.dumps(PROMPTS, indent=2, ensure_ascii=False))
    for mid, name in ok:
        png = ROOT / "motifs" / mid / f"scene_{name}.png"
        if png.exists():
            png.unlink()
        SCREEN.pop(f"scene_{name}.png", None)
    (ROOT / "data" / "scene_screen_v2.json").write_text(
        json.dumps(SCREEN, indent=2, ensure_ascii=False))
    print(f"\n{len(ok)} prompts rewritten, images deleted, ready for "
          "gen_scenes_expand.py")
    print("REGEN_PREP_DONE")


if __name__ == "__main__":
    main()
