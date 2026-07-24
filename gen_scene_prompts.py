#!/usr/bin/env python3
"""Author + screen new scene prompts to reach 10 scenes per motif.

Stage (a) of the scene expansion.  For each motif, gpt-5.5:stable writes the
missing scene prompt tuples in the exact house style of gen_motifs.SCENES
(REF prefix + structure-to-content mapping + NO_DIAGRAM suffix are added by
this script, the writer produces only the middle).  gpt-5.2:stable screens
every candidate: does the prompt force the drawn structure to BE the
composition; is the domain fresh; no text/diagram leakage.  One rewrite round
for rejects.

Output: data/scene_prompts_v2.json
    {motif_id: [{name, skel, size, prompt_body, judge}]}
Usage:  python3 gen_scene_prompts.py [--target 10] [motif_id ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gen_motifs as M
import gen_stories as G  # chat() + parse_json()

G.WRITER_MODEL = "gpt-5.5:stable"
G.JUDGE_MODEL = "gpt-5.2:stable"

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "scene_prompts_v2.json"
SIZES = {"1024x1024", "1536x1024", "1024x1536"}
SAVE_LOCK = threading.Lock()


def save_state(state: dict) -> None:
    with SAVE_LOCK:
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        tmp.rename(OUT)

WRITER_PROMPT = """You are the scene author for a benchmark of abstract \
structural motifs.  A motif is realized as a natural or artistic image whose \
COMPOSITION is the structure; the image is generated from a skeleton diagram \
used as reference, so your prompt must map every drawn element of the diagram \
onto scene content explicitly (\"every fork of the diagram becomes a fork of \
the river\", \"each circle of the diagram is a ring of city wall\").

Motif: {name_en}
Invariant (translate the idea, don't quote it): {invariant}
Skeleton files available as reference (pick per scene; variants differ only \
in nuisance parameters like depth/count/angle): {skels}
Curated scene ideas you may draw on or depart from: {ideas}

House-style examples from this benchmark (middle part only; a fixed prefix \
tells the model the diagram is only a compositional skeleton, and a fixed \
suffix forbids any diagram/text look):
{examples}

Domains already used for this motif (your scenes must use FRESH domains, and \
must also differ from each other): {used}

Write {k} new scene prompt tuples.  Hard rules:
1. The structure must be carried by the composition alone -- a viewer who \
traces the scene recovers the diagram.  Map drawn elements to scene content \
explicitly, in the style of the examples.
2. Vary the medium across your tuples (photograph / aerial or satellite / \
macro / classical painting / woodblock print / ink drawing / long exposure \
...), and vary palette and era.  No two tuples in the same domain.
3. The scene must contain NO text, NO diagrams, NO labels, and nothing that \
names the structure.  Everyday content only.
4. The structural invariant must be exactly satisfied -- do not smooth away \
asymmetries the diagram draws, do not add extra symmetry, do not let the \
medium break the topology (e.g. a mobius band must keep its half-twist \
visible).
5. name: short snake_case, unique, concrete (like river_delta, walled_city).
6. The prompt body must NEVER contain structure-naming words (symmetry, \
mirror, ring, cycle, spiral, recursive, nested, loop, fractal, pattern, \
diagram, skeleton, geometry...) -- describe the scene content and the \
mapping from drawn elements only ("each drawn circle is a ring of wall", \
never "concentric/nested rings").

Output ONLY JSON (no code fences):
{{"scenes": [{{"name": "...", "skel": "one of the skeleton names", \
"size": "1024x1024 or 1536x1024 or 1024x1536", "prompt_body": "the middle \
text, 40-90 words, English"}}]}}"""

JUDGE_PROMPT = """You are screening a scene-generation prompt for a \
structural-motif benchmark.  The image will be generated with a skeleton \
diagram as compositional reference.

Motif invariant: {invariant}
Metadata, NOT sent to the image model (do not judge it): name={name}, \
skeleton file={skel}, size={size}.
The ONLY text the image model sees is the prompt body below (plus a fixed \
prefix saying the diagram is only a compositional skeleton, and a fixed \
suffix forbidding any diagram/text look).  Judge the body alone:
  {prompt_body}
Domains already used for this motif: {used}

Checks (all must pass):
1. STRUCTURE: the prompt explicitly maps the diagram's drawn elements onto \
scene content, so the composition IS the motif -- not merely a scene that \
vaguely evokes it.
2. INVARIANT: nothing in the prompt breaks or weakens the invariant \
(no added symmetry, no smoothed asymmetry, no broken topology).
3. FRESH: the domain is not in the used list and is concretely different.
4. CLEAN: no text/labels/diagram content requested; no structure-naming words \
that could appear as visible text.

Output ONLY JSON (no code fences):
{{"passed": true or false, "problems": "empty string if all pass"}}"""


def author_for_motif(job) -> None:
    mid, k, state = job
    lib = M.LIBRARY[mid]
    d = ROOT / "motifs" / mid
    skels = sorted(p.stem.replace("skeleton_", "")
                   for p in d.glob("skeleton_*.png"))
    existing_names = sorted(p.stem.replace("scene_", "")
                            for p in d.glob("scene_*.png"))
    accepted = state.get(mid, [])
    used = existing_names + [s["name"] for s in accepted]
    examples = "\n".join(
        f'- name={n}, skel={sk}, size={sz}: "{p[len(M.REF):][:220]}..."'
        for n, sk, sz, p in (M.SCENES.get(mid, []) or M.SCENES["recursion"])[:2])
    ideas = json.dumps(lib["scene_ideas"], ensure_ascii=False)

    need = k - len(accepted)
    for round_ in range(3):
        if need <= 0:
            break
        reply = G.chat(G.WRITER_MODEL, "", WRITER_PROMPT.format(
            name_en=lib["name_en"], invariant=lib["invariant_zh"],
            skels=", ".join(skels), ideas=ideas, examples=examples,
            used="、".join(used) or "none", k=need))
        data = G.parse_json(reply) if reply else None
        if not data or "scenes" not in data:
            print(f"  {mid}: writer bad output (round {round_ + 1})")
            continue
        for cand in data["scenes"][:need]:
            if (not isinstance(cand, dict) or cand.get("skel") not in skels
                    or cand.get("size") not in SIZES
                    or not cand.get("prompt_body")
                    or not cand.get("name")
                    or cand["name"] in used):
                print(f"  {mid}: drop malformed candidate "
                      f"{cand.get('name') if isinstance(cand, dict) else '?'}")
                continue
            jr = G.chat(G.JUDGE_MODEL, "", JUDGE_PROMPT.format(
                invariant=lib["invariant_zh"], name=cand["name"],
                skel=cand["skel"], size=cand["size"],
                prompt_body=cand["prompt_body"],
                used="、".join(used) or "none"))
            verdict = G.parse_json(jr) if jr else None
            cand["judge"] = verdict
            if verdict and verdict.get("passed"):
                accepted.append(cand)
                used.append(cand["name"])
                print(f"  ok {mid}/{cand['name']} [{cand['skel']}]")
            else:
                prob = (verdict or {}).get("problems", "judge unavailable")
                print(f"  reject {mid}/{cand['name']}: {str(prob)[:100]}")
        need = k - len(accepted)
    state[mid] = accepted
    save_state(state)  # incremental: survive restarts
    if need > 0:
        print(f"  WARN {mid}: only {len(accepted)} accepted, wanted {k}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10)
    ap.add_argument("ids", nargs="*")
    args = ap.parse_args()
    ids = args.ids or list(M.LIBRARY)

    state: dict[str, list] = {}
    if OUT.exists():
        state = json.loads(OUT.read_text())

    jobs = []
    for mid in ids:
        have = len(list((ROOT / "motifs" / mid).glob("scene_*.png")))
        need_total = max(0, args.target - have)
        if len(state.get(mid, [])) >= need_total:
            continue
        jobs.append((mid, need_total, state))
    print(f"{len(jobs)} motifs need new scene prompts")
    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(author_for_motif, jobs))

    OUT.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    total = sum(len(v) for v in state.values())
    print(f"\n{total} accepted scene prompts -> {OUT}")
    print("SCENE_PROMPTS_DONE")


if __name__ == "__main__":
    main()
