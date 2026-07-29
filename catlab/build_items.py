#!/usr/bin/env python3
"""Build the functorial-abstraction pilot items.

Tasks
  T0 endpoint_id       25-way identification of each skeleton panel.
                       This is the DENOMINATOR: it lets us condition the
                       square score on whether the model could name both
                       endpoints, and lets us simulate the recognition-only
                       shortcut from the model's own answers.
  T1 square_thm        naturality square, target voice = masked theorem.
     square_story      naturality square, target voice = folk story
                       (includes the identity trap: another story of the
                       SAME motif, i.e. surface change with no structure
                       change).
  T2 composition       two edges shown, one target: does g2 o g1 land where
                       applying both should?  Distractors include the
                       "stopped after g1" answer.
  T3 id_vs_change      given two skeletons, was the structure changed and
                       how?  One option is always "no structural change,
                       only rendering parameters" -- invariance vs
                       equivariance.

Everything is exact-match multiple choice; no judge.
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import gen_pairs as G

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PAIRS = ROOT / "pairs"
OUT = ROOT / "items.json"

LIB = json.loads((REPO / "data" / "library_en.json").read_text())
V2 = json.loads((REPO / "eval" / "items_v2.json").read_text())

# accepted English stories, keyed by motif (from the shipped v2 item set)
STORIES: dict[str, list[str]] = {}
for it in V2["items"]:
    if it["task"] == "story2motif":
        STORIES.setdefault(it["answer"], []).append(it["text"])

MENU25 = V2["menu"]


def thm(mid: str) -> str:
    return LIB[mid]["theorem_en_masked"]


def inv(mid: str) -> str:
    return LIB[mid]["invariant_en"]


# per-edge option sets: (correct=dst, identity=src, two formal near-misses)
DISTRACTORS = {
    "sym_to_symbreak":          ["figure_ground", "interference"],
    "cycle_to_strangeloop":     ["mobius", "spiral"],
    "nesting_to_recursion":     ["self_reference", "infinite_descent"],
    "tiling_to_aperiodic":      ["interference", "alternation"],
    "iso_to_duality":           ["figure_ground", "symmetry"],
    "recursion_to_contraction": ["convergence", "infinite_descent"],
}

# third near-miss for the THEOREM voice, where the identity trap would be a
# verbatim duplicate of the given text (each motif has exactly one theorem):
# the theorem-voice square is 4 DISTINCT motifs, and the identity trap lives
# in the story voice and in id_vs_change instead.
THM_THIRD = {
    "sym_to_symbreak":          "mobius",
    "cycle_to_strangeloop":     "alternation",
    "nesting_to_recursion":     "strange_loop",
    "tiling_to_aperiodic":      "braiding",
    "iso_to_duality":           "mobius",
    "recursion_to_contraction": "spiral",
}

LETTERS = "ABCD"


def opt_block(options: dict[str, str]) -> str:
    return "\n\n".join(f"[{k}]\n{v}" for k, v in options.items())


def build():
    man = json.loads((PAIRS / "manifest.json").read_text())
    items = []
    rng = random.Random(20260726)

    # ---------------------------------------------------------- T0 endpoint id
    for m in man:
        for side, motif in (("base", m["src_motif"]), ("img", m["dst_motif"])):
            items.append({
                "task": "endpoint_id",
                "id": f"eid:{m['edge']}:s{m['seed']}:{side}",
                "image": m[f"{side}_png"],
                "answer": motif,
                "meta": {"edge": m["edge"], "seed": m["seed"], "side": side},
            })

    # ------------------------------------------------- T1 naturality squares
    for m in man:
        e, s = m["edge"], m["seed"]
        src, dst = m["src_motif"], m["dst_motif"]
        d1, d2 = DISTRACTORS[e]

        # --- target voice: masked theorem (4 distinct motifs, no dup text)
        cand = [dst, d1, d2, THM_THIRD[e]]
        assert len(set(cand)) == 4 and src not in cand
        order = cand[:]
        rng.shuffle(order)
        opts = {LETTERS[i]: thm(mid) for i, mid in enumerate(order)}
        items.append({
            "task": "square_thm",
            "id": f"sq_thm:{e}:s{s}",
            "image": m["ref_png"],
            "text": thm(src),
            "options": opts,
            "option_motifs": {LETTERS[i]: mid for i, mid in enumerate(order)},
            "answer": LETTERS[order.index(dst)],
            "meta": {"edge": e, "seed": s, "src": src, "dst": dst},
        })

        # --- target voice: story (identity trap = ANOTHER story of src)
        src_stories = STORIES.get(src, [])
        if len(src_stories) >= 2 and STORIES.get(dst):
            given = src_stories[s % len(src_stories)]
            trap = src_stories[(s + 1) % len(src_stories)]
            assert trap != given
            pool = {dst: STORIES[dst][s % len(STORIES[dst])],
                    d1: STORIES[d1][s % len(STORIES[d1])],
                    d2: STORIES[d2][s % len(STORIES[d2])],
                    src: trap}
            order = list(pool)
            rng.shuffle(order)
            opts = {LETTERS[i]: pool[mid] for i, mid in enumerate(order)}
            items.append({
                "task": "square_story",
                "id": f"sq_story:{e}:s{s}",
                "image": m["ref_png"],
                "text": given,
                "options": opts,
                "option_motifs": {LETTERS[i]: mid for i, mid in enumerate(order)},
                "answer": LETTERS[order.index(dst)],
                "meta": {"edge": e, "seed": s, "src": src, "dst": dst,
                         "identity_option": LETTERS[order.index(src)]},
            })

    # ------------------------------------------------------- T3 id vs change
    CHANGE_NONE = ("No structural change: the two figures realize the same "
                   "structure and differ only in rendering parameters "
                   "(counts, sizes, angles, placement).")
    all_changes = {k: v["change"][0].upper() + v["change"][1:] + "."
                   for k, v in G.EDGES.items()}
    for m in man:
        e, s = m["edge"], m["seed"]
        others = [k for k in all_changes if k != e]
        # (a) structural: base -> img  (answer = this edge's change)
        # (b) nuisance:  base(s) -> base(s') (answer = no structural change)
        sib = next((x for x in man if x["edge"] == e and x["seed"] != s), None)
        for kind in ("structural", "nuisance"):
            if kind == "nuisance" and sib is None:
                continue
            pick = rng.sample(others, 2)
            if kind == "structural":
                pair_png = m["ref_png"]
                correct = all_changes[e]
            else:
                pair_png = m["nuis_png"]
                correct = CHANGE_NONE
            pool = [correct] + [all_changes[k] for k in pick]
            pool.append(CHANGE_NONE if kind == "structural" else all_changes[e])
            order = pool[:]
            rng.shuffle(order)
            opts = {LETTERS[i]: t for i, t in enumerate(order)}
            items.append({
                "task": "id_vs_change",
                "id": f"ivc:{e}:s{s}:{kind}",
                "image": pair_png,
                "options": opts,
                "answer": LETTERS[order.index(correct)],
                "meta": {"edge": e, "seed": s, "kind": kind},
            })

    # -------------------------------------------------------- T2 composition
    # chain: nesting --g1--> recursion --g2--> contraction_fixed_point
    g1 = [x for x in man if x["edge"] == "nesting_to_recursion"]
    g2 = [x for x in man if x["edge"] == "recursion_to_contraction"]
    for a, b in zip(g1, g2):
        comp_png = f"catlab/pairs/comp_s{a['seed']}.png"
        pool = {"contraction_fixed_point": thm("contraction_fixed_point"),
                "recursion": thm("recursion"),          # stopped after g1
                "convergence": thm("convergence"),      # near-miss of the target
                "infinite_descent": thm("infinite_descent")}  # near-miss
        order = list(pool)
        rng.shuffle(order)
        opts = {LETTERS[i]: pool[mid] for i, mid in enumerate(order)}
        items.append({
            "task": "composition",
            "id": f"comp:s{a['seed']}",
            "image": comp_png,
            "text": thm("nesting"),
            "options": opts,
            "option_motifs": {LETTERS[i]: mid for i, mid in enumerate(order)},
            "answer": LETTERS[order.index("contraction_fixed_point")],
            "meta": {"seed": a["seed"],
                     "stop_after_g1_option": LETTERS[order.index("recursion")]},
        })

    return items


if __name__ == "__main__":
    its = build()
    from collections import Counter
    payload = {"menu25": MENU25, "items": its}
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"{len(its)} items -> {OUT}")
    for t, n in Counter(i["task"] for i in its).most_common():
        print(f"  {t:16s} {n}")
