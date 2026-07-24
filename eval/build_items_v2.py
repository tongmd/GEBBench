#!/usr/bin/env python3
"""GEB benchmark v2 — build the expanded item set (1000+ items).

Reviewer-driven expansion over build_items.py:
  - all expansion assets included: ~10 scenes/motif, base + variant skeletons,
    5 dual-screened stories/motif
  - crossvoice rotates the true story across a motif's story pool
  - two new cross-voice directions:
      xv_scene_thm  scene image + 4 masked theorems   (VLM)
      xv_story_thm  story text  + 4 masked theorems   (LLM -- gives text-only
                                                       models a mapping task)
  - a SEPARATE adversarial split (items_v2_adv.json):
      scene2motif_family4  4-way identification, options restricted to the
      designed confusable family -- forced fine discrimination, and the same
      arity as crossvoice so task difficulty is comparable at fixed k.

Usage:  python3 eval/build_items_v2.py
Output: eval/items_v2.json  (main)  +  eval/items_v2_adv.json  (adversarial)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from build_items import FAMILIES, LIB

ROOT = Path(__file__).parent.parent

# English-primary assets: invariants + pre-masked theorems
LIB_EN = json.loads((ROOT / "data" / "library_en.json").read_text())

# EN story corpus is generated separately; fall back gracefully if absent.
_stories_path = ROOT / "data" / "stories_en.json"
if _stories_path.exists():
    STORIES = [s for s in json.loads(_stories_path.read_text())
               if (s.get("judge") or {}).get("passed")]  # dual-screened only
else:
    print(f"WARN: {_stories_path} missing -- all story-dependent items "
          "will be skipped")
    STORIES = []
BY_MOTIF: dict[str, list[dict]] = {}
for s in STORIES:
    BY_MOTIF.setdefault(s["motif_id"], []).append(s)
for mid in BY_MOTIF:
    BY_MOTIF[mid].sort(key=lambda s: s["story_id"])


def menu_en() -> str:
    return "\n".join(f"- {mid}: {LIB_EN[mid]['invariant_en']}" for mid in LIB)


def family_pool(mid: str) -> list[str]:
    pool = []
    for fam in FAMILIES:
        if mid in fam:
            pool += [x for x in fam if x != mid]
    return list(dict.fromkeys(pool))


def negative_motifs(mid: str, k: int, rng: random.Random) -> list[str]:
    pool = [x for x in family_pool(mid) if x in BY_MOTIF]
    rest = [x for x in BY_MOTIF if x != mid and x not in pool]
    rng.shuffle(pool)
    rng.shuffle(rest)
    return (pool + rest)[:k]


def strict_family_negatives(mid: str, k: int, rng: random.Random) -> list[str]:
    """Family members first, then 2-hop family members, then rest."""
    pool = family_pool(mid)
    hop2 = []
    for x in pool:
        hop2 += [y for y in family_pool(x) if y != mid and y not in pool]
    rest = [y for y in LIB if y != mid and y not in pool and y not in hop2]
    rng.shuffle(pool)
    rng.shuffle(hop2)
    rng.shuffle(rest)
    return (pool + list(dict.fromkeys(hop2)) + rest)[:k]


def letter_options(true_key: str, others: list[str], texts: dict[str, str],
                   rng: random.Random) -> dict:
    keys = [true_key] + others
    rng.shuffle(keys)
    return {
        "options": {chr(65 + i): texts[k] for i, k in enumerate(keys)},
        "option_motifs": {chr(65 + i): k for i, k in enumerate(keys)},
        "answer": chr(65 + keys.index(true_key)),
    }


def main():
    rng = random.Random(20260719)
    items, adv = [], []
    skipped: Counter = Counter()
    skipped_motifs: list[str] = []
    masked_thm = {mid: LIB_EN[mid]["theorem_en_masked"] for mid in LIB}

    for mid, m in LIB.items():
        d = ROOT / "motifs" / mid
        scenes = sorted(d.glob("scene_*.png"))
        skels = sorted(d.glob("skeleton_*.png"))
        stories = BY_MOTIF.get(mid, [])
        if not stories:
            # no passed EN story for this motif -> skip its story-dependent
            # items (story2motif, xv_story_thm, crossvoice)
            skipped_motifs.append(mid)
            skipped["story2motif"] += 1
            skipped["xv_story_thm"] += 1
            skipped["crossvoice"] += len(scenes)

        for png in scenes:
            items.append({"task": "scene2motif", "split": "main",
                          "id": f"scene:{png.stem}:{mid}",
                          "image": str(png.relative_to(ROOT)), "answer": mid})
        for png in skels:
            items.append({"task": "skeleton2motif", "split": "main",
                          "id": f"skel:{mid}:{png.stem}",
                          "image": str(png.relative_to(ROOT)), "answer": mid})
        for s in stories:
            items.append({"task": "story2motif", "split": "main",
                          "id": f"story:{s['story_id']}",
                          "text": s["story_en"], "answer": mid})
        items.append({"task": "theorem2motif", "split": "main",
                      "id": f"thm:{mid}", "text": masked_thm[mid],
                      "answer": mid})

        # crossvoice: scene -> story, true story rotates across the pool
        if stories and len(BY_MOTIF) >= 8:  # need enough negative pools
            for i, png in enumerate(scenes):
                true_story = stories[i % len(stories)]
                negs = negative_motifs(mid, 3, rng)
                texts = {mid: true_story["story_en"][:600]}
                for n in negs:
                    texts[n] = rng.choice(BY_MOTIF[n])["story_en"][:600]
                items.append({"task": "crossvoice", "split": "main",
                              "id": f"xv:{png.stem}:{mid}",
                              "image": str(png.relative_to(ROOT)),
                              **letter_options(mid, negs, texts, rng)})
        elif stories:
            skipped["crossvoice"] += len(scenes)

        # xv_scene_thm: scene -> masked theorem
        for png in scenes:
            negs = strict_family_negatives(mid, 3, rng)
            texts = {k: masked_thm[k] for k in [mid] + negs}
            items.append({"task": "xv_scene_thm", "split": "main",
                          "id": f"xvt:{png.stem}:{mid}",
                          "image": str(png.relative_to(ROOT)),
                          **letter_options(mid, negs, texts, rng)})

        # xv_story_thm: story -> masked theorem (text-only mapping task)
        for s in stories:
            negs = strict_family_negatives(mid, 3, rng)
            texts = {k: masked_thm[k] for k in [mid] + negs}
            items.append({"task": "xv_story_thm", "split": "main",
                          "id": f"xvs:{s['story_id']}",
                          "text": s["story_en"],
                          **letter_options(mid, negs, texts, rng)})

        # adversarial split: 4-way identification inside the family
        for png in scenes:
            negs = strict_family_negatives(mid, 3, rng)
            opts = [mid] + negs
            rng.shuffle(opts)
            adv.append({"task": "scene2motif_family4", "split": "adversarial",
                        "id": f"adv4:{png.stem}:{mid}",
                        "image": str(png.relative_to(ROOT)),
                        "menu4": {o: LIB_EN[o]["invariant_en"] for o in opts},
                        "answer": mid})

    (ROOT / "eval" / "items_v2.json").write_text(
        json.dumps({"menu": menu_en(), "items": items}, indent=1,
                   ensure_ascii=False))
    (ROOT / "eval" / "items_v2_adv.json").write_text(
        json.dumps({"menu": menu_en(), "items": adv}, indent=1,
                   ensure_ascii=False))
    print("main:", Counter(i["task"] for i in items))
    print("adversarial:", Counter(i["task"] for i in adv))
    if skipped_motifs or skipped:
        print(f"skipped story-dependent items "
              f"({len(skipped_motifs)} motifs without passed EN stories: "
              f"{', '.join(skipped_motifs) or 'none'}):", dict(skipped))
    print(f"total: {len(items)} main + {len(adv)} adversarial "
          f"= {len(items) + len(adv)}")


if __name__ == "__main__":
    main()
