#!/usr/bin/env python3
"""GEB benchmark — build evaluation items from the benchmark artifacts.

All items are k-way multiple choice, judge-free.  The option menu is the motif
id + its one-line invariant (Chinese).  Answer format: JSON {"motif_id": ...}
(or {"story_id": ...} for cross-voice matching).

Tasks
  scene2motif     scene image            -> which motif?          (VLM)
  skeleton2motif  skeleton image         -> which motif?          (VLM ceiling)
  story2motif     form-device story text -> which motif?          (LLM)
  theorem2motif   theorem, names masked  -> which motif?          (LLM)
  crossvoice      scene image + 4 stories (1 true + 3 confusable
                  hard negatives)        -> which story matches?  (VLM)

Usage:  python3 eval/build_items.py
Output: eval/items.json
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
LIB = {m["id"]: m for m in
       json.loads((ROOT / "data" / "motif_library.json").read_text())}
STORIES = {s["motif_id"]: s for s in
           json.loads((ROOT / "data" / "stories_v2.json").read_text())}

# confusable families — hard-negative pools for crossvoice
FAMILIES = [
    ["self_reference", "strange_loop", "recursion", "nesting"],
    ["cycle", "strange_loop", "mobius", "spiral", "alternation"],
    ["symmetry", "figure_ground", "isomorphism", "mobius"],
    ["tiling", "symmetry_breaking", "interference", "alternation", "braiding"],
    ["convergence", "bottleneck", "hub"],
    # round-2 motifs, grouped with their documented near-neighbors
    ["diagonalization", "self_reference", "symmetry_breaking"],
    ["infinite_descent", "recursion", "contraction_fixed_point", "convergence"],
    ["coprime_orbit", "cycle", "interference", "pigeonhole"],
    ["pigeonhole", "convergence", "bottleneck", "hub"],
    ["projective_duality", "isomorphism", "figure_ground", "symmetry"],
    ["contraction_fixed_point", "spiral", "convergence", "nesting"],
    ["aperiodic_tiling", "tiling", "interference", "alternation"],
]


def menu() -> str:
    return "\n".join(f"- {mid}: {m['invariant_zh']}" for mid, m in LIB.items())


def negatives(mid: str, k: int, rng: random.Random) -> list[str]:
    pool = []
    for fam in FAMILIES:
        if mid in fam:
            pool += [x for x in fam if x != mid and x in STORIES]
    pool = list(dict.fromkeys(pool))
    rest = [x for x in STORIES if x != mid and x not in pool]
    rng.shuffle(pool)
    rng.shuffle(rest)
    return (pool + rest)[:k]


def mask_names(text: str) -> str:
    """Mask motif-revealing proper names in theorems (crude but effective)."""
    for w in ("莫比乌斯", "回文", "对称", "螺线", "螺旋", "辫", "同构", "自指",
              "递归", "嵌套", "密铺", "壁纸", "星图", "割点", "干涉", "拍频",
              "对角线", "递降", "鸽笼", "抽屉", "对偶", "不动点", "压缩映射",
              "收缩", "非周期", "五度圈"):
        text = text.replace(w, "□")
    return text


def main():
    rng = random.Random(20260715)
    items = []

    for mid, m in LIB.items():
        d = ROOT / "motifs" / mid
        for png in sorted(d.glob("scene_*.png")):
            items.append({"task": "scene2motif", "id": f"scene:{png.stem}:{mid}",
                          "image": str(png.relative_to(ROOT)),
                          "answer": mid})
        for png in sorted(d.glob("skeleton_*.png")):
            items.append({"task": "skeleton2motif",
                          "id": f"skel:{mid}:{png.stem}",
                          "image": str(png.relative_to(ROOT)), "answer": mid})
        if mid in STORIES:
            items.append({"task": "story2motif", "id": f"story:{mid}",
                          "text": STORIES[mid]["story_zh"], "answer": mid})
        items.append({"task": "theorem2motif", "id": f"thm:{mid}",
                      "text": mask_names(m["math_theorem"]), "answer": mid})

    # crossvoice: every scene image, 4 candidate stories
    for mid, m in LIB.items():
        if mid not in STORIES:
            continue
        d = ROOT / "motifs" / mid
        for png in sorted(d.glob("scene_*.png")):
            negs = negatives(mid, 3, rng)
            opts = [mid] + negs
            rng.shuffle(opts)
            items.append({
                "task": "crossvoice", "id": f"xv:{png.stem}:{mid}",
                "image": str(png.relative_to(ROOT)),
                "options": {chr(65 + i): STORIES[o]["story_zh"][:500]
                            for i, o in enumerate(opts)},
                "option_motifs": {chr(65 + i): o for i, o in enumerate(opts)},
                "answer": chr(65 + opts.index(mid)),
            })

    out = {"menu": menu(), "items": items}
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "items.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    from collections import Counter
    print(Counter(i["task"] for i in items), "->", ROOT / "eval" / "items.json")


if __name__ == "__main__":
    main()
