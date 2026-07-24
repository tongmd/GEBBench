#!/usr/bin/env python3
"""Build exp3_ladder/items.json — scene2motif items over the naturalization
ladder, same format as eval/items.json (25-way menu).

L0 items point at the existing skeleton PNGs (1 per motif); L1–L6 point at
generated images (3 seeds each).  Items are only emitted for images that
exist and are valid PNGs, so this can be re-run as generation progresses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import gen_motifs  # noqa: E402

EXP = ROOT / "exp3_ladder"
MENU = json.loads((ROOT / "eval" / "items.json").read_text())["menu"]


def valid_png(path: Path, min_size: int = 10_000) -> bool:
    if not path.exists() or path.stat().st_size < min_size:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    items = []
    for mid, names in gen_motifs.SKELETONS.items():
        skel = ROOT / "motifs" / mid / f"skeleton_{next(iter(names))}.png"
        if valid_png(skel, min_size=1_000):  # programmatic renders can be tiny
            items.append({"task": "scene2motif",
                          "id": f"ladder:{mid}:L0_s0",
                          "image": str(skel.relative_to(ROOT)),
                          "answer": mid, "level": 0, "seed": 0})
        for level in range(1, 7):
            for seed in range(3):
                p = EXP / "images" / mid / f"L{level}_s{seed}.png"
                if valid_png(p):
                    items.append({"task": "scene2motif",
                                  "id": f"ladder:{mid}:L{level}_s{seed}",
                                  "image": str(p.relative_to(ROOT)),
                                  "answer": mid, "level": level, "seed": seed})
    (EXP / "items.json").write_text(
        json.dumps({"menu": MENU, "items": items}, indent=1,
                   ensure_ascii=False))
    by_level = {}
    for it in items:
        by_level[it["level"]] = by_level.get(it["level"], 0) + 1
    print(f"{len(items)} items", {f"L{k}": v for k, v in sorted(by_level.items())})


if __name__ == "__main__":
    main()
