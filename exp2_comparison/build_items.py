#!/usr/bin/env python3
"""Experiment ② — build items.json: 25 motifs × 5 conditions, within-item.

Every condition of a motif shares the same TARGET scene (left panel of all
pairs == the C1 single image); answer is always the target's motif_id.
Menu = the same 25-way menu as eval/items.json.

Usage:  python3 exp2_comparison/build_items.py
Output: exp2_comparison/items.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "exp2_comparison"))
from design import DESIGN, CONDITIONS

MENU = json.loads((ROOT / "eval" / "items.json").read_text())["menu"]
STIM = ROOT / "exp2_comparison" / "stimuli"


def main():
    items = []
    for mid, (target_scene, far_scene, (foil_mid, foil_scene)) in DESIGN.items():
        target = f"motifs/{mid}/scene_{target_scene}.png"
        images = {
            "single": target,
            "near": f"exp2_comparison/stimuli/pair_near_{mid}.png",
            "far": f"exp2_comparison/stimuli/pair_far_{mid}.png",
            "foil": f"exp2_comparison/stimuli/pair_foil_{mid}.png",
            "instructed": f"exp2_comparison/stimuli/pair_far_{mid}.png",
        }
        meta = {
            "target_scene": target_scene,
            "far_scene": far_scene or f"gen:far_src_{mid}",
            "foil": f"{foil_mid}/scene_{foil_scene}",
        }
        for cond in CONDITIONS:
            img = images[cond]
            if not (ROOT / img).exists():
                print(f"  WARNING missing image: {img}")
            items.append({"task": "comparison", "condition": cond,
                          "id": f"cmp:{cond}:{mid}", "motif": mid,
                          "image": img, "answer": mid, "meta": meta})
    out = {"menu": MENU, "items": items}
    (ROOT / "exp2_comparison" / "items.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(Counter(i["condition"] for i in items),
          "->", ROOT / "exp2_comparison" / "items.json")


if __name__ == "__main__":
    main()
