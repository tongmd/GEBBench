#!/usr/bin/env python3
"""Parameterized skeleton instances: 2 extra variants per motif where the
generator takes structural nuisance parameters (depth, count, angle, turns...).

The paper's nuisance-variable contract says node counts / depths / angles are
unscored; multiple instances make that contract concrete in the released set.
Variants change ONLY nuisance parameters, never the invariant (a deeper tree
is still a tree; a 6-segment ring is still a ring).

Output: motifs/<mid>/skeleton_<name>_v2.{svg,png}, _v3.{svg,png}
Usage:  python3 gen_skeleton_variants.py [motif_id ...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import gen_motifs as M

ROOT = Path(__file__).parent

# (mid, base skeleton name) -> two alternate parameterizations
VARIANTS = {
    ("recursion", "tree"): [
        lambda: M.skel_tree(depth=3, angle=26, scale=0.66),
        lambda: M.skel_tree(depth=5, angle=42, scale=0.58),
    ],
    ("nesting", "concentric"): [
        lambda: M.skel_concentric(n=5),
        lambda: M.skel_concentric(n=9),
    ],
    ("cycle", "ring"): [
        lambda: M.skel_ring(segments=6),
        lambda: M.skel_ring(segments=11),
    ],
    ("self_reference", "frames"): [
        lambda: M.skel_frames(levels=3, scale=0.58),
        lambda: M.skel_frames(levels=5, scale=0.68),
    ],
    ("alternation", "chain"): [
        lambda: M.skel_alternation(n=8),
        lambda: M.skel_alternation(n=14),
    ],
    ("spiral", "logspiral"): [
        lambda: M.skel_spiral(turns=2.6, b=0.20),
        lambda: M.skel_spiral(turns=4.4, b=0.12),
    ],
    ("symmetry_breaking", "field"): [
        lambda: M.skel_broken(rows=5, cols=7),
        lambda: M.skel_broken(rows=7, cols=11),
    ],
    ("strange_loop", "ladder"): [
        lambda: M.skel_strange_loop(levels=3),
        lambda: M.skel_strange_loop(levels=5),
    ],
    ("figure_ground", "tessellation"): [
        lambda: M.skel_figure_ground(rows=3, cols=5),
        lambda: M.skel_figure_ground(rows=5, cols=9),
    ],
    ("hub", "star"): [
        lambda: M.skel_hub(n=7),
        lambda: M.skel_hub(n=12),
    ],
    ("tiling", "honeycomb"): [
        lambda: M.skel_tiling(r=80),
        lambda: M.skel_tiling(r=150),
    ],
    ("braiding", "weave"): [
        lambda: M.skel_braid(strands=3, cycles=1.4),
        lambda: M.skel_braid(strands=4, cycles=2.6),
    ],
}


def main() -> None:
    ids = sys.argv[1:] or [mid for mid, _ in VARIANTS]
    made = 0
    for (mid, name), renders in VARIANTS.items():
        if mid not in ids:
            continue
        d = ROOT / "motifs" / mid
        for i, render in enumerate(renders, start=2):
            svg = d / f"skeleton_{name}_v{i}.svg"
            png = d / f"skeleton_{name}_v{i}.png"
            if png.exists():
                print(f"  skip {mid}/{png.name}")
                continue
            svg.write_text(render())
            subprocess.run(["rsvg-convert", str(svg), "-o", str(png)],
                           check=True)
            print(f"  ok {mid}/{png.name}")
            made += 1
    print(f"\n{made} variant skeletons rendered")
    print("SKELETON_VARIANTS_DONE")


if __name__ == "__main__":
    main()
