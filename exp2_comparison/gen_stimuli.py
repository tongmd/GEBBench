#!/usr/bin/env python3
"""Experiment ② — generate missing stimuli and compose side-by-side pairs.

Steps
  1. NEAR renders: 25 nuisance-resamples of each motif's target scene
     (same prompt + NEAR_VARIATION, gpt-image-2, skeleton ref where the
     original used one)                    -> stimuli/near_src_<motif>.png
  2. FAR renders: 7 new far scenes for single-scene motifs (FAR_GEN)
                                           -> stimuli/far_src_<motif>.png
  3. Pair composites (PIL, equal heights, 24px white gutter, target LEFT)
                                           -> stimuli/pair_{near,far,foil}_<motif>.png
     (C5 instructed reuses pair_far_<motif>.png — identical stimulus to C3)

Usage:  python3 exp2_comparison/gen_stimuli.py [--compose-only]
Idempotent: existing files are skipped.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp2_comparison"))

import gen_motifs  # SCENES / T2I_SCENES prompts + gpt_edit / gpt_t2i
from design import DESIGN, FAR_GEN, NEAR_VARIATION

STIM = ROOT / "exp2_comparison" / "stimuli"
GUTTER = 24


def target_prompt(mid: str, scene: str):
    """Return ('edit', skel_png, size, prompt) or ('t2i', None, size, prompt)."""
    for name, skel, size, prompt in gen_motifs.SCENES.get(mid, []):
        if name == scene:
            return "edit", ROOT / "motifs" / mid / f"skeleton_{skel}.png", size, prompt
    for name, size, prompt in gen_motifs.T2I_SCENES.get(mid, []):
        if name == scene:
            return "t2i", None, size, prompt
    raise KeyError(f"no prompt found for {mid}/{scene}")


def gen_near(mid: str) -> None:
    out = STIM / f"near_src_{mid}.png"
    if out.exists():
        print(f"  skip {out.name}")
        return
    mode, skel, size, prompt = target_prompt(mid, DESIGN[mid][0])
    ok = (gen_motifs.gpt_edit(skel, prompt + NEAR_VARIATION, size, out)
          if mode == "edit" else
          gen_motifs.gpt_t2i(prompt + NEAR_VARIATION, size, out))
    print(f"  {'ok' if ok else 'FAIL'} {out.name}")


def gen_far(mid: str) -> None:
    out = STIM / f"far_src_{mid}.png"
    if out.exists():
        print(f"  skip {out.name}")
        return
    name, skel, size, prompt = FAR_GEN[mid]
    ok = gen_motifs.gpt_edit(ROOT / "motifs" / mid / f"skeleton_{skel}.png",
                             prompt, size, out)
    print(f"  {'ok' if ok else 'FAIL'} {out.name}")


def compose(left: Path, right: Path, out: Path) -> None:
    a, b = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
    h = min(a.height, b.height, 1024)
    a = a.resize((round(a.width * h / a.height), h), Image.LANCZOS)
    b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)
    canvas = Image.new("RGB", (a.width + GUTTER + b.width, h), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + GUTTER, 0))
    canvas.save(out)


def pair_sources(mid: str) -> dict[str, tuple[Path, Path]]:
    target_scene, far_scene, (foil_mid, foil_scene) = DESIGN[mid]
    target = ROOT / "motifs" / mid / f"scene_{target_scene}.png"
    far = (ROOT / "motifs" / mid / f"scene_{far_scene}.png" if far_scene
           else STIM / f"far_src_{mid}.png")
    return {
        "near": (target, STIM / f"near_src_{mid}.png"),
        "far": (target, far),
        "foil": (target, ROOT / "motifs" / foil_mid / f"scene_{foil_scene}.png"),
    }


def main():
    STIM.mkdir(parents=True, exist_ok=True)
    if "--compose-only" not in sys.argv:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(gen_near, DESIGN))
            list(ex.map(gen_far, FAR_GEN))
    missing = 0
    for mid in DESIGN:
        for kind, (left, right) in pair_sources(mid).items():
            out = STIM / f"pair_{kind}_{mid}.png"
            if not (left.exists() and right.exists()):
                print(f"  MISSING source for {out.name}: {left.name}, {right.name}")
                missing += 1
                continue
            compose(left, right, out)
    print(f"composites done ({missing} missing sources)")


if __name__ == "__main__":
    main()
