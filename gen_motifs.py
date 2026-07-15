#!/usr/bin/env python3
"""GEB benchmark — motif pipeline: abstract structure -> skeleton -> natural scene.

The unit of a sample is an ABSTRACT STRUCTURAL MOTIF (self-reference, recursion,
nesting, cycle ...), not a concrete graph.  Node counts, depths, branching
factors, angles are nuisance variables and are recorded as such.

Pipeline per motif:
    motif z
      -> skeleton(s): programmatic vis-graph renderings of the motif
      -> scene(s):    gpt-image with the skeleton as REFERENCE image, the scene
                      prompt reinterprets it as a natural / artistic image
                      (rivers, lightning, city walls, ouroboros ...), so the
                      composition IS the structure — not a style transfer
      -> story / math voices written at motif level (no counts, no sizes)

Usage:  python3 gen_motifs.py [motif_id ...]
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "gpt-image-2"
ROOT = Path(__file__).parent
OUT = ROOT / "motifs"

NO_DIAGRAM = ("It must look like a real photograph or a piece of art, NOT a "
              "diagram: no visible diagram lines, no text, no letters.")


# ---------------------------------------------------------------------------
# skeleton renderers (SVG strings; plain, thin, reference-only)
# ---------------------------------------------------------------------------

def svg_doc(w: int, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
            f'<rect width="100%" height="100%" fill="#ffffff"/>{body}</svg>')


def skel_tree(depth: int = 4, angle: float = 34, scale: float = 0.62,
              length: float = 300, w: int = 1024, h: int = 1024) -> str:
    parts = []

    def branch(x, y, heading, ln, level):
        x2 = x + ln * math.cos(math.radians(heading))
        y2 = y + ln * math.sin(math.radians(heading))
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#1a1a1a" stroke-width="{max(2, 9 - 2 * level)}" '
                     'stroke-linecap="round"/>')
        if level < depth:
            branch(x2, y2, heading - angle, ln * scale, level + 1)
            branch(x2, y2, heading + angle, ln * scale, level + 1)

    branch(w / 2, h - 40, -90, length, 0)
    return svg_doc(w, h, "".join(parts))


def skel_concentric(n: int = 7, w: int = 1024, h: int = 1024) -> str:
    cx, cy, r_max = w / 2, h / 2, min(w, h) / 2 - 40
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{r_max * (k + 1) / n:.1f}" fill="none" '
             'stroke="#1a1a1a" stroke-width="4"/>' for k in range(n)]
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="#1a1a1a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_ring(w: int = 1024, h: int = 1024, segments: int = 8) -> str:
    cx, cy, r = w / 2, h / 2, min(w, h) / 2 - 90
    marker = ('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" '
              'markerWidth="8" markerHeight="8" orient="auto">'
              '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a"/></marker></defs>')
    parts = [marker]
    for k in range(segments):
        a1 = 2 * math.pi * k / segments
        a2 = a1 + 2 * math.pi / segments * 0.72
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
        parts.append(f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
                     'fill="none" stroke="#1a1a1a" stroke-width="5" marker-end="url(#a)"/>')
    return svg_doc(w, h, "".join(parts))


def skel_frames(levels: int = 4, scale: float = 0.62, w: int = 1536, h: int = 1024) -> str:
    parts, fw, fh = [], w - 160, h - 160
    for _ in range(levels):
        parts.append(f'<rect x="{(w - fw) / 2:.1f}" y="{(h - fh) / 2:.1f}" '
                     f'width="{fw:.1f}" height="{fh:.1f}" fill="none" '
                     'stroke="#1a1a1a" stroke-width="5"/>')
        fw, fh = fw * scale, fh * scale
    return svg_doc(w, h, "".join(parts))


# ---------------------------------------------------------------------------
# motifs
# ---------------------------------------------------------------------------

MOTIFS = {
    "recursion": {
        "name": "Recursive branching — the part repeats the whole",
        "equation": "T = \\operatorname{Node}(T,\\dots,T)",
        "invariant": "every branch, looked at alone, has the same shape as the whole",
        "nuisance": ["depth", "branching factor", "angles", "scale ratio", "domain"],
        "story": (
            "[domain: 河与支流]\n\n"
            "问大河：你从哪里来？大河说：我由几条小河汇成。问其中一条小河，"
            "它说：我也是一条河，也由更小的河汇成。再问下去，答案永远一样——"
            "每一条河，都是由和它一样的东西组成的。\n\n"
            "[domain: same structure · English]\n\n"
            "Ask the great river where it comes from, and it answers: from smaller "
            "rivers. Ask any of those, and it gives the same answer. However far you "
            "go, the answer never changes — each river is made of things shaped "
            "exactly like itself."
        ),
        "math": (
            "A recursively defined object — the type on the left appears, unchanged, "
            "on the right:\n\n"
            "$$T = \\operatorname{Node}(T,\\dots,T).$$\n\n"
            "Unrolling the equation gives the same branching shape at every scale. "
            "How many children, how deep, at what angle — all irrelevant: the "
            "structure is that **the part repeats the whole**."
        ),
        "skeletons": {"tree": lambda: skel_tree()},
        "scenes": [
            ("river_delta", "tree", "1536x1024",
             "Use this branching diagram ONLY as a compositional skeleton. Paint a "
             "photorealistic AERIAL photograph of a river delta in golden wetlands: "
             "one main river enters where the trunk is, and every fork of the diagram "
             "becomes a fork of the river channels, splitting into narrower and "
             "narrower waterways along the drawn branches. Water glinting in low sun, "
             "marsh grass, mudflats. " + NO_DIAGRAM),
            ("lightning", "tree", "1536x1024",
             "Use this branching diagram ONLY as a compositional skeleton. A dramatic "
             "long-exposure photograph of a single lightning bolt at night: the main "
             "discharge channel where the trunk is, every fork of the diagram becomes "
             "a fork of the lightning, branching into thinner glowing filaments along "
             "the drawn branches. Dark storm sky, city horizon barely visible. "
             + NO_DIAGRAM),
        ],
        "extra_scenes": [],
        "t2i_scenes": [
            ("winter_tree", "1024x1024",
             "A striking photograph of a single leafless tree against a pale sky in "
             "winter, seen so its branching is fully visible. The trunk splits into a "
             "few boughs; each bough splits the same way into smaller boughs; each of "
             "those into finer twigs — the identical branching pattern repeating at "
             "every scale, clearly self-similar and fractal. Elegant, high-contrast. "
             "No text, no words, no letters."),
        ],
    },
    "nesting": {
        "name": "Nesting — the same form inside itself, at every scale",
        "equation": "S_\\lambda(X) = X",
        "invariant": "inside each layer sits a smaller copy of the same layer",
        "nuisance": ["number of layers", "spacing", "exact shape", "domain"],
        "story": (
            "[domain: 环城]\n\n"
            "这座古城有一圈城墙。走进城门，里面还是一座城，有自己的一圈城墙；"
            "再走进去，还是一座城。城里有城，墙里有墙，一直到最中心的一座小小的宫殿。\n\n"
            "[domain: same structure · English]\n\n"
            "The ancient city has a wall. Pass through the gate, and inside is another "
            "city with its own wall; pass through again, another still. City within "
            "city, wall within wall, down to one small palace at the very center."
        ),
        "math": (
            "One scaling map carries every layer onto the next, and the whole pattern "
            "onto itself:\n\n"
            "$$S_\\lambda(X) = X, \\qquad 0 < \\lambda < 1.$$\n\n"
            "How many rings there are, how far apart — irrelevant: the structure is "
            "**the same form, containing itself**."
        ),
        "skeletons": {"concentric": lambda: skel_concentric()},
        "scenes": [
            ("walled_city", "concentric", "1024x1024",
             "Use these concentric circles ONLY as a compositional skeleton. A "
             "photorealistic aerial photograph at dawn of an ancient walled city: "
             "each circle of the diagram is a ring of city wall, one inside another, "
             "with rooftops, trees and lanes filling the space between the walls, and "
             "a small palace at the exact center. Warm morning light, long shadows. "
             + NO_DIAGRAM),
        ],
        "extra_scenes": [],
        "t2i_scenes": [
            ("pond_ripples", "1024x1024",
             "A photograph-like, tranquil overhead view of a still dark pond at dawn "
             "where a single raindrop has just struck the center. Perfectly concentric "
             "circular ripples spread outward — many evenly spaced rings, each one a "
             "larger copy of the ring inside it, nested from a tiny bright point at "
             "the center out to the edges. Soft morning light, calm and meditative, "
             "photorealistic water surface. No text, no words, no letters."),
        ],
    },
    "cycle": {
        "name": "Cycle — following the arrows leads back to the start",
        "equation": "v \\leadsto v",
        "invariant": "there is no first element: the end feeds the beginning",
        "nuisance": ["length of the loop", "what the elements are", "domain"],
        "story": (
            "[domain: 鸡与蛋]\n\n"
            "鸡从蛋里孵出来。蛋是鸡下的。那第一只鸡呢？——从蛋里孵出来。"
            "那第一个蛋呢？——是鸡下的。这个问题绕了一圈，回到了它自己开始的地方。\n\n"
            "[domain: same structure · English]\n\n"
            "The hen hatches from the egg; the egg is laid by the hen. Then where did "
            "the first hen come from? From an egg. And the first egg? From a hen. The "
            "question walks in a circle and arrives back exactly where it began."
        ),
        "math": (
            "Follow the arrows and you return to your starting point:\n\n"
            "$$v_0 \\to v_1 \\to \\dots \\to v_n \\to v_0, \\qquad v \\leadsto v.$$\n\n"
            "How many steps the loop has, what the elements are — irrelevant: the "
            "structure is **closure: the end feeds the beginning**."
        ),
        "skeletons": {"ring": lambda: skel_ring()},
        "scenes": [
            ("ouroboros", "ring", "1024x1024",
             "Use this circular diagram of arrows ONLY as a compositional skeleton. "
             "A classical ink-and-gold painting of an ouroboros: a serpent whose body "
             "follows exactly the circle of the diagram, its mouth closing on its own "
             "tail where the arrows meet, scales rendered in fine ink lines with "
             "muted gold accents on aged paper. " + NO_DIAGRAM),
            ("canal_town", "ring", "1024x1024",
             "Use this circular diagram of arrows ONLY as a compositional skeleton. "
             "A photorealistic aerial photograph of a small old town built around a "
             "perfectly circular canal that follows exactly the circle of the "
             "diagram; boats moving one way around it, little bridges, tiled roofs, "
             "morning mist. " + NO_DIAGRAM),
        ],
    },
    "self_reference": {
        "name": "Self-reference — the thing contains itself",
        "equation": "s = T(s)",
        "invariant": "the whole appears, unchanged, inside itself",
        "nuisance": ["visible nesting levels", "medium", "domain"],
        "story": (
            "[domain: 民间口传]\n\n"
            "从前有座山，山里有座庙，庙里有个老和尚，正在给小和尚讲故事。"
            "他讲的是：从前有座山，山里有座庙，庙里有个老和尚，正在给小和尚讲故事……\n\n"
            "[domain: same structure · English]\n\n"
            "Once upon a time there was a mountain; on the mountain, a temple; in the "
            "temple, an old monk telling a young monk a story. The story was: once "
            "upon a time there was a mountain…"
        ),
        "math": (
            "The story is a fixed point of its own telling:\n\n"
            "$$s = T(s), \\qquad f(x) = x.$$\n\n"
            "There is no base case and no bottom level. The structure is "
            "**the whole appearing, unchanged, inside itself**."
        ),
        "skeletons": {"frames": lambda: skel_frames()},
        "scenes": [
            ("droste_study", "frames", "1536x1024",
             "Use these nested rectangles ONLY as a compositional skeleton. A warm "
             "photograph of a quiet wood-panelled study: the outermost rectangle is "
             "the room's back wall, and on that wall hangs a framed painting — the "
             "next rectangle — showing the SAME study with the same desk and lamp; "
             "inside it, the next frame shows the study again, smaller, and again, "
             "receding with each drawn rectangle (Droste effect). Soft lamplight. "
             + NO_DIAGRAM),
        ],
        "extra_scenes": [],
        "t2i_scenes": [
            ("mountain_temple", "1024x1024",
             "A serene traditional Chinese ink-wash and light-color landscape painting. "
             "A misty mountain; on it a small temple; inside, through the open door, an "
             "old monk sits telling a story to a young monk. On the wall behind them "
             "hangs a painting — and that painting shows the SAME mountain, the SAME "
             "temple, the SAME two monks, smaller; and inside THAT scene, another "
             "painting of the same scene, smaller again — a picture within a picture, "
             "receding infinitely toward the center (mise en abyme, Droste effect). "
             "Exactly three clearly visible nested levels before it fades into mist. "
             "Muted ink tones with soft gold light. No text, no words, no letters."),
        ],
    },
}


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def gpt_edit(ref: Path, prompt: str, size: str, out: Path) -> bool:
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/images/edits",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-F", f"model={MODEL}", "-F", f"image[]=@{ref}",
           "-F", f"prompt={prompt}", "-F", f"size={size}", "-F", "quality=high"]
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  retry {out.name} (attempt {attempt + 1})")
            continue
        if "data" in resp:
            out.write_bytes(base64.b64decode(resp["data"][0]["b64_json"]))
            return True
        print(f"  FAIL {out.name}: {json.dumps(resp)[:200]}")
        return False
    return False


def gpt_t2i(prompt: str, size: str, out: Path) -> bool:
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/images/generations",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json",
           "-d", json.dumps({"model": MODEL, "prompt": prompt,
                             "size": size, "quality": "high", "n": 1})]
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  retry {out.name} (attempt {attempt + 1})")
            continue
        if "data" in resp:
            item = resp["data"][0]
            if item.get("b64_json"):
                out.write_bytes(base64.b64decode(item["b64_json"]))
            else:
                subprocess.run(["curl", "-s", "-o", str(out), item["url"]], check=True)
            return True
        print(f"  FAIL {out.name}: {json.dumps(resp)[:200]}")
        return False
    return False


def build_motif(mid: str) -> None:
    spec = MOTIFS[mid]
    d = OUT / mid
    d.mkdir(parents=True, exist_ok=True)

    (d / "schema.json").write_text(json.dumps({
        "motif_id": mid,
        "name": spec["name"],
        "equation": spec["equation"],
        "invariant": spec["invariant"],
        "nuisance_variables": spec["nuisance"],
    }, indent=2, ensure_ascii=False))
    (d / "story.txt").write_text(spec["story"])
    (d / "math.md").write_text(spec["math"])

    # skeletons
    for name, render in spec["skeletons"].items():
        svg = d / f"skeleton_{name}.svg"
        png = d / f"skeleton_{name}.png"
        svg.write_text(render())
        subprocess.run(["rsvg-convert", str(svg), "-o", str(png)], check=True)

    # scenes from skeleton refs
    for scene, skel, size, prompt in spec["scenes"]:
        out = d / f"scene_{scene}.png"
        if out.exists():
            print(f"  skip {mid}/{out.name}")
            continue
        if gpt_edit(d / f"skeleton_{skel}.png", prompt, size, out):
            print(f"  ok {mid}/{out.name}")

    # adopt existing fitting images
    for scene, src in spec.get("extra_scenes", []):
        out = d / f"scene_{scene}.png"
        if not out.exists() and Path(src).exists():
            shutil.copy2(src, out)
            print(f"  copied {mid}/{out.name}")

    # scenes with no skeleton (pure text-to-image)
    for scene, size, prompt in spec.get("t2i_scenes", []):
        out = d / f"scene_{scene}.png"
        if out.exists():
            print(f"  skip {mid}/{out.name}")
            continue
        if gpt_t2i(prompt, size, out):
            print(f"  ok {mid}/{out.name}")


def main():
    ids = sys.argv[1:] or list(MOTIFS)
    for mid in ids:
        build_motif(mid)


if __name__ == "__main__":
    main()
