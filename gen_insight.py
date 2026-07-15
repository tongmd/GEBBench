#!/usr/bin/env python3
"""GEB benchmark — "insight" examples: one structure, seen as a landscape.

Unlike the diagram families, these render as NATURAL / ARTISTIC images where the
abstract structure IS the scene, not an overlaid graph.  You look at the picture
and the structure dawns on you (mise-en-abyme = self-reference = f(x)=x).

Text-to-image via gpt-image (images/generations).  Structure is fixed by the
schema and spelled into the prompt; the image is meant to be *understood*, not
decoded node-by-node.

Usage:  python3 gen_insight.py [sample_id ...]
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL = "gpt-image-2"
ROOT = Path(__file__).parent / "examples"

# Each insight sample: the same latent structure across story / math / image.
SAMPLES = {
    "fix_mountain_temple_0006": {
        "family": "fixpoint",
        "structure": "self_reference",
        "concept": "A picture that contains a smaller copy of itself, forever "
                   "(mise en abyme). The scene is its own subpicture: f(x) = x.",
        "story": (
            "[domain: 民间口传]\n\n"
            "从前有座山，山里有座庙，庙里有个老和尚，正在给小和尚讲故事。"
            "他讲的是：从前有座山，山里有座庙，庙里有个老和尚，正在给小和尚讲故事……\n\n"
            "[domain: 同一结构·英文]\n\n"
            "Once upon a time there was a mountain. On the mountain stood a temple. "
            "In the temple an old monk was telling a story to a young monk. "
            "The story was: once upon a time there was a mountain…"
        ),
        "math": (
            "The story is a **fixed point** of the telling operator.\n\n"
            "Let $T$ be \"a monk telling story $s$\". The content of the telling is "
            "the very same story:\n\n"
            "$$s = T(s).$$\n\n"
            "Equivalently the story is a fixed point of the identity-through-nesting "
            "map $f(x)=x$, unrolled as an infinite descent\n\n"
            "$$s = (\\text{mountain},\\ \\text{temple},\\ \\text{monk telling } s),$$\n\n"
            "so $s = g(s)$ with no base case — the defining signature of self-reference "
            "and of a picture that contains itself."
        ),
        # text-to-image prompt: NATURAL / ARTISTIC, structure lives in the scene
        "image_prompt": (
            "A serene traditional Chinese ink-wash and light-color landscape painting. "
            "A misty mountain; on it a small temple; inside, through the open door, an "
            "old monk sits telling a story to a young monk. On the wall behind them hangs "
            "a painting — and that painting shows the SAME mountain, the SAME temple, the "
            "SAME two monks, smaller; and inside THAT scene, another painting of the same "
            "scene, smaller again — a picture within a picture within a picture, receding "
            "infinitely toward the center (mise en abyme, Droste effect). Exactly three "
            "clearly visible nested levels before it fades into mist. Calm, contemplative, "
            "muted ink tones with soft gold light. No text, no words, no letters anywhere."
        ),
        "size": "1024x1024",
        "answer": "self-reference / fixed point: the scene contains itself, s = T(s), f(x)=x",
    },
    "fix_concentric_pond_0007": {
        "family": "fixpoint",
        "structure": "nested_selfsimilar",
        "concept": "Concentric circles as the natural shape of 'the same thing at every "
                   "scale' — a raindrop's ripples, each ring a scaled copy of the last.",
        "story": (
            "[domain: 自然景象]\n\n"
            "雨滴落进静静的池塘。一圈波纹荡开，外面又是一圈一样的波纹，再外面还是一圈——"
            "每一圈都是里面那一圈放大后的样子，一圈套着一圈，直到池边。\n\n"
            "[domain: 同一结构·英文]\n\n"
            "A single raindrop falls on a still pond. One ring spreads out, then an "
            "identical ring outside it, then another — each ring the same shape as the "
            "one inside, only larger, nested outward to the shore."
        ),
        "math": (
            "Each ripple ring is the previous one under a **scaling map**, so the "
            "pattern is a fixed point of scale-similarity.\n\n"
            "$$R_{n+1} = \\lambda \\cdot R_n, \\qquad \\lambda > 1,$$\n\n"
            "and the whole family is invariant under the same map $S_\\lambda$ that "
            "carries any ring to the next:\n\n"
            "$$S_\\lambda(\\{R_n\\}) = \\{R_n\\}.$$\n\n"
            "Concentric self-similar rings are the visual signature of $f(x)=x$ under "
            "scaling — the same form recurring at every radius."
        ),
        "image_prompt": (
            "A photograph-like, tranquil overhead view of a still dark pond at dawn where "
            "a single raindrop has just struck the center. Perfectly concentric circular "
            "ripples spread outward — many evenly spaced rings, each one a larger copy of "
            "the ring inside it, nested from a tiny bright point at the very center out to "
            "the edges. Soft morning light, gentle reflections, calm and meditative, "
            "photorealistic water surface. No text, no words, no letters."
        ),
        "size": "1024x1024",
        "answer": "self-similarity under scaling: each ring is a scaled copy, S_λ(x)=x",
    },
    "fix_branch_tree_0008": {
        "family": "fixpoint",
        "structure": "recursive_branching",
        "concept": "A real tree as the natural image of recursion: each branch is a "
                   "smaller tree — the whole equals a node joining copies of itself.",
        "story": (
            "[domain: 自然景象]\n\n"
            "一棵大树，主干上分出几根粗枝。你走近看任何一根粗枝——它自己又是一棵小树，"
            "同样地分叉；再看它的一根细枝，还是一棵更小的树。整棵树，是由许多和它一模一样、"
            "只是更小的树拼成的。\n\n"
            "[domain: 同一结构·英文]\n\n"
            "A great tree splits from its trunk into several boughs. Look closely at any "
            "bough — it is itself a little tree that branches the same way; and a twig of "
            "that is a smaller tree again. The whole is made of smaller copies of itself."
        ),
        "math": (
            "A tree is a **recursively defined** object — a node joining subtrees that "
            "are themselves trees:\n\n"
            "$$\\mathrm{Tree} = \\mathrm{Node}(\\underbrace{\\mathrm{Tree}, \\dots, "
            "\\mathrm{Tree}}_{b}).$$\n\n"
            "This equation is a fixed point of the branching functor: the type on the "
            "left appears unchanged on the right. Unrolling it gives self-similarity at "
            "every scale, the natural-world face of $f(x)=x$."
        ),
        "image_prompt": (
            "A striking photograph of a single leafless tree against a pale sky in winter, "
            "seen so its branching is fully visible. The trunk splits into a few boughs; "
            "each bough splits the same way into smaller boughs; each of those into finer "
            "twigs — the identical branching pattern repeating at every scale, clearly "
            "self-similar and fractal. Elegant, high-contrast, fine detail in the "
            "silhouette. No text, no words, no letters."
        ),
        "size": "1024x1024",
        "answer": "recursion / self-similarity: Tree = Node(Tree,…,Tree), a fixed point",
    },
}


def gen_image(sample_id: str, spec: dict) -> None:
    d = ROOT / sample_id
    d.mkdir(parents=True, exist_ok=True)
    # write the non-image voices
    schema = {
        "sample_id": sample_id,
        "family": spec["family"],
        "structure": spec["structure"],
        "concept": spec["concept"],
        "answer": {"insight": spec["answer"]},
    }
    (d / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    (d / "story.txt").write_text(spec["story"])
    (d / "math.md").write_text(spec["math"])
    (d / "image_prompt.txt").write_text(spec["image_prompt"])

    out = d / "image.png"
    if out.exists():
        print(f"  skip {sample_id} (image exists)")
        return
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/images/generations",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json",
           "-d", json.dumps({"model": MODEL, "prompt": spec["image_prompt"],
                             "size": spec["size"], "quality": "high", "n": 1})]
    resp = None
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            resp = json.loads(raw)
            if "data" in resp:
                break
        except json.JSONDecodeError:
            pass
        print(f"  retry {sample_id} (attempt {attempt + 1})")
    if not resp or "data" not in resp:
        print(f"  FAIL {sample_id}: {json.dumps(resp)[:200] if resp else 'no response'}")
        return
    item = resp["data"][0]
    if item.get("b64_json"):
        out.write_bytes(base64.b64decode(item["b64_json"]))
    else:  # some proxies return a URL
        subprocess.run(["curl", "-s", "-o", str(out), item["url"]], check=True)
    print(f"  ok {sample_id}/image.png")


def main():
    ids = sys.argv[1:] or list(SAMPLES)
    for sid in ids:
        gen_image(sid, SAMPLES[sid])


if __name__ == "__main__":
    main()
