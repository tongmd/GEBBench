#!/usr/bin/env python3
"""GEB benchmark — stylized-image generation via gpt-image (images/edits).

For each example, sends canonical.png plus a structure-explicit prompt and
saves style_gpt_<style>.png.  Structure preservation is spelled out in the
prompt from the schema itself (node/edge lists, depth, group order).

Usage:  python3 gen_stylized.py [examples_dir]
Env:    OPENAI_API_KEEY / OPENAI_BASE_URL (already set in shell)
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL = "gpt-image-2"

STYLES = {
    "graph_world": [
        ("chalkboard", "a hand-drawn white chalk drawing on a dark green chalkboard, "
                       "chalk texture, slightly dusty"),
        ("papercut", "a paper cut-out collage: nodes and arrows as flat cut paper shapes "
                     "with subtle drop shadows on a craft-paper background"),
    ],
    "recursive_garden": [
        ("botanical_ink", "a vintage botanical ink illustration on cream paper, "
                          "fine ink linework"),
        ("woodcut", "a monochrome woodcut print, bold carved strokes, paper grain"),
    ],
    "symmetry_lab": [
        ("ceramic_tile", "a glazed ceramic tile design, flat enamel colors with thin "
                         "dark outlines"),
        ("neon_sign", "a neon sign on a dark wall: glowing tubes tracing the shapes"),
    ],
}


def structure_clause(schema: dict) -> str:
    fam = schema["family"]
    if fam == "graph_world":
        nodes = [n["id"] for n in schema["nodes"]]
        arrows = ", ".join(f"{u} to {v}" for u, v in schema["edges"])
        return (f"the same {len(nodes)} circular nodes labeled {', '.join(nodes)} in the "
                f"same positions, and exactly these {len(schema['edges'])} arrows, each "
                f"pointing the same way: {arrows}. Keep every letter label legible. "
                "Do not add, remove, merge or move any node or arrow")
    if fam == "recursive_garden":
        d = schema["depth"]
        return (f"the same recursive binary tree: one trunk, every branch splits into "
                f"exactly two shorter child branches, {d} levels of splitting, "
                f"{2 ** d} tips in the same positions. Do not add, remove or merge any "
                "branch, keep all branching angles and proportions")
    if fam == "symmetry_lab":
        k = schema["answer"]["rotation_order"]
        return (f"the same {k} copies of the same asymmetric arrow shape arranged around "
                f"the center at rotations of 0, 90, 180 and 270 degrees, one copy visually "
                "distinguished as the base element. Keep every copy's position and "
                "orientation exactly; do not mirror, add or remove any copy")
    raise ValueError(fam)


def canvas_size(schema: dict) -> str:
    return "1024x1024" if schema["family"] == "symmetry_lab" else "1536x1024"


def stylize(sample_dir: Path) -> list[str]:
    schema = json.loads((sample_dir / "schema.json").read_text())
    made = []
    for slug, style_desc in STYLES[schema["family"]]:
        out = sample_dir / f"style_gpt_{slug}.png"
        if out.exists():
            made.append(out.name)
            continue
        prompt = (f"Redraw this exact diagram as {style_desc}. "
                  f"CRITICAL, keep exactly the same structure: {structure_clause(schema)}. "
                  "No extra decorations, no text other than the existing labels.")
        cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
               f"{os.environ['OPENAI_BASE_URL']}/images/edits",
               "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
               "-F", f"model={MODEL}",
               "-F", f"image[]=@{sample_dir / 'canonical.png'}",
               "-F", f"prompt={prompt}",
               "-F", f"size={canvas_size(schema)}",
               "-F", "quality=medium"]
        resp = None
        for attempt in range(3):
            raw = subprocess.run(cmd, capture_output=True, text=True).stdout
            try:
                resp = json.loads(raw)
                break
            except json.JSONDecodeError:
                print(f"  retry {sample_dir.name}/{slug} (attempt {attempt + 1}, "
                      f"empty/bad response)")
        if resp is None or "data" not in resp:
            print(f"  FAIL {sample_dir.name}/{slug}: "
                  f"{json.dumps(resp)[:200] if resp else 'no response'}")
            continue
        out.write_bytes(base64.b64decode(resp["data"][0]["b64_json"]))
        made.append(out.name)
        print(f"  ok {sample_dir.name}/{out.name}")
    return made


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "examples"
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        stylize(d)


if __name__ == "__main__":
    main()
