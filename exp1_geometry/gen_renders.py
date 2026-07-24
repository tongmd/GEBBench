#!/usr/bin/env python3
"""Exp1 (Formal Geometry, Not Pixel Geometry) — nuisance-resample renders.

Brings every motif to >=5 renders total (existing motifs/<id>/scene_*.png
count toward the quota).  Each new render = gpt-image-2 /images/edits with
the motif's skeleton PNG as reference image, prompt = one of the motif's
scene_ideas (data/motif_library.json) realized under a varied medium/palette.

Output: exp1_geometry/renders/<motif>/resample_XX.png
Robust: 3 retries per image, PNG validated with PIL, idempotent (skips
existing valid files).  Run with THREADS=4 parallel workers.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exp1_geometry" / "renders"
MODEL = "gpt-image-2"
TARGET = 5
THREADS = 4

LIB = {m["id"]: m for m in
       json.loads((ROOT / "data" / "motif_library.json").read_text())}

NO_DIAGRAM = ("It must look like a real photograph or a piece of art, NOT a "
              "diagram: no visible diagram lines, no text, no letters.")
REF = "Use this diagram ONLY as a compositional skeleton. "

# medium / palette nuisance axes — cycled so same idea never repeats a medium
MEDIA = [
    "Render it as a photorealistic photograph in natural daylight, neutral "
    "true-to-life palette.",
    "Render it as a photorealistic photograph at golden hour, warm amber "
    "palette, long soft shadows.",
    "Render it as a fine traditional ink-and-wash painting with muted earth "
    "tones on aged paper.",
    "Render it as a crisp studio photograph against a dark background, cool "
    "blue-grey palette, dramatic side light.",
    "Render it as a detailed woodblock print with a restrained two-tone "
    "palette on textured paper.",
]


def skeleton_png(mid: str) -> Path:
    ps = sorted((ROOT / "motifs" / mid).glob("skeleton_*.png"))
    assert ps, f"no skeleton for {mid}"
    return ps[0]


def valid_png(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 10_000:
        return False
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def gpt_edit(ref: Path, prompt: str, size: str, out: Path) -> bool:
    cmd = ["curl", "-s", "--max-time", "420", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/images/edits",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-F", f"model={MODEL}", "-F", f"image[]=@{ref}",
           "-F", f"prompt={prompt}", "-F", f"size={size}", "-F", "quality=high"]
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  retry {out.parent.name}/{out.name} (attempt {attempt+1})",
                  flush=True)
            continue
        if "data" in resp:
            out.write_bytes(base64.b64decode(resp["data"][0]["b64_json"]))
            if valid_png(out):
                return True
            out.unlink(missing_ok=True)
            print(f"  invalid png {out.name}, retrying", flush=True)
            continue
        print(f"  FAIL {out.parent.name}/{out.name}: "
              f"{json.dumps(resp)[:200]}", flush=True)
    return False


def jobs():
    js = []
    for mid, m in LIB.items():
        existing = len(list((ROOT / "motifs" / mid).glob("scene_*.png")))
        need = max(0, TARGET - existing)
        d = OUT / mid
        d.mkdir(parents=True, exist_ok=True)
        skel = skeleton_png(mid)
        with Image.open(skel) as im:
            w, h = im.size
        size = f"{w}x{h}"
        ideas = m["scene_ideas"]
        for k in range(need):
            out = d / f"resample_{k:02d}.png"
            if valid_png(out):
                continue
            idea = ideas[k % len(ideas)]
            medium = MEDIA[(k + existing) % len(MEDIA)]
            prompt = (REF + "Create an image realizing this concept: "
                      + idea + " " + medium + " The composition must follow "
                      "the drawn skeleton exactly — the structure IS the "
                      "picture. " + NO_DIAGRAM)
            js.append((skel, prompt, size, out))
    return js


def main():
    js = jobs()
    print(f"{len(js)} renders to generate", flush=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = {ex.submit(gpt_edit, *j): j for j in js}
        for f in as_completed(futs):
            skel, prompt, size, out = futs[f]
            if f.result():
                ok += 1
                print(f"  ok {out.parent.name}/{out.name} [{ok+fail}/{len(js)}]",
                      flush=True)
            else:
                fail += 1
                print(f"  FAILED {out.parent.name}/{out.name}", flush=True)
    print(f"done: {ok} ok, {fail} failed", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
