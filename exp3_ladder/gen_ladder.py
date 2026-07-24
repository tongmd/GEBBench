#!/usr/bin/env python3
"""Experiment ③ — psychophysical function of abstraction.

Generates the 7-level skeleton→photoreal naturalization ladder:
  L0  the programmatic skeleton PNG itself (no generation)
  L1  pencil/ink/charcoal redraw of the diagram
  L2  stylized flat-color infographic
  L3  minimal vector illustration (structure dominant)
  L4  illustration embedded in a simple scene
  L5  naturalistic photo-like image, moderate clutter
  L6  photoreal cluttered, structure must be found

25 motifs × 6 levels (L1–L6) × 3 seeds = 450 images via gpt-image-2
/images/edits with the motif skeleton as reference.  Seeds vary the medium
(L1–L2) or the motif's scene idea (L3–L6).

Usage:
  python3 exp3_ladder/gen_ladder.py [--motifs a,b] [--levels 1,2] \
      [--seeds 0,1,2] [--workers 4] [--dry-run] [--force id/Lk_sN ...]

Outputs:
  exp3_ladder/images/<motif>/L<k>_s<seed>.png
  exp3_ladder/manifest.json   (every attempted job + prompt + status)
  exp3_ladder/failures.json   (jobs that failed after retries)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import gen_motifs  # noqa: E402  (skeleton registry + library)

EXP = ROOT / "exp3_ladder"
IMAGES = EXP / "images"
MODEL = "gpt-image-2"

LIBRARY = gen_motifs.LIBRARY
SKELETON_PNG = {mid: ROOT / "motifs" / mid / f"skeleton_{next(iter(names))}.png"
                for mid, names in gen_motifs.SKELETONS.items()}

REF = ("Use the attached diagram ONLY as a compositional skeleton: the "
       "composition of the output must follow the drawn structure exactly — "
       "every connection, containment, crossing, repetition and exception "
       "preserved, nothing structural added or removed. ")
NO_TEXT = "No text, no words, no letters, no numbers."
NO_DIAGRAM = ("It must look like a real photograph, NOT a diagram: no visible "
              "diagram lines, no floating abstract shapes. " + NO_TEXT)

# seed-varying media / styles for the diagram-like levels
L1_MEDIA = [
    "a light graphite pencil sketch on slightly grainy sketchbook paper",
    "a fine fountain-pen ink drawing on cream laid paper",
    "a soft charcoal sketch on rough toned paper",
]
L2_STYLES = [
    "a modern corporate brand guide (muted blue, slate grey and one warm accent)",
    "a mid-century science textbook plate (ochre, teal and off-white)",
    "a contemporary flat-design app illustration (soft pastel fills)",
]

# ONE template per level; {scene} = motif scene idea, {medium}/{style} vary by seed
TEMPLATES = {
    1: (REF + "Redraw this exact diagram as {medium}. Reproduce every line, "
        "node, arrow and their exact arrangement faithfully — same shapes, "
        "same topology, same layout; the only change is the slight hand-made "
        "texture of the medium. Plain paper background, nothing else in the "
        "frame. " + NO_TEXT),
    2: (REF + "Redraw this exact diagram as a clean, flat-color minimalist "
        "infographic in the style of {style}. Keep exactly the same structure "
        "and layout, but render the elements as crisp geometric shapes with "
        "flat fills, rounded strokes and a subtle background tint — a "
        "designed diagram, not a photo. " + NO_TEXT),
    3: (REF + "Turn this diagram into a MINIMAL flat vector illustration of "
        "the following scene: {scene}. Render ONLY the objects needed to "
        "realize the scene's structure, arranged exactly along the drawn "
        "skeleton, on a plain single-color background with no other elements. "
        "Flat colors, simple shapes, structure dominant. " + NO_TEXT),
    4: (REF + "Turn this diagram into a stylized illustration of the "
        "following scene: {scene}. Embed the structure in a simple setting — "
        "a ground plane, a sky or room, and two or three supporting "
        "background elements — but the structure-bearing objects remain the "
        "clear focal subject, arranged exactly along the drawn skeleton. "
        "Painted illustration look. " + NO_TEXT),
    5: (REF + "Create a naturalistic, photorealistic image of the following "
        "scene: {scene}. Real materials, natural light, and a moderate "
        "amount of ordinary surrounding detail from the scene's world; the "
        "structure is carried by the composition of the main elements, which "
        "follow the drawn skeleton exactly. " + NO_DIAGRAM),
    6: (REF + "Create a richly detailed photorealistic image of the "
        "following scene: {scene}. A dense, busy environment: many "
        "surrounding objects, textures, bystanders and incidental detail "
        "from the scene's world fill the frame, so the structure must be "
        "found within the clutter — yet the scene's key elements still "
        "realize the drawn composition exactly. " + NO_DIAGRAM),
}


def prompt_for(mid: str, level: int, seed: int) -> str:
    tpl = TEMPLATES[level]
    if level == 1:
        return tpl.format(medium=L1_MEDIA[seed % 3])
    if level == 2:
        return tpl.format(style=L2_STYLES[seed % 3])
    scene = LIBRARY[mid]["scene_ideas"][seed % 3]
    return tpl.format(scene=scene)


def size_for(mid: str) -> str:
    with Image.open(SKELETON_PNG[mid]) as im:
        w, h = im.size
    return "1536x1024" if w > h else "1024x1024"


def valid_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def gpt_edit(ref: Path, prompt: str, size: str, out: Path) -> str | None:
    """Returns None on success, error string on failure.  3 retries."""
    cmd = ["curl", "-s", "--max-time", "420", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/images/edits",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-F", f"model={MODEL}", "-F", f"image[]=@{ref}",
           "-F", f"prompt={prompt}", "-F", f"size={size}", "-F", "quality=high"]
    err = "no attempts"
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            err = f"non-JSON response ({len(raw)} bytes)"
            time.sleep(5 * (attempt + 1))
            continue
        if "data" in resp:
            out.write_bytes(base64.b64decode(resp["data"][0]["b64_json"]))
            if valid_png(out):
                return None
            out.unlink(missing_ok=True)
            err = "invalid PNG after decode"
        else:
            err = json.dumps(resp)[:300]
            if "rate" in err.lower() or "429" in err:
                time.sleep(20 * (attempt + 1))
                continue
        time.sleep(5 * (attempt + 1))
    return err


LOCK = threading.Lock()


def run_job(job, manifest, failures):
    mid, level, seed = job
    out = IMAGES / mid / f"L{level}_s{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    p = prompt_for(mid, level, seed)
    rec = {"motif": mid, "level": level, "seed": seed,
           "path": str(out.relative_to(ROOT)), "prompt": p}
    if valid_png(out):
        rec["status"] = "exists"
    else:
        err = gpt_edit(SKELETON_PNG[mid], p, size_for(mid), out)
        rec["status"] = "ok" if err is None else "fail"
        if err:
            rec["error"] = err
    with LOCK:
        manifest.append(rec)
        if rec["status"] == "fail":
            failures.append(rec)
            print(f"  FAIL {mid}/L{level}_s{seed}: {rec['error'][:120]}",
                  flush=True)
        elif rec["status"] == "ok":
            print(f"  ok   {mid}/L{level}_s{seed}", flush=True)
        done = len(manifest)
        if done % 10 == 0:
            print(f"[{done} jobs done]", flush=True)
            save(manifest, failures)
    return rec


def save(manifest, failures):
    (EXP / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False))
    (EXP / "failures.json").write_text(
        json.dumps(failures, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motifs", default="")
    ap.add_argument("--levels", default="1,2,3,4,5,6")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", nargs="*", default=[],
                    help="motif/Lk_sN entries to regenerate even if present")
    args = ap.parse_args()

    motifs = args.motifs.split(",") if args.motifs else list(LIBRARY)
    levels = [int(x) for x in args.levels.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    for f in args.force:
        p = IMAGES / (f if f.endswith(".png") else f + ".png")
        p.unlink(missing_ok=True)

    jobs = [(m, l, s) for m in motifs for l in levels for s in seeds]
    pending = [(m, l, s) for m, l, s in jobs
               if not valid_png(IMAGES / m / f"L{l}_s{s}.png")]
    print(f"{len(jobs)} jobs, {len(pending)} to generate "
          f"({len(jobs) - len(pending)} already valid)", flush=True)
    if args.dry_run:
        for m, l, s in pending[:8]:
            print(f"--- {m} L{l} s{s} ---\n{prompt_for(m, l, s)}\n")
        return

    manifest, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda j: run_job(j, manifest, failures), jobs))
    save(manifest, failures)
    ok = sum(r["status"] in ("ok", "exists") for r in manifest)
    print(f"DONE: {ok}/{len(manifest)} valid, {len(failures)} failures",
          flush=True)


if __name__ == "__main__":
    main()
