#!/usr/bin/env python3
"""Generate the expansion scene images from data/scene_prompts_v2.json.

Stage (b) of the scene expansion.  Each accepted prompt tuple becomes a
gpt-image call with its skeleton as reference (gen_motifs.gpt_edit), wrapped
REF + body + NO_DIAGRAM exactly like the pilot scenes.  Resumable: skips any
scene_<name>.png that already exists.  Outer retry with backoff on top of
gpt_edit's inner 3 attempts.

Usage:  python3 gen_scenes_expand.py [--workers 4] [motif_id ...]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gen_motifs as M

M.MODEL = "gpt-image-2:stable"  # bare gpt-image-2 hits saturated upstreams

ROOT = Path(__file__).parent
PROMPTS = ROOT / "data" / "scene_prompts_v2.json"


def gen_one(job) -> bool:
    mid, cand = job
    d = ROOT / "motifs" / mid
    out = d / f"scene_{cand['name']}.png"
    if out.exists():
        print(f"  skip {mid}/{out.name}")
        return True
    ref = d / f"skeleton_{cand['skel']}.png"
    if not ref.exists():
        print(f"  MISSING REF {mid}/{ref.name}")
        return False
    prompt = M.REF + cand["prompt_body"].strip() + " " + M.NO_DIAGRAM
    for attempt in range(6):  # saturated-upstream errors need patience
        if M.gpt_edit(ref, prompt, cand["size"], out):
            print(f"  ok {mid}/{out.name}")
            return True
        time.sleep(min(30 * (attempt + 1), 120))
    print(f"  GIVEUP {mid}/{out.name}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("ids", nargs="*")
    args = ap.parse_args()

    state = json.loads(PROMPTS.read_text())
    ids = args.ids or list(state)
    jobs = [(mid, cand) for mid in ids for cand in state.get(mid, [])]
    todo = [(mid, c) for mid, c in jobs
            if not (ROOT / "motifs" / mid / f"scene_{c['name']}.png").exists()]
    print(f"{len(todo)} scenes to generate ({len(jobs) - len(todo)} done), "
          f"model {M.MODEL}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(gen_one, todo))
    print(f"\n{sum(results)}/{len(todo)} succeeded")
    print("SCENES_EXPAND_DONE")


if __name__ == "__main__":
    main()
