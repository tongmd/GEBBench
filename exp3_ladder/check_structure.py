#!/usr/bin/env python3
"""Experiment ③ — independent structure check for L5/L6 generated images.

Two-step blind check per image (gpt-image silently breaks topology, so we
verify with an independent describer + judge):

  1. DESCRIBE (vision, blind to the motif): "describe the compositional
     structure of this image in one sentence".
  2. JUDGE (text-only): does that description realize the motif's invariant?

Images judged non-matching are flagged in exp3_ladder/flags.json.
--regen deletes flagged images and regenerates them ONCE (marks them as
having used their regeneration), after which re-running the checker
re-judges them; still-failing images stay flagged for transparency.

Usage:
  python3 exp3_ladder/check_structure.py [--model gpt-5.2] [--workers 8]
  python3 exp3_ladder/check_structure.py --regen   # regenerate flagged once
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import gen_motifs  # noqa: E402

EXP = ROOT / "exp3_ladder"
LIBRARY = gen_motifs.LIBRARY

DESCRIBE = ("Describe the compositional structure of this image in one "
            "sentence: how the main elements are arranged and related "
            "(repetition, containment, connection, crossing, exception...), "
            "not what they depict.")
JUDGE = ("An image was generated to realize this abstract structural "
         "invariant through its composition:\n\n【invariant】{inv}\n\n"
         "A blind observer described the image's composition as:\n\n"
         "【description】{desc}\n\n"
         "Does the described composition plausibly realize the invariant? "
         "Judge structure only, not subject matter. Output JSON only "
         '(no code block): {{"match": true/false, "reason": "..."}}')


def call(base, key, model, content, max_retries=3):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 2048,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                print("  ERR", e)
                return None
            time.sleep(3 * (attempt + 1))
    return None


def check_image(args, path: Path, mid: str) -> dict:
    rec = {"image": str(path.relative_to(ROOT)), "motif": mid}
    b64 = base64.b64encode(path.read_bytes()).decode()
    desc = call(args.base, args.key, args.model, [
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text", "text": DESCRIBE}])
    rec["description"] = desc
    if not desc:
        rec["match"] = None
        return rec
    verdict = call(args.base, args.key, args.model,
                   JUDGE.format(inv=LIBRARY[mid]["invariant_zh"], desc=desc))
    m = re.search(r'"match"\s*:\s*(true|false)', verdict or "")
    rec["match"] = (m.group(1) == "true") if m else None
    r = re.search(r'"reason"\s*:\s*"([^"]*)"', verdict or "")
    rec["reason"] = r.group(1) if r else (verdict or "")[:200]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--base", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--regen", action="store_true",
                    help="regenerate flagged images once, then exit")
    args = ap.parse_args()

    flags_f = EXP / "flags.json"
    flags = json.loads(flags_f.read_text()) if flags_f.exists() else {
        "checker_model": args.model, "checks": {}, "regenerated": []}

    if args.regen:
        bad = [img for img, rec in flags["checks"].items()
               if rec["match"] is False and img not in flags["regenerated"]]
        print(f"regenerating {len(bad)} flagged images (once each)")
        for img in bad:
            p = ROOT / img
            p.unlink(missing_ok=True)
            m = re.match(r".*/([^/]+)/L(\d)_s(\d)\.png", img)
            mid, level, seed = m.group(1), m.group(2), m.group(3)
            subprocess.run([sys.executable, str(EXP / "gen_ladder.py"),
                            "--motifs", mid, "--levels", level,
                            "--seeds", seed], check=False)
            flags["regenerated"].append(img)
            del flags["checks"][img]     # force re-check
            flags_f.write_text(json.dumps(flags, indent=1, ensure_ascii=False))
        flags_f.write_text(json.dumps(flags, indent=1, ensure_ascii=False))
        print("done — re-run checker without --regen to re-judge")
        return

    todo = []
    for mid in LIBRARY:
        for level in (5, 6):
            for seed in range(3):
                p = EXP / "images" / mid / f"L{level}_s{seed}.png"
                key = str(p.relative_to(ROOT))
                if p.exists() and key not in flags["checks"]:
                    todo.append((p, mid))
    print(f"checking {len(todo)} L5/L6 images with {args.model}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(lambda t: check_image(args, *t), todo):
            flags["checks"][rec["image"]] = rec
            mark = {True: "ok  ", False: "FLAG", None: "??? "}[rec["match"]]
            print(f"  {mark} {rec['image']}")

    flags_f.write_text(json.dumps(flags, indent=1, ensure_ascii=False))
    n = len(flags["checks"])
    bad = sum(1 for r in flags["checks"].values() if r["match"] is False)
    unk = sum(1 for r in flags["checks"].values() if r["match"] is None)
    print(f"{n} checked: {n - bad - unk} pass, {bad} flagged, {unk} unknown "
          f"-> {flags_f}")


if __name__ == "__main__":
    main()
