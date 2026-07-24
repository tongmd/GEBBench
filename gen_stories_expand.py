#!/usr/bin/env python3
"""Expand the story voice: 5 dual-screened stories per motif (was 1).

Reuses gen_stories.py wholesale (writer prompt, device judge, blind judge,
rewrite loop) with three changes:
  - models pinned to the :stable channel on yunwu
  - per-slot domain-freshness constraint built from already-accepted stories
  - incremental, resumable output keyed by story_id = <motif>_s<k>

data/stories_v2.json (the 25 pilot stories) is imported as slot s0 and never
rewritten.  Output: data/stories_v3.json (list; each entry adds story_id).

Usage:  python3 gen_stories_expand.py [--per-motif 5] [motif_id ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gen_stories as G

G.WRITER_MODEL = "gpt-5.5:stable"
G.JUDGE_MODEL = "gpt-5.2:stable"

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "stories_v2.json"
OUT = ROOT / "data" / "stories_v3.json"
LOCK = threading.Lock()


def load_state() -> dict[str, dict]:
    """story_id -> story object."""
    state: dict[str, dict] = {}
    if OUT.exists():
        for s in json.loads(OUT.read_text()):
            state[s["story_id"]] = s
    else:
        for s in json.loads(V2.read_text()):
            s = dict(s, story_id=f"{s['motif_id']}_s0")
            state[s["story_id"]] = s
    return state


def save_state(state: dict[str, dict]) -> None:
    order = list(G.LIBRARY)
    items = sorted(state.values(),
                   key=lambda s: (order.index(s["motif_id"]), s["story_id"]))
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    tmp.rename(OUT)


def gen_slot(job: tuple[str, int, dict[str, dict]]) -> None:
    mid, k, state = job
    sid = f"{mid}_s{k}"
    # freshness: avoid domains of every already-accepted story for this motif
    with LOCK:
        used = [s.get("domain", "") for s in state.values()
                if s["motif_id"] == mid and s.get("domain")]
    base_avoid = G.AVOID.get(mid, "")
    avoid_list = "、".join(x for x in ([base_avoid] + used) if x)
    G.AVOID[mid] = avoid_list  # gen_one reads AVOID at call time
    for cycle in range(2):  # gen_one itself rewrites up to 4x per cycle
        item = G.gen_one(mid)
        if item and (item.get("judge") or {}).get("passed"):
            break
        print(f"  slot {sid}: cycle {cycle + 1} failed screening, "
              f"{'retrying fresh' if cycle == 0 else 'keeping last'}")
    if not item:
        print(f"  slot {sid}: no output, giving up")
        return
    item["story_id"] = sid
    with LOCK:
        state[sid] = item
        save_state(state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-motif", type=int, default=5)
    ap.add_argument("ids", nargs="*")
    args = ap.parse_args()
    ids = args.ids or list(G.LIBRARY)

    state = load_state()
    save_state(state)  # materialize v3 with s0 imports on first run
    jobs = []
    for mid in ids:
        for k in range(1, args.per_motif):
            sid = f"{mid}_s{k}"
            done = state.get(sid)
            if done and (done.get("judge") or {}).get("passed"):
                continue
            jobs.append((mid, k, state))
    # interleave by slot index so same-motif slots seldom run concurrently
    # (each sees the domains of previously accepted siblings)
    jobs.sort(key=lambda j: (j[1], j[0]))
    print(f"{len(jobs)} story slots to fill "
          f"(writer {G.WRITER_MODEL}, judge {G.JUDGE_MODEL})")
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(gen_slot, jobs))

    passed = sum(1 for s in state.values()
                 if (s.get("judge") or {}).get("passed"))
    print(f"\n{len(state)} stories total, {passed} passed screening -> {OUT}")
    print("STORIES_EXPAND_DONE")


if __name__ == "__main__":
    main()
