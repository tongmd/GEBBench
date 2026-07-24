#!/usr/bin/env python3
"""GEB benchmark — ENGLISH story voice: form devices, mechanically verified.

The English dataset is primary; stories must SATISFY their form device in
English (first-letter acrostics, sentence counts, word-for-word overlaps...),
not merely retell a Chinese story.  Mirrors gen_stories.py:

  writer  gpt-5.5:stable  — writes a folk story satisfying device_en
  judge1  gpt-5.2:stable  — device judge: executes verify_steps mechanically
  judge2  gpt-5.2:stable  — blind judge: story only, picks motif from the
                            25-item ENGLISH menu (invariant_en)
  accept iff device passes AND blind judge names the right motif;
  reject -> rewrite with judges' reasons (up to 4 attempts, 2 cycles).

Inputs : data/devices_en.json  {mid: {device_en, verify_steps, distinguish}}
         data/library_en.json  {mid: {invariant_en, ...}}
Output : data/stories_en.json  [{story_id, motif_id, title, domain, story_en,
                                 why_shape, model, judge}]
Usage  : python3 gen_stories_en.py [--per-motif 5] [motif_id ...]
Resumable: passed slots are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WRITER_MODEL = "gpt-5.5:stable"
JUDGE_MODEL = "gpt-5.2:stable"
ROOT = Path(__file__).parent
LIBRARY = {m["id"]: m for m in
           json.loads((ROOT / "data" / "motif_library.json").read_text())}
OUT = ROOT / "data" / "stories_en.json"
LOCK = threading.Lock()

# foundation files (must exist; wait_for_foundation() can block on them)
DEV_PATH = ROOT / "data" / "devices_en.json"
LIB_PATH = ROOT / "data" / "library_en.json"


def wait_for_foundation(timeout_s: int = 7200) -> tuple[dict, dict]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEV_PATH.exists() and LIB_PATH.exists():
            try:
                dev = json.loads(DEV_PATH.read_text())
                lib = json.loads(LIB_PATH.read_text())
                if len(dev) == 25 and len(lib) == 25:
                    return dev, lib
            except json.JSONDecodeError:
                pass  # mid-write
        print("waiting for devices_en.json + library_en.json ...", flush=True)
        time.sleep(60)
    raise SystemExit("foundation files never appeared")


SYSTEM = (
    "You are a writer of folk tales and a scholar of the dialogues in "
    "Godel, Escher, Bach.  GEB's dialogues endure because form IS content: "
    "the Crab Canon's lines run back on themselves line by line; one voice "
    "of the Sloth Canon is a transformation of another.  Your stories work "
    "the same way: plain, warm, read-aloud folk prose -- but the TELLING "
    "strictly realizes a formal device.  Read straight through, it is a "
    "good story; look back, and the story's shape IS the structure."
)

PROMPT = """Write a short English folk story for the abstract structure below.

Structure:
- Name: {name_en}
- Equation: {equation}
- Invariant: {invariant}

* FORM DEVICE (hard spec -- the telling must satisfy every checkable clause;
a reviewer will verify each one mechanically):
{device}

Verification the reviewer will run:
{verify}

Rules:
1. Every checkable clause must hold exactly -- sentence counts, first
   letters, word-for-word overlaps, labels: no deviations.
2. Within the device, the story must feel natural and warm, with people,
   places and things (names or textures); a child could listen to it.
   The structure lives in the telling and must never be named or explained
   by the narrator or characters.
3. STRICTLY FORBIDDEN anywhere in title or text: mathematical or structural
   vocabulary (structure, cycle, recursion, symmetric, mirror, level,
   infinite, pattern, loop, spiral, nested, and the like).
4. Use a fresh domain of everyday life{avoid}.
5. Except where the device demands it, the story must not return to or
   repeat its own opening.
{feedback}
Output ONLY JSON (no code fences):
{{"title": "the title (if the device constrains it, the device wins)",
  "domain": "one or two words for the life-domain",
  "story_en": "the story text (use newlines if the device requires numbered
sentences / sections / labels)",
  "why_shape": "one sentence, for reviewers, on how the device is realized
(technical terms allowed here)"}}"""

DEVICE_JUDGE_PROMPT = """You are a merciless formal reviewer.  Below are a
form-device spec and a story.  Execute the verification procedure literally
-- count sentences, compare first letters one by one, check word-for-word
overlaps, count words -- exactly as specified.  Judgment calls are not
allowed: if a clause cannot be verified mechanically, it fails.

Tokenization conventions (apply unless the device explicitly overrides):
- A "word" is a maximal run of letters/digits/internal apostrophes-hyphens;
  strip surrounding punctuation and quote marks before comparing words or
  extracting first/last letters ("Beth'" and "Beth" are the same word).
- Sentence boundaries are ., !, ? followed by whitespace/end; a quoted
  sentence inside dialogue counts as part of its carrier sentence unless the
  device says otherwise.
- Letter comparisons are case-insensitive unless the device says otherwise.

Device spec:
{device}

Verification procedure:
{verify}

Story:
{story}

Output ONLY JSON (no code fences):
{{"passed": true or false, "checks": "short list of per-clause results",
"problems": "what fails (empty string if everything passes)"}}"""

BLIND_JUDGE_PROMPT = """Read the story below and judge its FORM -- the shape
of its telling and plot skeleton -- against the option list.

Disambiguation notes for commonly confused shapes:
{distinguish}

Story:
{story}

Options:
{menu}

Output ONLY JSON (no code fences):
{{"motif_id": "one id from the options"}}"""


def chat(model: str, system: str, user: str) -> str | None:
    body = json.dumps({"model": model, "messages": (
        [{"role": "system", "content": system}] if system else []) +
        [{"role": "user", "content": user}]})
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/chat/completions",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json", "-d", body]
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            return json.loads(raw)["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError):
            time.sleep(5 * (attempt + 1))
    return None


def parse_json(text: str) -> dict | None:
    try:
        return json.loads(re.sub(r"^```(json)?|```$", "", text.strip(),
                                 flags=re.M))
    except (json.JSONDecodeError, AttributeError):
        return None


class Gen:
    def __init__(self, devices: dict, lib_en: dict):
        self.dev = devices
        self.lib = lib_en
        self.menu = "\n".join(f"- {mid}: {v['invariant_en']}"
                              for mid, v in lib_en.items())
        self.distinguish = "\n".join(
            f"- {mid}: {d.get('distinguish', '')}"
            for mid, d in devices.items() if d.get("distinguish"))

    def judge_device(self, mid: str, story: str) -> dict | None:
        d = self.dev[mid]
        reply = chat(JUDGE_MODEL, "", DEVICE_JUDGE_PROMPT.format(
            device=d["device_en"],
            verify="\n".join(f"{i+1}. {s}"
                             for i, s in enumerate(d.get("verify_steps", []))),
            story=story))
        return parse_json(reply) if reply else None

    def judge_blind(self, story: str) -> dict | None:
        reply = chat(JUDGE_MODEL, "", BLIND_JUDGE_PROMPT.format(
            distinguish=self.distinguish, story=story, menu=self.menu))
        return parse_json(reply) if reply else None

    def gen_one(self, mid: str, avoid: list[str]) -> dict | None:
        lib = LIBRARY[mid]
        d = self.dev[mid]
        avoid_txt = (f" (already used: {', '.join(avoid)})" if avoid else "")
        feedback = ""
        item = None
        for attempt in range(4):
            reply = chat(WRITER_MODEL, SYSTEM, PROMPT.format(
                name_en=lib["name_en"], equation=lib["equation_katex"],
                invariant=self.lib[mid]["invariant_en"],
                device=d["device_en"],
                verify="\n".join(f"{i+1}. {s}" for i, s in
                                 enumerate(d.get("verify_steps", []))),
                avoid=avoid_txt, feedback=feedback))
            item = parse_json(reply) if reply else None
            if not item or "story_en" not in item:
                print(f"  retry {mid} (attempt {attempt + 1}: bad output)",
                      flush=True)
                continue
            item.update(motif_id=mid, model=WRITER_MODEL)
            dev = self.judge_device(mid, item["story_en"])
            blind = self.judge_blind(item["story_en"])
            if dev is None or blind is None:
                print(f"  {mid}: judge unavailable, accepting unscreened",
                      flush=True)
                item["judge"] = None
                return item
            ok_dev = bool(dev.get("passed"))
            ok_blind = blind.get("motif_id") == mid
            item["judge"] = {"device_passed": ok_dev,
                             "device_checks": dev.get("checks", ""),
                             "blind_motif": blind.get("motif_id"),
                             "model": JUDGE_MODEL,
                             "passed": ok_dev and ok_blind}
            if ok_dev and ok_blind:
                print(f"  ok {mid}: {item.get('title', '?')} "
                      f"[{item.get('domain', '?')}]"
                      + (f" (attempt {attempt + 1})" if attempt else ""),
                      flush=True)
                return item
            reasons = []
            if not ok_dev:
                reasons.append(f"device check failed: "
                               f"{dev.get('problems', '')}")
            if not ok_blind:
                reasons.append(f"blind judge read the shape as "
                               f"'{blind.get('motif_id')}'")
            print(f"  rewrite {mid} (attempt {attempt + 1}): "
                  f"{'; '.join(reasons)[:150]}", flush=True)
            feedback = ("\nProblems with the previous version: "
                        + "; ".join(reasons) +
                        "\nFix them and rewrite, checking the device "
                        "clause by clause as you go.\n")
        print(f"  WARN {mid}: kept last version despite failed screening",
              flush=True)
        return item


def load_state() -> dict[str, dict]:
    if OUT.exists():
        return {s["story_id"]: s for s in json.loads(OUT.read_text())}
    return {}


def save_state(state: dict[str, dict]) -> None:
    order = list(LIBRARY)
    items = sorted(state.values(),
                   key=lambda s: (order.index(s["motif_id"]), s["story_id"]))
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    tmp.rename(OUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-motif", type=int, default=5)
    ap.add_argument("ids", nargs="*")
    args = ap.parse_args()
    ids = args.ids or list(LIBRARY)

    devices, lib_en = wait_for_foundation()
    gen = Gen(devices, lib_en)
    state = load_state()

    jobs = []
    for mid in ids:
        for k in range(args.per_motif):
            sid = f"{mid}_en_s{k}"
            done = state.get(sid)
            if done and (done.get("judge") or {}).get("passed"):
                continue
            jobs.append((mid, k))
    jobs.sort(key=lambda j: (j[1], j[0]))  # interleave motifs
    print(f"{len(jobs)} EN story slots to fill "
          f"(writer {WRITER_MODEL}, judge {JUDGE_MODEL})", flush=True)

    def run_slot(job):
        mid, k = job
        sid = f"{mid}_en_s{k}"
        with LOCK:
            avoid = [s.get("domain", "") for s in state.values()
                     if s["motif_id"] == mid and s.get("domain")]
        for cycle in range(2):
            item = gen.gen_one(mid, avoid)
            if item and (item.get("judge") or {}).get("passed"):
                break
        if not item:
            print(f"  slot {sid}: no output, giving up", flush=True)
            return
        item["story_id"] = sid
        with LOCK:
            state[sid] = item
            save_state(state)

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(run_slot, jobs))

    passed = sum(1 for s in state.values()
                 if (s.get("judge") or {}).get("passed"))
    print(f"\n{len(state)} EN stories, {passed} passed screening -> {OUT}",
          flush=True)
    print("STORIES_EN_DONE", flush=True)


if __name__ == "__main__":
    main()
