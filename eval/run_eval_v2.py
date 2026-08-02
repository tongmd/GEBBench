#!/usr/bin/env python3
"""GEB benchmark v2 — run the expanded eval against OpenAI-compatible endpoints.

Extends eval/run_eval.py:
  - reads eval/items_v2.json (+ --adv adds the adversarial split)
  - new tasks: xv_scene_thm (VLM), xv_story_thm (LLM), scene2motif_family4
  - images downscaled to <=896px JPEG before base64 (some endpoints reject
    large payloads; learned from the pilot's retry pass)
  - results -> eval/results_v2/<tag>.json

All reported v2 results were produced through one unified inference API; pass
its base URL explicitly.  --base has no default so a run cannot silently fall
back to a different serving path.

Usage:
  python3 eval/run_eval_v2.py --base "$OPENAI_BASE_URL" \
      --model gpt-5.5 --tag or_gpt-5.5 --vision [--adv]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

ROOT = Path(__file__).parent.parent

TEXT_TASKS = {"story2motif", "theorem2motif", "xv_story_thm"}
VISION_TASKS = {"scene2motif", "skeleton2motif", "crossvoice",
                "xv_scene_thm", "scene2motif_family4"}
CHOICE_TASKS = {"crossvoice", "xv_scene_thm", "xv_story_thm"}

PROMPTS = {
    "scene2motif": (
        "The composition of this image realizes an abstract structural motif. "
        "Pick the motif it realizes from the options below.\n\n"
        "Options:\n{menu}\n\n"
        "Output JSON only (no code block): {{\"motif_id\": \"...\"}}"),
    "skeleton2motif": (
        "This diagram depicts an abstract structural motif. "
        "Pick it from the options below.\n\n"
        "Options:\n{menu}\n\n"
        "Output JSON only (no code block): {{\"motif_id\": \"...\"}}"),
    "story2motif": (
        "The telling and plot skeleton of the short story below realize an "
        "abstract structural motif. Pick it.\n\n"
        "Story:\n{text}\n\nOptions:\n{menu}\n\n"
        "Output JSON only (no code block): {{\"motif_id\": \"...\"}}"),
    "theorem2motif": (
        "The mathematical theorem below (some proper names masked with □) "
        "characterizes an abstract structural motif. Pick it.\n\n"
        "Theorem: {text}\n\nOptions:\n{menu}\n\n"
        "Output JSON only (no code block): {{\"motif_id\": \"...\"}}"),
    "crossvoice": (
        "The composition of this image realizes an abstract structure. Exactly "
        "one of the four short stories below is told in a way that realizes "
        "the same structure as the image. Pick it.\n\n{options}\n\n"
        "Output JSON only (no code block): {{\"choice\": \"one of A-D\"}}"),
    "xv_scene_thm": (
        "The composition of this image realizes an abstract structure. Exactly "
        "one of the four mathematical theorems below (some proper names masked "
        "with □) characterizes the same structure as the image. "
        "Pick it.\n\n{options}\n\n"
        "Output JSON only (no code block): {{\"choice\": \"one of A-D\"}}"),
    "xv_story_thm": (
        "The telling of the short story below realizes an abstract structure. "
        "Exactly one of the four mathematical theorems that follow (some "
        "proper names masked with □) characterizes the same structure as "
        "the story. Pick it.\n\n"
        "Story:\n{text}\n\n{options}\n\n"
        "Output JSON only (no code block): {{\"choice\": \"one of A-D\"}}"),
    "scene2motif_family4": (
        "The composition of this image realizes an abstract structural motif. "
        "The four motifs below are formally close to one another; exactly one "
        "of them is the one the image realizes. Pick it.\n\n"
        "Options:\n{menu4}\n\n"
        "Output JSON only (no code block): {{\"motif_id\": \"...\"}}"),
}


def img_payload(path: Path, max_side: int = 896, quality: int = 88) -> str:
    """Downscaled JPEG data URL; falls back to raw PNG if PIL is missing."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return ("data:image/jpeg;base64," +
                base64.b64encode(buf.getvalue()).decode())
    except ImportError:
        return ("data:image/png;base64," +
                base64.b64encode(path.read_bytes()).decode())


def call(base: str, key: str, model: str, content, max_retries=5) -> str | None:
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 4096,
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
            time.sleep(5 * (attempt + 1))
    return None


def parse(reply: str | None, field: str) -> str | None:
    if not reply:
        return None
    m = re.search(r'"%s"\s*:\s*"([^"]+)"' % field, reply)
    if m:
        return m.group(1).strip()
    if field == "choice":
        letters = re.findall(r"\b([A-D])\b", reply)
        return letters[-1] if letters else None
    # fallback for models that answer in prose: first motif id mentioned
    ids = re.findall(
        r"\b(self_reference|recursion|nesting|cycle|symmetry_breaking|"
        r"symmetry|alternation|spiral|convergence|bottleneck|strange_loop|"
        r"figure_ground|isomorphism|mobius|braiding|interference|"
        r"aperiodic_tiling|tiling|hub|diagonalization|infinite_descent|"
        r"coprime_orbit|pigeonhole|projective_duality|"
        r"contraction_fixed_point)\b", reply)
    return ids[0] if ids else None


def run_item(args, menu, item) -> dict:
    task = item["task"]
    tpl = PROMPTS[task]
    kw = {"menu": menu, "text": item.get("text", "")}
    if "options" in item:
        kw["options"] = "\n\n".join(f"[{k}]\n{v}"
                                    for k, v in item["options"].items())
    if "menu4" in item:
        kw["menu4"] = "\n".join(f"- {k}: {v}"
                                for k, v in item["menu4"].items())
    prompt = tpl.format(**kw)
    if task in VISION_TASKS:
        content = [
            {"type": "image_url",
             "image_url": {"url": img_payload(ROOT / item["image"])}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    reply = call(args.base, args.key, args.model, content)
    field = "choice" if task in CHOICE_TASKS else "motif_id"
    pred = parse(reply, field)
    return {"id": item["id"], "task": task, "split": item.get("split", "main"),
            "answer": item["answer"], "pred": pred,
            "correct": pred == item["answer"], "raw": (reply or "")[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--adv", action="store_true",
                    help="also run the adversarial split")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data = json.loads((ROOT / "eval" / "items_v2.json").read_text())
    menu, items = data["menu"], data["items"]
    if args.adv:
        items = items + json.loads(
            (ROOT / "eval" / "items_v2_adv.json").read_text())["items"]

    tasks = set(args.tasks.split(",")) if args.tasks else (
        TEXT_TASKS | (VISION_TASKS if args.vision else set()))
    items = [i for i in items if i["task"] in tasks]
    if args.limit:
        items = items[:args.limit]
    print(f"{args.model} @ {args.base}: {len(items)} items, "
          f"tasks={sorted(tasks)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(lambda it: run_item(args, menu, it), items))

    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)
    summary = {t: {"n": len(rs),
                   "acc": round(sum(r["correct"] for r in rs) / len(rs), 4)}
               for t, rs in by_task.items()}
    out = {"model": args.model, "base": args.base, "summary": summary,
           "records": records}
    tag = args.tag or re.sub(r"[^\w.-]", "_", args.model)
    outf = ROOT / "eval" / "results_v2" / f"{tag}.json"
    outf.parent.mkdir(exist_ok=True)
    outf.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1), "->", outf)
    print("EVAL_V2_DONE", tag)


if __name__ == "__main__":
    main()
