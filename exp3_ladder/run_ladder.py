#!/usr/bin/env python3
"""Experiment ③ runner — evaluate the naturalization ladder against any
OpenAI-compatible endpoint.  Adapted from eval/run_eval.py; the scene2motif
prompt is IDENTICAL to the main eval.

Usage:
  python3 exp3_ladder/run_ladder.py --base $OPENAI_BASE_URL --model gpt-5.5 \
      [--levels 0,1,2] [--limit N] [--workers 8] [--tag TAG]

Results are merged into exp3_ladder/results/<tag or model>.json — items
already answered (non-null pred) are skipped, so the runner can be invoked
repeatedly as images become available.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

ROOT = Path(__file__).parent.parent
EXP = ROOT / "exp3_ladder"

# identical to eval/run_eval.py scene2motif
PROMPT = ("这张图像的构图体现了一个抽象结构母题。从下面的选项中选出它体现的母题。\n\n"
          "选项：\n{menu}\n\n只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}")


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def call(base: str, key: str, model: str, content, max_retries=3) -> str | None:
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
            time.sleep(3 * (attempt + 1))
    return None


def parse(reply: str | None) -> str | None:
    if not reply:
        return None
    m = re.search(r'"motif_id"\s*:\s*"([^"]+)"', reply)
    return m.group(1).strip() if m else None


def run_item(args, menu, item) -> dict:
    prompt = PROMPT.format(menu=menu)
    content = [
        {"type": "image_url", "image_url": {"url":
            "data:image/png;base64," + img_b64(ROOT / item["image"])}},
        {"type": "text", "text": prompt},
    ]
    reply = call(args.base, args.key, args.model, content)
    pred = parse(reply)
    return {"id": item["id"], "task": item["task"], "answer": item["answer"],
            "level": item["level"], "seed": item["seed"], "pred": pred,
            "correct": pred == item["answer"], "raw": (reply or "")[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--levels", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data = json.loads((EXP / "items.json").read_text())
    menu, items = data["menu"], data["items"]
    if args.levels:
        lv = {int(x) for x in args.levels.split(",")}
        items = [i for i in items if i["level"] in lv]

    tag = args.tag or re.sub(r"[^\w.-]", "_", args.model)
    outf = EXP / "results" / f"{tag}.json"
    done: dict[str, dict] = {}
    if outf.exists():
        done = {r["id"]: r for r in json.loads(outf.read_text())["records"]
                if r.get("pred") is not None}
    todo = [i for i in items if i["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.model} @ {args.base}: {len(items)} items, "
          f"{len(done)} already done, {len(todo)} to run")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        new = list(ex.map(lambda it: run_item(args, menu, it), todo))
    records = list(done.values()) + new

    by_level = {}
    for r in records:
        by_level.setdefault(r["level"], []).append(r)
    summary = {f"L{l}": {"n": len(rs),
                         "acc": round(sum(r["correct"] for r in rs) / len(rs), 4)}
               for l, rs in sorted(by_level.items())}
    outf.parent.mkdir(exist_ok=True)
    outf.write_text(json.dumps(
        {"model": args.model, "base": args.base, "summary": summary,
         "records": records}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1), "->", outf)


if __name__ == "__main__":
    main()
