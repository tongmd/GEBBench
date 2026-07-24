#!/usr/bin/env python3
"""Exp1 — evaluate nuisance-resample renders as scene2motif items.

Adapted from eval/run_eval.py: same prompt, same 25-way menu, same JSON
parse.  Takes an --items file (exp1_geometry/items_renders.json) and writes
to exp1_geometry/results/<tag>.json so analyze.py can pool the extra
confusion events with the original eval/results runs.

Usage:
  python3 exp1_geometry/run_renders.py --items exp1_geometry/items_renders.json \
      --base $OPENAI_BASE_URL --model gpt-5.5 --tag api_gpt-5.5_renders
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

ROOT = Path(__file__).resolve().parent.parent
MENU = json.loads((ROOT / "eval" / "items.json").read_text())["menu"]

PROMPT = (
    "这张图像的构图体现了一个抽象结构母题。从下面的选项中选出它体现的母题。\n\n"
    "选项：\n{menu}\n\n只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}")


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def call(base: str, key: str, model: str, content, max_retries=4) -> str | None:
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
                print("  ERR", e, flush=True)
                return None
            time.sleep(5 * (attempt + 1))
    return None


def parse(reply: str | None) -> str | None:
    if not reply:
        return None
    m = re.search(r'"motif_id"\s*:\s*"([^"]+)"', reply)
    return m.group(1).strip() if m else None


def run_item(args, item) -> dict:
    prompt = PROMPT.format(menu=MENU)
    content = [
        {"type": "image_url", "image_url": {"url":
            "data:image/png;base64," + img_b64(ROOT / item["image"])}},
        {"type": "text", "text": prompt},
    ]
    reply = call(args.base, args.key, args.model, content)
    pred = parse(reply)
    return {"id": item["id"], "task": item["task"], "answer": item["answer"],
            "pred": pred, "correct": pred == item["answer"],
            "raw": (reply or "")[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    items = json.loads(Path(args.items).read_text())["items"]
    print(f"{args.model} @ {args.base}: {len(items)} items", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(lambda it: run_item(args, it), items))

    # retry items whose reply failed entirely (pred None + empty raw)
    for k, r in enumerate(records):
        if r["pred"] is None and not r["raw"]:
            print("retrying", r["id"], flush=True)
            records[k] = run_item(args, items[k])

    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    n_null = sum(1 for r in records if r["pred"] is None)
    out = {"model": args.model, "base": args.base,
           "summary": {"scene2motif": {"n": n, "acc": round(acc, 4),
                                       "n_unparsed": n_null}},
           "records": records}
    tag = args.tag or re.sub(r"[^\w.-]", "_", args.model) + "_renders"
    outf = ROOT / "exp1_geometry" / "results" / f"{tag}.json"
    outf.parent.mkdir(exist_ok=True)
    outf.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["summary"], indent=1), "->", outf, flush=True)


if __name__ == "__main__":
    main()
