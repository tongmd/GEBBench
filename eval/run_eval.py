#!/usr/bin/env python3
"""GEB benchmark — run evals against OpenAI-compatible endpoints.

Works with vLLM (kiti, via ssh tunnel) and API proxies (yunwu).  Text tasks go
to any chat model; image tasks require a vision-capable model and are skipped
otherwise (--vision to enable).

Usage:
  python3 eval/run_eval.py --base http://localhost:18001/v1 --model NAME \
      [--vision] [--tasks story2motif,theorem2motif] [--limit N] [--tag TAG]

Results -> eval/results/<tag or model>.json  (per-item records + accuracy)
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
DATA = json.loads((ROOT / "eval" / "items.json").read_text())
MENU = DATA["menu"]

TEXT_TASKS = {"story2motif", "theorem2motif"}
VISION_TASKS = {"scene2motif", "skeleton2motif", "crossvoice"}

PROMPTS = {
    "scene2motif": (
        "这张图像的构图体现了一个抽象结构母题。从下面的选项中选出它体现的母题。\n\n"
        "选项：\n{menu}\n\n只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}"),
    "skeleton2motif": (
        "这张示意图画的是一个抽象结构母题。从下面的选项中选出它。\n\n"
        "选项：\n{menu}\n\n只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}"),
    "story2motif": (
        "下面这个小故事的讲法与情节骨架实现了一个抽象结构母题。选出它。\n\n"
        "故事：\n{text}\n\n选项：\n{menu}\n\n"
        "只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}"),
    "theorem2motif": (
        "下面这条数学定理（部分专名已用□遮盖）刻画了一个抽象结构母题。选出它。\n\n"
        "定理：{text}\n\n选项：\n{menu}\n\n"
        "只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}"),
    "crossvoice": (
        "这张图像的构图体现了某个抽象结构。下面四个小故事中，恰有一个的讲法"
        "实现了与图像相同的结构。选出它。\n\n{options}\n\n"
        "只输出 JSON（不要代码块）：{{\"choice\": \"A/B/C/D 之一\"}}"),
}


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


def parse(reply: str | None, field: str) -> str | None:
    if not reply:
        return None
    m = re.search(r'"%s"\s*:\s*"([^"]+)"' % field, reply)
    if m:
        return m.group(1).strip()
    # fallback: last A-D letter or bare id
    if field == "choice":
        letters = re.findall(r"\b([A-D])\b", reply)
        return letters[-1] if letters else None
    return None


def run_item(args, item) -> dict:
    task = item["task"]
    if task == "crossvoice":
        opts = "\n\n".join(f"【{k}】\n{v}" for k, v in item["options"].items())
        prompt = PROMPTS[task].format(options=opts)
    else:
        prompt = PROMPTS[task].format(menu=MENU, text=item.get("text", ""))
    if task in VISION_TASKS:
        content = [
            {"type": "image_url", "image_url": {"url":
                "data:image/png;base64," + img_b64(ROOT / item["image"])}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    reply = call(args.base, args.key, args.model, content)
    field = "choice" if task == "crossvoice" else "motif_id"
    pred = parse(reply, field)
    return {"id": item["id"], "task": task, "answer": item["answer"],
            "pred": pred, "correct": pred == item["answer"],
            "raw": (reply or "")[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tasks = set(args.tasks.split(",")) if args.tasks else (
        TEXT_TASKS | (VISION_TASKS if args.vision else set()))
    items = [i for i in DATA["items"] if i["task"] in tasks]
    if args.limit:
        items = items[:args.limit]
    print(f"{args.model} @ {args.base}: {len(items)} items, tasks={sorted(tasks)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(lambda it: run_item(args, it), items))

    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)
    summary = {t: {"n": len(rs),
                   "acc": round(sum(r["correct"] for r in rs) / len(rs), 4)}
               for t, rs in by_task.items()}
    out = {"model": args.model, "base": args.base, "summary": summary,
           "records": records}
    tag = args.tag or re.sub(r"[^\w.-]", "_", args.model)
    outf = ROOT / "eval" / "results" / f"{tag}.json"
    outf.parent.mkdir(exist_ok=True)
    outf.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1), "->", outf)


if __name__ == "__main__":
    main()
