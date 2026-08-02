#!/usr/bin/env python3
"""Experiment ② — run the comparison conditions against an OpenAI-compatible
endpoint.  Reported results come from the unified inference API for the
API-served models and a local vLLM server for the smaller open VLMs.

Prompt protocol matches eval/run_eval.py (Chinese, 25-way menu, JSON answer).
Wording is IDENTICAL across C1 (single) and across C2-C4 (pairs); C5 adds
only the explicit compare instruction.  Scoring is always a single motif_id.

Usage:
  python3 exp2_comparison/run_comparison.py \
      --base "$OPENAI_BASE_URL" --model gpt-5.5 [--tag TAG] \
      [--conditions single,near] [--limit N] [--workers 8]

Results -> exp2_comparison/results/<tag or model>.json
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

import httpx

ROOT = Path(__file__).parent.parent

# Optional SOCKS/HTTP proxy so calls can egress through a region where the
# frontier models are not geoblocked (set GEB_PROXY=socks5://127.0.0.1:1080).
_client = httpx.Client(proxy=os.environ.get("GEB_PROXY") or None, timeout=300.0)
DATA = json.loads((ROOT / "exp2_comparison" / "items.json").read_text())
MENU = DATA["menu"]

TAIL = "\n\n选项：\n{menu}\n\n只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}"

PROMPTS = {
    # C1 — verbatim the scene2motif prompt from eval/run_eval.py
    "single": "这张图像的构图体现了一个抽象结构母题。从下面的选项中选出它体现的母题。" + TAIL,
    # C2/C3/C4 — identical wording; the answer is defined as the LEFT panel
    "pair": ("这张图里并排放着左右两幅画面。左边画面的构图体现了一个抽象结构母题。"
             "从下面的选项中选出左边画面体现的母题。" + TAIL),
    # C5 — same stimulus as C3, plus the explicit compare step
    "instructed": ("这张图里并排放着左右两幅画面。请先比较这两幅画面，找出它们共享的结构，"
                   "然后回答：左边画面的构图体现了一个抽象结构母题。"
                   "从下面的选项中选出左边画面体现的母题。" + TAIL),
}


def prompt_for(cond: str) -> str:
    key = cond if cond in ("single", "instructed") else "pair"
    return PROMPTS[key].format(menu=MENU)


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def call(base: str, key: str, model: str, content, max_retries=3) -> str | None:
    payload = {"model": model, "temperature": 0.0, "max_tokens": 4096,
               "messages": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            r = _client.post(f"{base}/chat/completions",
                             json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 - transient API failures
            if attempt == max_retries - 1:
                print("  ERR", type(e).__name__, str(e)[:120])
                return None
            time.sleep(3 * (attempt + 1))
    return None


def parse(reply: str | None) -> str | None:
    if not reply:
        return None
    m = re.search(r'"motif_id"\s*:\s*"([^"]+)"', reply)
    return m.group(1).strip() if m else None


def run_item(args, item) -> dict:
    content = [
        {"type": "image_url", "image_url": {"url":
            "data:image/png;base64," + img_b64(ROOT / item["image"])}},
        {"type": "text", "text": prompt_for(item["condition"])},
    ]
    reply = call(args.base, args.key, args.model, content)
    pred = parse(reply)
    return {"id": item["id"], "condition": item["condition"],
            "motif": item["motif"], "answer": item["answer"], "pred": pred,
            "correct": pred == item["answer"], "raw": (reply or "")[:200],
            # A call that never returned is a transport failure, not a wrong
            # answer.  Scoring it as wrong is how a collapsed tunnel once
            # produced a structurally valid file reading acc=0.0 everywhere.
            "empty": reply is None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--conditions", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    items = DATA["items"]
    if args.conditions:
        keep = set(args.conditions.split(","))
        items = [i for i in items if i["condition"] in keep]
    if args.limit:
        items = items[:args.limit]
    print(f"{args.model} @ {args.base}: {len(items)} items")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(lambda it: run_item(args, it), items))

    n_empty = sum(r["empty"] for r in records)
    if n_empty:
        print(f"  !! {n_empty}/{len(records)} calls returned nothing")
    # Refuse to write a scored file when a meaningful share of the run never
    # reached the endpoint: the accuracies would be transport loss, not ability.
    if n_empty > 0.05 * len(records):
        raise SystemExit(
            f"ABORT: {n_empty}/{len(records)} empty replies (>5%). "
            "This is a transport failure, not a result -- fix the connection "
            "and re-run. No file written.")

    by_cond = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r)
    summary = {c: {"n": len(rs),
                   "n_empty": sum(r["empty"] for r in rs),
                   "acc": round(sum(r["correct"] for r in rs) / len(rs), 4)}
               for c, rs in by_cond.items()}
    out = {"model": args.model, "base": args.base, "n_empty": n_empty,
           "summary": summary, "records": records}
    tag = args.tag or re.sub(r"[^\w.-]", "_", args.model)
    outf = ROOT / "exp2_comparison" / "results" / f"{tag}.json"
    outf.parent.mkdir(exist_ok=True)
    outf.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1), "->", outf)


if __name__ == "__main__":
    main()
