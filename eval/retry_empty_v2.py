#!/usr/bin/env python3
"""Retry records whose reply was empty/unparsed (pred=None) in a
results_v2/<tag>.json, in place.

Root cause: run_eval_v2.call() returned HTTP-200 empty content without
retrying, so OpenRouter's intermittent empty-reply failure mode was scored
as wrong. This re-asks exactly the same items (same prompt builder, same
parser, greedy) with a call() that treats empty content as a failure and
retries with backoff.

  python3 eval/retry_empty_v2.py --tag or_kimi-k2.5 \
      --model moonshotai/kimi-k2.5 [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval_v2 as rv  # noqa: E402


MAX_TOKENS = 4096


def call_strict(base, key, model, content, max_retries=6):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": MAX_TOKENS,
                       "messages": [{"role": "user", "content": content}]}).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{base}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            if txt and txt.strip():
                return txt
            # empty content: fall through to retry
        except Exception as e:
            if attempt == max_retries - 1:
                print("  ERR", type(e).__name__, str(e)[:100], flush=True)
                return None
        time.sleep(4 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    global MAX_TOKENS
    MAX_TOKENS = args.max_tokens

    rv.call = call_strict

    data = json.loads((rv.ROOT / "eval" / "items_v2.json").read_text())
    menu = data["menu"]
    items = {i["id"]: i for i in data["items"]}
    adv = json.loads((rv.ROOT / "eval" / "items_v2_adv.json").read_text())
    for i in adv["items"]:
        items[i["id"]] = i

    res_path = rv.ROOT / "eval" / "results_v2" / f"{args.tag}.json"
    d = json.loads(res_path.read_text())
    recs = d["records"] if isinstance(d, dict) else d
    ns = SimpleNamespace(base=args.base, key=args.key, model=args.model)

    todo = [i for i, r in enumerate(recs) if r.get("pred") is None]
    print(f"{args.tag}: {len(todo)} empty/unparsed records to retry",
          flush=True)

    def redo(idx):
        old = recs[idx]
        new = rv.run_item(ns, menu, items[old["id"]])
        return idx, new

    fixed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, new in ex.map(redo, todo):
            if new["pred"] is not None:
                recs[idx] = new
                fixed += 1

    # refresh summary if the file carries one
    if isinstance(d, dict) and "summary" in d:
        by = {}
        for r in recs:
            k, n = by.setdefault(r["task"], [0, 0])
            by[r["task"]] = [k + int(bool(r["correct"])), n + 1]
        d["summary"] = {t: {"n": n, "acc": round(k / n, 4)}
                        for t, (k, n) in sorted(by.items())}

    res_path.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    left = sum(1 for r in recs if r.get("pred") is None)
    print(f"fixed {fixed}, remaining unparsed {left}", flush=True)
    if isinstance(d, dict) and "summary" in d:
        print(json.dumps(d["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
