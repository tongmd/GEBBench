#!/usr/bin/env python3
"""Retry unparsed (pred=None) records of a catlab result file in place.

Reuses run_catlab.run_item verbatim (same prompt, same image payload, same
parser); only the underlying HTTP call is swapped for one with a larger
max_tokens so long chain-of-thought replies can reach their final JSON.

  python3 catlab/retry_unparsed.py --tag or_gemini-3.1-pro \
      --model google/gemini-3.1-pro-preview
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_catlab as rc  # noqa: E402


def call_big(base, key, model, content, max_retries=6, max_tokens=16384):
    body = json.dumps({"model": model, "temperature": 0.0,
                       "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": content}]}).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{base}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            if txt and txt.strip():
                return txt
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
    ap.add_argument("--max-tokens", type=int, default=16384)
    args = ap.parse_args()

    rc.call = lambda base, key, model, content, max_retries=6: call_big(
        base, key, model, content, max_retries, args.max_tokens)

    data = json.loads((rc.ROOT / "items.json").read_text())
    menu = data["menu25"]
    items = {i["id"]: i for i in data["items"]}

    res_path = rc.ROOT / "results" / f"{args.tag}.json"
    d = json.loads(res_path.read_text())
    ns = SimpleNamespace(base=args.base, key=args.key, model=args.model)

    todo = [i for i, r in enumerate(d["records"]) if r.get("pred") is None]
    print(f"{args.tag}: {len(todo)} unparsed records to retry", flush=True)
    fixed = 0
    for i in todo:
        old = d["records"][i]
        new = rc.run_item(ns, menu, items[old["id"]])
        if new["pred"] is not None:
            d["records"][i] = new
            fixed += 1
        print(f"  {old['id']}: -> {new['pred']} "
              f"({'OK' if new['correct'] else 'wrong/none'})", flush=True)

    res_path.write_text(json.dumps(d, indent=1))
    left = sum(1 for r in d["records"] if r.get("pred") is None)
    print(f"fixed {fixed}, remaining unparsed {left}", flush=True)


if __name__ == "__main__":
    main()
