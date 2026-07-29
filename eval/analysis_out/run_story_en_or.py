#!/usr/bin/env python3
"""story2motif-EN bilingual transport replication — OpenRouter provenance.

Same items/prompt/decoding as run_story_en.py (25 English story items,
25-way menu, greedy), but run against OpenRouter so no paper number traces
to the yunwu proxy.  Outputs eval/analysis_out/story_en_or_<model>.json
"""
from __future__ import annotations
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "analysis_out"))
from menu_en import INVARIANT_EN  # noqa: E402

LIB = json.loads((ROOT / "data" / "motif_library.json").read_text())
IDS = [m["id"] for m in LIB]
STORIES = {s["motif_id"]: s for s in
           json.loads((ROOT / "data" / "stories_v2.json").read_text())}

MENU_EN = "\n".join(f"- {mid}: {INVARIANT_EN[mid]}" for mid in IDS)

PROMPT_EN = (
    "The narration and plot skeleton of the following short story realize an "
    "abstract structural motif. Choose it.\n\n"
    "Story:\n{text}\n\nOptions:\n{menu}\n\n"
    "Output only JSON (no code block): {{\"motif_id\": \"...\"}}")

BASE = "https://openrouter.ai/api/v1"
KEY = os.environ["OPENROUTER_API_KEY"]

MODELS = [
    ("openai/gpt-5.5", "or_gpt-5.5"),
    ("openai/gpt-5.2", "or_gpt-5.2"),
    ("anthropic/claude-opus-4.8", "or_claude-opus-4.8"),
]


def call(model: str, content: str, max_retries: int = 4) -> str | None:
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 4096,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                print("  ERR", type(e).__name__, str(e)[:120])
                return None
            time.sleep(3 * (attempt + 1))
    return None


def parse(reply: str | None) -> str | None:
    if not reply:
        return None
    m = re.search(r'"motif_id"\s*:\s*"([^"]+)"', reply)
    if m:
        return m.group(1).strip()
    for mid in IDS:
        if re.search(r'\b%s\b' % re.escape(mid), reply):
            return mid
    return None


def items_en():
    out = []
    for mid in IDS:
        if mid not in STORIES:
            continue
        out.append({"task": "story2motif", "id": f"story_en:{mid}",
                    "text": STORIES[mid]["story_en"], "answer": mid})
    return out


def run_one(model, item):
    prompt = PROMPT_EN.format(text=item["text"], menu=MENU_EN)
    reply = call(model, prompt)
    pred = parse(reply)
    if pred is None:
        reply2 = call(model, prompt)
        pred2 = parse(reply2)
        if pred2 is not None:
            reply, pred = reply2, pred2
    return {"id": item["id"], "task": "story2motif", "answer": item["answer"],
            "pred": pred, "correct": pred == item["answer"],
            "raw": (reply or "")[:200]}


def main():
    its = items_en()
    print(f"base={BASE}  n_items={len(its)}")
    for model, tag in MODELS:
        outf = ROOT / "eval" / "analysis_out" / f"story_en_{tag}.json"
        if outf.exists():
            print(f"=== {tag}: exists, skip ===")
            continue
        print(f"=== {model} ({tag}) ===")
        with ThreadPoolExecutor(max_workers=6) as ex:
            recs = list(ex.map(lambda it: run_one(model, it), its))
        acc = round(sum(r["correct"] for r in recs) / len(recs), 4)
        n_parsefail = sum(1 for r in recs if r["pred"] is None)
        out = {"model": model, "base": BASE, "lang": "en",
               "summary": {"story2motif": {"n": len(recs), "acc": acc}},
               "parse_failures": n_parsefail, "records": recs}
        outf.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"  acc={acc}  parse_fail={n_parsefail}  -> {outf}")


if __name__ == "__main__":
    main()
