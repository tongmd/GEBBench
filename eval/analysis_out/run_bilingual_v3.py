#!/usr/bin/env python3
"""Bilingual structural-transport arm, expanded corpus (v3).

The shipped bilingual pilot (appendix "Structure survives translation") used
the 25 story-per-motif v2 corpus: one story per motif, so 25 paired items per
model and only three models.  data/stories_v3.json carries 90 stories that all
have both a Chinese and an English telling, covering all 25 motifs (2-4 each),
so the same paired design runs at 90 items -- 3.6x the pairs -- without
generating anything new.

Protocol is identical to the shipped arm so the two are comparable:
  * 25-way menu, ZH menu for the ZH telling (eval/items.json "menu"),
    EN menu for the EN telling (menu_en.INVARIANT_EN);
  * the same story2motif prompt wording as run_eval.py (ZH) and
    run_story_en_or.py (EN);
  * greedy decoding (temperature 0), single JSON field, one retry on an
    unparsable reply, parse failure counted wrong.

Outputs eval/analysis_out/bilingual_v3_<lang>_<tag>.json, one file per
model per language, each holding per-item records keyed by story_id so the
ZH and EN runs pair item for item.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "analysis_out"
sys.path.insert(0, str(OUT))
from menu_en import INVARIANT_EN  # noqa: E402

LIB = json.loads((ROOT / "data" / "motif_library.json").read_text())
IDS = [m["id"] for m in LIB]

# ZH menu: exactly the string the shipped ZH run administered.
MENU_ZH = json.loads((ROOT / "eval" / "items.json").read_text())["menu"]
MENU_EN = "\n".join(f"- {mid}: {INVARIANT_EN[mid]}" for mid in IDS)

PROMPT_ZH = (
    "下面这个小故事的讲法与情节骨架实现了一个抽象结构母题。选出它。\n\n"
    "故事：\n{text}\n\n选项：\n{menu}\n\n"
    "只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}")

PROMPT_EN = (
    "The narration and plot skeleton of the following short story realize an "
    "abstract structural motif. Choose it.\n\n"
    "Story:\n{text}\n\nOptions:\n{menu}\n\n"
    "Output only JSON (no code block): {{\"motif_id\": \"...\"}}")

BASE = "https://openrouter.ai/api/v1"
KEY = os.environ["OPENROUTER_API_KEY"]

# Optional SOCKS/HTTP proxy so calls can egress through a region where the
# frontier models are not geoblocked (set GEB_PROXY=socks5://127.0.0.1:1080).
PROXY = os.environ.get("GEB_PROXY") or None
_client = httpx.Client(proxy=PROXY, timeout=300.0)

# The three models of the shipped arm, plus Gemini/Kimi/Qwen-72B so the
# bilingual claim covers the same six models as the catlab pilot.
MODELS = [
    ("openai/gpt-5.5", "or_gpt-5.5"),
    ("openai/gpt-5.2", "or_gpt-5.2"),
    ("anthropic/claude-opus-4.8", "or_claude-opus-4.8"),
    ("google/gemini-3.1-pro-preview", "or_gemini-3.1-pro"),
    ("moonshotai/kimi-k2.5", "or_kimi-k2.5"),
    ("qwen/qwen2.5-vl-72b-instruct", "or_qwen2.5-vl-72b"),
]


def call(model: str, content: str, max_retries: int = 5) -> str | None:
    payload = {"model": model, "temperature": 0.0, "max_tokens": 4096,
               "messages": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {KEY}",
               "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            r = _client.post(f"{BASE}/chat/completions",
                             json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 - transient API failures
            if attempt == max_retries - 1:
                print("  ERR", type(e).__name__, str(e)[:120], flush=True)
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
        if re.search(r"\b%s\b" % re.escape(mid), reply):
            return mid
    return None


def items(lang: str) -> list[dict]:
    """The 90 v3 stories that carry both tellings, in a fixed order."""
    corpus = json.loads((ROOT / "data" / "stories_v3.json").read_text())
    field = "story_zh" if lang == "zh" else "story_en"
    out = []
    for s in corpus:
        if not (s.get("story_zh", "").strip() and s.get("story_en", "").strip()):
            continue
        out.append({"id": s["story_id"], "motif_id": s["motif_id"],
                    "text": s[field]})
    return sorted(out, key=lambda r: r["id"])


def run_one(model: str, lang: str, item: dict) -> dict:
    tmpl = PROMPT_ZH if lang == "zh" else PROMPT_EN
    menu = MENU_ZH if lang == "zh" else MENU_EN
    prompt = tmpl.format(text=item["text"], menu=menu)
    reply = call(model, prompt)
    pred = parse(reply)
    if pred is None:  # one retry on an unparsable reply, as in the shipped arm
        reply2 = call(model, prompt)
        pred2 = parse(reply2)
        if pred2 is not None:
            reply, pred = reply2, pred2
    return {"id": item["id"], "task": "story2motif", "lang": lang,
            "answer": item["motif_id"], "pred": pred,
            "correct": pred == item["motif_id"], "raw": (reply or "")[:200]}


def main() -> None:
    # Optional allowlist so a partial route (e.g. only the models reachable
    # from the current egress) can be run and later resumed: the per-model
    # output files that already exist are skipped on the next run.
    only = os.environ.get("GEB_ONLY_TAGS", "").strip()
    allow = set(only.split(",")) if only else None
    models = [(m, t) for m, t in MODELS if allow is None or t in allow]
    for lang in ("zh", "en"):
        its = items(lang)
        print(f"lang={lang}  n_items={len(its)}", flush=True)
        for model, tag in models:
            outf = OUT / f"bilingual_v3_{lang}_{tag}.json"
            if outf.exists():
                print(f"=== {lang}/{tag}: exists, skip ===", flush=True)
                continue
            print(f"=== {model} ({tag}, {lang}) ===", flush=True)
            with ThreadPoolExecutor(max_workers=6) as ex:
                recs = list(ex.map(lambda it: run_one(model, lang, it), its))
            acc = round(sum(r["correct"] for r in recs) / len(recs), 4)
            nfail = sum(1 for r in recs if r["pred"] is None)
            outf.write_text(json.dumps(
                {"model": model, "base": BASE, "lang": lang,
                 "corpus": "data/stories_v3.json",
                 "summary": {"story2motif": {"n": len(recs), "acc": acc}},
                 "parse_failures": nfail, "records": recs},
                indent=1, ensure_ascii=False))
            print(f"  acc={acc}  parse_fail={nfail}  -> {outf.name}", flush=True)


if __name__ == "__main__":
    main()
