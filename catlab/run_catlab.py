#!/usr/bin/env python3
"""Run the functorial-abstraction pilot against an OpenAI-compatible endpoint.

  python3 catlab/run_catlab.py --base https://openrouter.ai/api/v1 \
      --model openai/gpt-5.5 --key $OPENROUTER_API_KEY --tag or_gpt-5.5
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

CHOICE_TASKS = {"square_thm", "square_story", "id_vs_change", "composition"}

PROMPTS = {
    "endpoint_id": (
        "This diagram depicts an abstract structural motif. "
        "Pick the motif it depicts from the options below.\n\n"
        "Options:\n{menu}\n\n"
        "Output JSON only (no code block): {{\"motif_id\": \"...\"}}"),

    "square_thm": (
        "The image shows one structural change applied to a diagram: the "
        "BEFORE panel on the left, the AFTER panel on the right. Call this "
        "change g.\n\n"
        "The mathematical theorem below (some proper names masked with □) "
        "characterizes the structure of the BEFORE panel. Exactly one of the "
        "four theorems that follow characterizes the structure you would get "
        "by applying the SAME change g to it. Pick that one.\n\n"
        "Theorem for the BEFORE structure:\n{text}\n\n{options}\n\n"
        "Output JSON only (no code block): {{\"choice\": \"one of A-D\"}}"),

    "square_story": (
        "The image shows one structural change applied to a diagram: the "
        "BEFORE panel on the left, the AFTER panel on the right. Call this "
        "change g.\n\n"
        "The short story below is told in a way that realizes the same "
        "structure as the BEFORE panel. Exactly one of the four stories that "
        "follow is told in a way that realizes the structure you would get by "
        "applying the SAME change g to it. Pick that one. (Beware: one of the "
        "options realizes the unchanged BEFORE structure in a different "
        "setting.)\n\n"
        "Story for the BEFORE structure:\n{text}\n\n{options}\n\n"
        "Output JSON only (no code block): {{\"choice\": \"one of A-D\"}}"),

    "id_vs_change": (
        "The image shows two diagrams: BEFORE on the left, AFTER on the "
        "right. Exactly one statement below correctly describes the "
        "relationship between them. Pick it.\n\n{options}\n\n"
        "Output JSON only (no code block): {{\"choice\": \"one of A-D\"}}"),

    "composition": (
        "The image shows two structural changes, one per row: STEP 1 "
        "(its own BEFORE and AFTER) and STEP 2 (its own BEFORE and AFTER).\n\n"
        "The mathematical theorem below (some proper names masked with "
        "□) characterizes a starting structure. Apply STEP 1's change to "
        "it, then apply STEP 2's change to the result. Exactly one of the "
        "four theorems that follow characterizes the final structure. Pick "
        "it.\n\n"
        "Theorem for the starting structure:\n{text}\n\n{options}\n\n"
        "Output JSON only (no code block): {{\"choice\": \"one of A-D\"}}"),
}


def img_payload(path: Path, max_side: int = 1100, quality: int = 90) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# GEB_PROXY is the repo-wide name (run_comparison.py, run_bilingual_v3.py);
# GEB_SOCKS stays accepted so older launch scripts keep working.
_SOCKS = os.environ.get("GEB_PROXY") or os.environ.get("GEB_SOCKS", "")
_CLIENT = None
if _SOCKS:  # route egress through a proxy for region-restricted endpoints
    import httpx
    # Granular timeouts, not a single scalar: a SOCKS tunnel can drop a
    # connection's upstream while the local socket stays half-open, and a
    # lone read that never returns will hang the worker indefinitely (one
    # such stall wedged a run for ~2h with the CPU idle). A bounded read
    # timeout lets the retry loop below fire instead. 180s comfortably
    # exceeds the ~90s a slow reasoning-model vision call needs.
    _CLIENT = httpx.Client(
        proxy=_SOCKS,
        timeout=httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=30.0))


def call(base, key, model, content, max_retries=6):
    payload = {"model": model, "temperature": 0.0, "max_tokens": 4096,
               "messages": [{"role": "user", "content": content}]}
    body = json.dumps(payload).encode()
    hdrs = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            if _CLIENT is not None:
                r = _CLIENT.post(f"{base}/chat/completions",
                                 content=body, headers=hdrs)
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"]
            else:
                req = urllib.request.Request(
                    f"{base}/chat/completions", data=body, headers=hdrs)
                with urllib.request.urlopen(req, timeout=300) as resp:
                    txt = json.loads(resp.read())["choices"][0]["message"]["content"]
            if txt and txt.strip():
                return txt
        except Exception as e:
            if attempt == max_retries - 1:
                print("  ERR", type(e).__name__, str(e)[:100], flush=True)
                return None
        time.sleep(4 * (attempt + 1))
    return None


MOTIF_RE = re.compile(
    r"\b(self_reference|recursion|nesting|cycle|symmetry_breaking|symmetry|"
    r"alternation|spiral|convergence|bottleneck|strange_loop|figure_ground|"
    r"isomorphism|mobius|braiding|interference|aperiodic_tiling|tiling|hub|"
    r"diagonalization|infinite_descent|coprime_orbit|pigeonhole|"
    r"projective_duality|contraction_fixed_point)\b")


def parse(reply, field):
    if not reply:
        return None
    m = re.search(r'"%s"\s*:\s*"([^"]+)"' % field, reply)
    if m:
        return m.group(1).strip()
    if field == "choice":
        ls = re.findall(r"\b([A-D])\b", reply)
        return ls[-1] if ls else None
    ids = MOTIF_RE.findall(reply)
    return ids[0] if ids else None


def run_item(args, menu, item):
    task = item["task"]
    kw = {"menu": menu, "text": item.get("text", "")}
    if "options" in item:
        kw["options"] = "\n\n".join(f"[{k}]\n{v}"
                                    for k, v in item["options"].items())
    prompt = PROMPTS[task].format(**kw)
    content = [
        {"type": "image_url",
         "image_url": {"url": img_payload(REPO / item["image"])}},
        {"type": "text", "text": prompt},
    ]
    reply = call(args.base, args.key, args.model, content)
    field = "choice" if task in CHOICE_TASKS else "motif_id"
    pred = parse(reply, field)
    return {"id": item["id"], "task": task, "answer": item["answer"],
            "pred": pred, "correct": pred == item["answer"],
            "meta": item.get("meta", {}),
            "option_motifs": item.get("option_motifs"),
            "raw": (reply or "")[:200]}


def remap_cached(rec, item, legacy):
    """Re-letter a cached record onto the current item's option ordering.

    Returns the rescored record, or None if the cached choice cannot be
    identified under the current options (then the item is re-queried).
    Motif-name tasks need no remapping; letter tasks are remapped through the
    option text the model actually saw, so `correct` reflects what it chose.
    """
    if rec["task"] not in CHOICE_TASKS:            # answer is a motif name
        return {**rec, "answer": item["answer"],
                "correct": rec["pred"] == item["answer"]}

    # What the cached letter pointed at, under the key it was administered with.
    old_opts = rec.get("option_motifs") or legacy.get(rec["id"], {}).get("options")
    if not old_opts:
        return None
    chosen = old_opts.get(rec["pred"])
    if chosen is None:
        return None

    new_opts = item.get("option_motifs") or item.get("options") or {}
    letter = next((k for k, v in new_opts.items() if v == chosen), None)
    if letter is None:                 # that option is not offered any more
        return None
    return {**rec, "pred": letter, "answer": item["answer"],
            "correct": letter == item["answer"],
            "option_motifs": item.get("option_motifs"),
            "meta": item.get("meta", rec.get("meta", {}))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY", "EMPTY"))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data = json.loads((ROOT / "items.json").read_text())
    menu, items = data["menu25"], data["items"]
    if args.tasks:
        keep = set(args.tasks.split(","))
        items = [i for i in items if i["task"] in keep]
    if args.limit:
        items = items[:args.limit]

    # Reuse any answer already on disk. The item set grew (more seeds), and
    # build_items.py now seeds its RNG per item, so a cached record's option
    # LETTERS may not match the current item's. What the model chose is a
    # motif/statement, not a letter, so remap: recover the cached choice via the
    # key it was administered under, then re-letter it under the current item.
    # A cached record whose choice cannot be located in the current options is
    # dropped and re-queried rather than scored against a mismatched key.
    tag = args.tag or re.sub(r"[^\w.-]", "_", args.model)
    outf = ROOT / "results" / f"{tag}.json"
    legacy = {}
    legacy_f = ROOT / "results" / "_legacy_ivc_key.json"
    if legacy_f.exists():
        legacy = json.loads(legacy_f.read_text())["items"]
    cur = {i["id"]: i for i in items}
    done, dropped = {}, 0
    if outf.exists():
        for r in json.loads(outf.read_text())["records"]:
            if r.get("pred") is None or r["id"] not in cur:
                continue
            r2 = remap_cached(r, cur[r["id"]], legacy)
            if r2 is None:
                dropped += 1
            else:
                done[r["id"]] = r2
    todo = [i for i in items if i["id"] not in done]
    print(f"{args.model}: {len(items)} items, {len(done)} cached, "
          f"{len(todo)} to run"
          + (f", {dropped} cached dropped (option text no longer present)"
             if dropped else ""), flush=True)
    if not todo:
        print("CATLAB_DONE", tag, "(nothing to do)", flush=True)
        return

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fresh = list(ex.map(lambda it: run_item(args, menu, it), todo))
    by_id = {**done, **{r["id"]: r for r in fresh}}
    recs = [by_id[i["id"]] for i in items if i["id"] in by_id]

    # one retry pass for anything that came back empty/unparsed
    for i, r in enumerate(recs):
        if r["pred"] is None:
            recs[i] = run_item(args, menu,
                               next(x for x in items if x["id"] == r["id"]))

    by = {}
    for r in recs:
        by.setdefault(r["task"], []).append(r)
    summary = {t: {"n": len(rs),
                   "acc": round(sum(x["correct"] for x in rs) / len(rs), 4),
                   "unparsed": sum(1 for x in rs if x["pred"] is None)}
               for t, rs in by.items()}
    outf.parent.mkdir(exist_ok=True)
    outf.write_text(json.dumps(
        {"model": args.model, "base": args.base, "summary": summary,
         "records": recs}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    print("CATLAB_DONE", tag, flush=True)


if __name__ == "__main__":
    main()
