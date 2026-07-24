#!/usr/bin/env python3
"""Retry failed (unparseable) vision items with downscaled JPEG images, merge into result JSON.

Usage: retry_failed_vision.py <result_json> <items_json> --base ... --model ... [--max-px 768]
Items are matched by id; only records with pred==None are retried.
"""
import argparse, base64, io, json, re, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

ROOT = Path(__file__).parent.parent

def small_b64(path: str, max_px: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

def call(base, key, model, content, retries=3):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 4096,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if a == retries - 1:
                print("  ERR", e); return None
            time.sleep(3 * (a + 1))

def parse(reply, field):
    if not reply: return None
    m = re.search(r'"%s"\s*:\s*"([^"]+)"' % field, reply)
    if m: return m.group(1).strip()
    if field == "choice":
        ls = re.findall(r"\b([A-D])\b", reply)
        return ls[-1] if ls else None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json"); ap.add_argument("items_json")
    ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=None); ap.add_argument("--max-px", type=int, default=768)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import os
    key = args.key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    res = json.loads(Path(args.result_json).read_text())
    data = json.loads(Path(args.items_json).read_text())
    items = {i["id"]: i for i in (data["items"] if isinstance(data, dict) else data)}
    menu = data.get("menu", "") if isinstance(data, dict) else ""

    failed = [r for r in res["records"] if r.get("pred") is None]
    print(f"{len(failed)} failed records to retry")

    P_SCENE = ("这张图像的构图体现了一个抽象结构母题。从下面的选项中选出它体现的母题。\n\n"
               "选项：\n{menu}\n\n只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}")
    P_SKEL = ("这张示意图画的是一个抽象结构母题。从下面的选项中选出它。\n\n"
              "选项：\n{menu}\n\n只输出 JSON（不要代码块）：{{\"motif_id\": \"...\"}}")
    P_XV = ("这张图像的构图体现了某个抽象结构。下面四个小故事中，恰有一个的讲法"
            "实现了与图像相同的结构。选出它。\n\n{options}\n\n"
            "只输出 JSON（不要代码块）：{{\"choice\": \"A/B/C/D 之一\"}}")

    def redo(rec):
        it = items.get(rec["id"])
        if it is None: return rec
        task = it["task"]
        if task == "crossvoice":
            opts = "\n\n".join(f"【{k}】\n{v}" for k, v in it["options"].items())
            prompt, field = P_XV.format(options=opts), "choice"
        elif task == "skeleton2motif":
            prompt, field = P_SKEL.format(menu=menu), "motif_id"
        else:
            prompt, field = P_SCENE.format(menu=menu), "motif_id"
        img = it.get("image")
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url":
                    f"data:image/jpeg;base64,{small_b64(str(ROOT / img) if not Path(img).is_absolute() else img, args.max_px)}"}}]
        raw = call(args.base, key, args.model, content)
        pred = parse(raw, field)
        out = dict(rec)
        out["raw"] = raw or ""
        out["pred"] = pred
        out["correct"] = (pred == rec["answer"])
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        redone = list(ex.map(redo, failed))

    by_id = {r["id"]: r for r in redone}
    res["records"] = [by_id.get(r["id"], r) for r in res["records"]]
    by_task = {}
    for r in res["records"]:
        by_task.setdefault(r["task"], []).append(r)
    res["summary"] = {t: {"n": len(rs), "acc": round(sum(x["correct"] for x in rs) / len(rs), 4)}
                      for t, rs in by_task.items()}
    Path(args.result_json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    still = sum(1 for r in res["records"] if r.get("pred") is None)
    print(json.dumps(res["summary"], indent=1))
    print(f"still unparseable: {still}")

if __name__ == "__main__":
    main()
