#!/usr/bin/env python3
"""Post-generation structural screen for expansion scenes.

The pilot's 45 scenes were few enough to eyeball; at 250 the check must be
automatic.  Two-probe screen per generated scene, both gpt-5.2:stable vision:

  verify   given the intended invariant: does the composition instantiate it?
           (catches generation that symmetrized / regularized the structure)
  family4  blind 4-way pick among the designed confusable family's invariants
           (catches scenes that carry *a* structure but not distinguishably
           the right one)

pass = verify ok AND family4 correct;  weak = verify ok only;  fail = rest.
Failed scenes are deleted (regenerate with gen_scenes_expand.py, which skips
existing files); the verdict file records everything for the paper's repro
appendix.

Usage:  python3 verify_scenes_v2.py [--delete-fails] [motif_id ...]
Output: data/scene_screen_v2.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
LIB = {m["id"]: m for m in
       json.loads((ROOT / "data" / "motif_library.json").read_text())}
sys.path.insert(0, str(ROOT / "eval"))
from build_items import FAMILIES  # noqa: E402

JUDGE_MODEL = "gpt-5.2:stable"
PROMPTS_FILE = ROOT / "data" / "scene_prompts_v2.json"
OUT = ROOT / "data" / "scene_screen_v2.json"

VERIFY_PROMPT = """这张图像是为一个抽象结构母题生成的**静态场景**。判断图像的\
**构图**是否实现了下面的结构不变量。

读法约定（重要）：不变量原文可能用叙事/时间语言表述（事件序列、时间中点、\
逐层返回）。静态图像用**空间**实现同一结构：序列→沿空间路径的排列，时间中点→\
构图中轴，回到起点→路径闭合，逐层生成→尺度层级。只要构图的空间结构与不变量\
的抽象结构同构，即算实现——镜像对称的画面实现回文式不变量、多层分叉实现递归、\
环形闭合实现环。

要抓的失败方式（生成模型常见）：把不对称抹平、把恰一处异常补齐、丢失拓扑特征\
（半扭转、回指、唯一割点被画成多条并行通路）、把结构画成表面纹理而非构图骨架、\
两状态交替出现连续同态。

判定：
- passed=false 仅当构图**没有**承载该结构、或承载的是**另一个**结构、或关键\
拓扑/计数特征被明确破坏（如交替中连续两个同色、"必经点"有旁路）。
- 结构在但程度弱（层数少、尺度链短）→ passed=true 并在 problems 里注明。

不变量：{invariant}
{motif_note}
只输出 JSON（不要代码块）：
{{"passed": true或false, "problems": "若未实现，指出构图缺了什么（否则空字符串）"}}"""

# per-motif reading notes for invariants whose narrative wording maps poorly
# onto text-free images
MOTIF_NOTES = {
    "self_reference": (
        "\n本母题的图像读法：场景禁止出现文字，自指由**视觉自含**实现——画面内"
        "某处（画框/屏幕/织样/招牌）呈现的内容恰是**这幅场景本身**（同一房间、"
        "同一构图，可辨认），只此一层即可。判据：内层内容是否可辨认为整幅场景的"
        "摹本。若内层只是无关图案或空白套框（内容与整体无关），才判 nesting/"
        "false；不要因为\"没有文字语句\"而判 false。\n"
    ),
    "symmetry": (
        "\n本母题的图像读法：回文式序列在静态图像中即镜像对称——沿中轴翻转后"
        "逐点重合。判据看对称本身，不要求\"事件序列\"。\n"
    ),
    "tiling": (
        "\n本母题的图像读法：实景照片必然有边界、透视和光影差异——判据是"
        "**构图骨架**（单元格网格）是否为同形单元的周期重复，被摄单元形状一致、"
        "排列规则即算实现；不要求像素级平移重合，不要因画面有边框/透视/明暗"
        "变化判 false。若存在**恰一个**显著异常单元格则判 symmetry_breaking。\n"
    ),
    "mobius": (
        "\n本母题的图像读法：判据是带面是否有**可见的半扭转**（带的正反面在"
        "扭转处交换、或路径以8字/扭结形式接回）。真失败=画成了普通平环（无扭转"
        "痕迹），那才判 false / cycle。\n"
    ),
    "braiding": (
        "\n本母题的图像读法：判据是**三股以上**线条按交替上/下穿越交叉。真失败"
        "=只有两股（那是缠绕不是辫）或交叉无上下穿越关系。不要求端点排列可读出"
        "置换积。\n"
    ),
    "interference": (
        "\n本母题的图像读法：判据是**两个独立波源/周期系统**的叠加产生可见的"
        "干涉图样（相长/相消条纹、莫尔纹）。两组同心圆交叠且交叠区有明暗调制即"
        "算实现。\n"
    ),
    "strange_loop": (
        "\n本母题的图像读法：\"谁叙述谁\"在图像中即**层级链的视觉闭环**——一串"
        "看似单向上升/包含的层级（楼梯、画框、舞台、屏幕），最高层却以某个构图"
        "元素接回最低层（埃舍尔式）。判据：是否存在可沿构图追踪的\"上行后回到"
        "起点\"路径。不要求文字或叙述者。\n"
    ),
    "isomorphism": (
        "\n本母题的图像读法：两套**异质素材**（如港口与车辆段）各占一半画面，"
        "两边的**连接结构**（节点数、连线关系、布局图样）逐项对应可互译。判据："
        "遮住内容只看骨架时两半是否同构。不要求文字字典或显式配对标记；元素"
        "样式不同正是设计意图。\n"
    ),
    "figure_ground": (
        "\n本母题的图像读法：图与底互补——正形与负空间共享同一轮廓且两者各自"
        "成形（埃舍尔式镶嵌，或上下半图底反转）。判据：负空间是否自成可辨识的"
        "形。不要求\"事件顺序\"。\n"
    ),
}

FAMILY4_PROMPT = """这张图像的构图体现了一个抽象结构母题。下面四个母题形式上\
相近，其中恰有一个是图像体现的。选出它。\n\n选项：\n{menu4}\n\n\
只输出 JSON（不要代码块）：{{"motif_id": "..."}}"""


def family_pool(mid: str) -> list[str]:
    pool = []
    for fam in FAMILIES:
        if mid in fam:
            pool += [x for x in fam if x != mid]
    out = list(dict.fromkeys(pool))
    for x in list(out):
        for y in (f for fam in FAMILIES if x in fam for f in fam):
            if y != mid and y not in out:
                out.append(y)
    rest = [y for y in LIB if y != mid and y not in out]
    return out + rest


def img_url(path: Path, max_side: int = 896) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def vision_call(path: Path, prompt: str) -> dict | None:
    import os
    body = json.dumps({
        "model": JUDGE_MODEL, "temperature": 0.0, "max_tokens": 1024,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_url(path)}},
            {"type": "text", "text": prompt}]}]})
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/chat/completions",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json", "-d", body]
    for _ in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            text = json.loads(raw)["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        try:
            import re
            return json.loads(
                re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M))
        except json.JSONDecodeError:
            continue
    return None


def screen_one(job) -> tuple[str, dict]:
    mid, png = job
    rng = random.Random(hash(png.name) & 0xFFFF)
    ver = vision_call(png, VERIFY_PROMPT.format(
        invariant=LIB[mid]["invariant_zh"],
        motif_note=MOTIF_NOTES.get(mid, "")))
    fam = family_pool(mid)[:3]
    opts = [mid] + fam
    rng.shuffle(opts)
    f4 = vision_call(png, FAMILY4_PROMPT.format(
        menu4="\n".join(f"- {o}: {LIB[o]['invariant_zh']}" for o in opts)))
    v_ok = bool(ver and ver.get("passed"))
    f_ok = bool(f4 and f4.get("motif_id") == mid)
    verdict = "pass" if (v_ok and f_ok) else ("weak" if v_ok else "fail")
    print(f"  {verdict:4s} {mid}/{png.name}"
          + (f"  ({(ver or {}).get('problems', 'judge unavailable')[:80]})"
             if verdict == "fail" else "")
          + ("" if f_ok else f"  [family4 -> {(f4 or {}).get('motif_id')}]"))
    return png.name, {"motif": mid, "verify": ver, "family4": f4,
                      "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-fails", action="store_true")
    ap.add_argument("ids", nargs="*")
    args = ap.parse_args()

    prompts = json.loads(PROMPTS_FILE.read_text())
    ids = args.ids or list(prompts)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}

    jobs = []
    for mid in ids:
        for cand in prompts.get(mid, []):
            png = ROOT / "motifs" / mid / f"scene_{cand['name']}.png"
            if png.exists() and png.name not in prev:
                jobs.append((mid, png))
    print(f"{len(jobs)} new scenes to screen (judge {JUDGE_MODEL})")

    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, rec in ex.map(screen_one, jobs):
            prev[name] = rec
    OUT.write_text(json.dumps(prev, indent=2, ensure_ascii=False))

    from collections import Counter
    counts = Counter(r["verdict"] for r in prev.values())
    print(f"\nverdicts: {dict(counts)} -> {OUT}")
    if args.delete_fails:
        for name, rec in prev.items():
            if rec["verdict"] == "fail":
                p = ROOT / "motifs" / rec["motif"] / name
                if p.exists():
                    p.unlink()
                    print(f"  deleted {rec['motif']}/{name} (regen next pass)")
    print("SCENE_SCREEN_DONE")


if __name__ == "__main__":
    main()
