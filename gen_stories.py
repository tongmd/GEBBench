#!/usr/bin/env python3
"""GEB benchmark — story voice v2: GEB-style FORM DEVICES.

v1 asked for stories vaguely "shaped like" a motif.  v2 is rigorous in the
manner of GEB's dialogues (Crab Canon = palindrome, Little Harmonic
Labyrinth = pushes never popped, Sloth Canon = one voice transformed):
each motif has a verified 形式装置 (story_form_device) in
data/motif_library.json — a formal constraint on the TELLING that a blind
reader can verify mechanically.  The story must satisfy its device.

Screening, two independent judges per story:
  1. device judge — given the device spec, mechanically checks every clause
  2. blind judge  — reads ONLY the story, picks the motif from the 18-item
     menu (no device spec, no hints)
Accept iff device passes AND the blind judge names the right motif.
Fail -> rewrite with the judges' reasons as feedback (up to 4 attempts).

Guardrail kept from v1: the writing prompt contains NO example story.

Usage:  python3 gen_stories.py [motif_id ...]
Output: data/stories_v2.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WRITER_MODEL = "gpt-5.5"
JUDGE_MODEL = "gpt-5.2"
ROOT = Path(__file__).parent
LIBRARY = {m["id"]: m for m in
           json.loads((ROOT / "data" / "motif_library.json").read_text())}
OUT = ROOT / "data" / "stories_v2.json"

# extra writer hints for devices that proved hard in the first pass
HINTS = {
    "self_reference": (
        "\n提示：先定标题——标题必须逐字是「本篇每句首字连读恰为此标题」这样一句"
        "关于本篇形式的完整真断言（可自拟等价说法，但必须是断言本篇藏头事实的话）；"
        "然后写正文，正文第 1 句的首字 = 标题第 1 字，第 2 句首字 = 标题第 2 字……"
        "句数恰等于标题字数。写完自查一遍首字连读。"),
    "alternation": (
        "\n提示：「恰为 2k 句」意思是对白总句数为偶数（如 8、10、12 句），"
        "不是两千句。"),
    "hub": (
        "\n提示：为免被误读成瓶颈或环——五名村民必须两两之间零往来（连见面都不写），"
        "凡事只与磨坊主一对一说；磨坊主与每人都有来回，消息经他双向中转、"
        "转完各自继续过日子，不是大家路过同一地点，也不是众人汇到一处为止。"),
    "interference": (
        "\n提示：为免被误读成汇聚——两条规则（每逢 3 的倍数句、每逢 4 的倍数句）"
        "各自贯穿全文、各自守拍，双命中出现在第 12 句与第 24 句两处；"
        "故事在第 12 句处绝不能收束，两种周期在情节里应是两件各自反复的事"
        "（如打更与涨潮），叠在一起才偶尔同时发生。"),
}
AVOID = {
    "self_reference": "染坊、纸灯、山/庙/和尚",
    "recursion": "糖画、河流支流",
    "nesting": "酱缸、城墙",
    "cycle": "染布、鸡和蛋",
    "symmetry": "晒盐",
    "alternation": "葫芦瓢取水",
    "spiral": "风筝",
    "convergence": "纸鸢",
    "bottleneck": "山口茶棚",
    "symmetry_breaking": "腌菜",
}

SYSTEM = (
    "你是一位民间故事作者，也是《哥德尔、埃舍尔、巴赫》对话录的研究者。"
    "GEB 的对话录之所以不朽，在于形式即内容：螃蟹卡农的台词沿中心逐句倒放，"
    "树懒卡农的一个声部是另一个的变换。你写的故事同样：口吻质朴自然、"
    "像口耳相传的民间故事，但讲法本身严格实现一个形式装置——"
    "读者顺着读是个好故事，回头看会突然发现故事的形状就是那个结构。"
)

PROMPT = """请为下面这个抽象结构写一个中文小故事。

结构：
- 名称：{name_zh}
- 方程：{equation}
- 不变量：{invariant}

★ 形式装置（这是硬性规格，故事的讲法必须逐条实现它，审稿人会逐条机械核查）：
{device}

要求：
1. 装置的每一条可核查条款都必须严格满足——句数、逐字重合、标签、字数，一条不差。
2. 在满足装置的前提下，故事要自然和谐、有人有物有地方（有名字或质感），
   可以朗读给孩子听；结构长在讲法里，不许被人物或叙述者点破。
3. 严禁出现数学、结构类词语（结构、循环、递归、对称、镜像、层级、无限……）。
4. 生活域要新鲜{avoid}。
5. 除装置明确要求外，故事不得回到或复读自己的开头。
{feedback}
只输出 JSON（不要代码块标记）：
{{"title": "标题（若装置对标题有规定，以装置为准）",
  "domain": "一两个词的生活域",
  "story_zh": "正文（若装置要求分句/分段/标签，用换行呈现）",
  "story_en": "英文重述（自然英语，传达故事与形状即可，不必复刻装置）",
  "why_shape": "一句话说明装置如何实现（审核用，可用术语）"}}"""

DEVICE_JUDGE_PROMPT = """你是苛刻的形式审稿人。下面是一个「形式装置」规格和一篇故事。
逐条机械核查故事是否满足装置的每一个可核查条款（数句数、比对逐字重合、查标签、
数字数……照规格执行）。

装置规格：
{device}

故事：
{story}

只输出 JSON（不要代码块标记）：
{{"passed": true或false, "checks": "逐条核查结果的简短清单", "problems": "不满足之处（若全满足则空字符串）"}}"""

BLIND_JUDGE_PROMPT = """读下面这个小故事，判断它的「形」——讲法与情节骨架——最像哪一个结构。
注意区分易混项：环是同层因果闭合一圈即复原；怪圈是层级上行却回到起点；
嵌套只进不出；递归有进有出、部分与整体同形；回文是下半逆序重演上半；
图底互换是同序重讲、主宾对调；同构是两套词汇经字典逐句互译；
汇聚是多线一次性汇入终点即止；瓶颈是必经点在中途、过点后各线仍分头续行、
两岸各自成块；星型枢纽是诸端点彼此从不直接往来、事事经同一中心双向中转，
端点间无边；干涉是两种不同周期的规则各自守拍、叠加在公倍位置同相浮现更大
周期——不是线索汇聚。
若正文分句编号或句式整齐，先把各句首字连读一遍——若首字连读恰拼出标题、
且标题是关于本篇形式的断言，那是自指（文本内含一句对自身为真的描述），
不是图底互换也不是汇聚。

故事：
{story}

选项：
{menu}

只输出 JSON（不要代码块标记）：
{{"motif_id": "选项里的一个id"}}"""


def chat(model: str, system: str, user: str) -> str | None:
    body = json.dumps({"model": model, "messages": (
        [{"role": "system", "content": system}] if system else []) +
        [{"role": "user", "content": user}]})
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/chat/completions",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json", "-d", body]
    raw = subprocess.run(cmd, capture_output=True, text=True).stdout
    try:
        return json.loads(raw)["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def parse_json(text: str) -> dict | None:
    try:
        return json.loads(re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M))
    except (json.JSONDecodeError, AttributeError):
        return None


MENU = "\n".join(f"- {m['id']}: {m['invariant_zh']}" for m in LIBRARY.values())


def judge_device(mid: str, story: str) -> dict | None:
    reply = chat(JUDGE_MODEL, "", DEVICE_JUDGE_PROMPT.format(
        device=LIBRARY[mid]["story_form_device"], story=story))
    return parse_json(reply) if reply else None


def judge_blind(story: str) -> dict | None:
    reply = chat(JUDGE_MODEL, "", BLIND_JUDGE_PROMPT.format(story=story, menu=MENU))
    return parse_json(reply) if reply else None


def gen_one(mid: str) -> dict | None:
    lib = LIBRARY[mid]
    avoid = f"（避免：{AVOID[mid]}，已用过）" if mid in AVOID else ""
    feedback = ""
    item = None
    for attempt in range(4):
        reply = chat(WRITER_MODEL, SYSTEM, PROMPT.format(
            name_zh=lib["name_zh"], equation=lib["equation_katex"],
            invariant=lib["invariant_zh"],
            device=lib["story_form_device"] + HINTS.get(mid, ""),
            avoid=avoid, feedback=feedback))
        item = parse_json(reply) if reply else None
        if not item or "story_zh" not in item:
            print(f"  retry {mid} (attempt {attempt + 1}: bad output)")
            continue
        item.update(motif_id=mid, equation=lib["equation_katex"],
                    model=WRITER_MODEL)
        dev = judge_device(mid, item["story_zh"])
        blind = judge_blind(item["story_zh"])
        if dev is None or blind is None:
            print(f"  {mid}: judge unavailable, accepting unscreened")
            item["judge"] = None
            return item
        ok_dev = bool(dev.get("passed"))
        ok_blind = blind.get("motif_id") == mid
        item["judge"] = {"device_passed": ok_dev,
                         "device_checks": dev.get("checks", ""),
                         "blind_motif": blind.get("motif_id"),
                         "model": JUDGE_MODEL,
                         "passed": ok_dev and ok_blind}
        if ok_dev and ok_blind:
            print(f"  ok {mid}: {item.get('title', '?')} [{item.get('domain', '?')}]"
                  + (f" (attempt {attempt + 1})" if attempt else ""))
            return item
        reasons = []
        if not ok_dev:
            reasons.append(f"装置核查未过：{dev.get('problems', '')}")
        if not ok_blind:
            reasons.append(f"盲评者把形认成了「{blind.get('motif_id')}」")
        print(f"  rewrite {mid} (attempt {attempt + 1}): "
              f"{'; '.join(reasons)[:150]}")
        feedback = ("\n上一版的问题：" + "；".join(reasons) +
                    "\n请修正后重写，装置条款逐条对照执行。\n")
    print(f"  WARN {mid}: kept last version despite failed screening")
    return item


def main():
    ids = sys.argv[1:] or list(LIBRARY)
    OUT.parent.mkdir(exist_ok=True)
    existing = {}
    if OUT.exists():
        existing = {s["motif_id"]: s for s in json.loads(OUT.read_text())}
    todo = [mid for mid in ids
            if not (existing.get(mid, {}).get("judge") or {}).get("passed")]
    print(f"{len(todo)} motifs to write: {', '.join(todo)}")
    with ThreadPoolExecutor(max_workers=6) as ex:
        for item in ex.map(gen_one, todo):
            if item:
                existing[item["motif_id"]] = item
    items = [existing[mid] for mid in LIBRARY if mid in existing]
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    passed = sum(1 for i in items if i.get("judge") and i["judge"].get("passed"))
    print(f"\n{len(items)} stories ({passed} passed screening) -> {OUT}")


if __name__ == "__main__":
    main()
