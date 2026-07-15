#!/usr/bin/env python3
"""GEB benchmark — story-voice generation via yunwu API (gpt-5.5).

One story per motif, 自然和谐: the story's shape IS the motif, told in a plain
folk-tale voice, no structural vocabulary, no exact counting.

Guardrails learned from the first pilot:
- the writing prompt contains NO example story (an example's narrative devices
  leak into every motif — the first pilot ended 5/10 stories with a verbatim
  reprise of their own opening, contaminating labels with self-reference)
- unless the motif IS self_reference, the story must not contain / reprise
  itself in any form
- recursion vs nesting are given sharply distinct shape instructions
- back-translation screening: an uninformed judge model reads only the story
  and picks the motif; mismatch or illegal self-loop -> rewrite with feedback

Usage:  python3 gen_stories.py
Output: data/stories_pilot.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WRITER_MODEL = "gpt-5.5"
JUDGE_MODEL = "gpt-5.2"
OUT = Path(__file__).parent / "data" / "stories_pilot.json"

# (motif_id, equation, shape instruction for the writer, avoid-domains)
MOTIF_SPECS = [
    ("self_reference", "s = T(s)（不动点）",
     "故事必须原样包含它自己：故事里出现的某样东西（一幅画、一段话、一出戏……）"
     "的内容，正是这个故事本身，于是没有底。只有这个母题允许、并且必须这样写。",
     "山/庙/和尚、馄饨摊（已用）"),
    ("recursion", "T = Node(T, …, T)",
     "整体由几个与它自己形状相同的部分组成：凑近看任何一个部分，它自己又是同样"
     "几个更小部分组成的。重点是「部分构成整体、部分与整体同形」，像分岔、组成、"
     "拼合——不要写成一层套一层的容器，也不要让故事回到开头。",
     "河流与支流、绣被（已用）"),
    ("nesting", "S_λ(X) = X（缩放不变）",
     "一层套一层的容器：打开（或走进）一层，里面是缩小的同一层。重点是「包含与"
     "进入」——不要写成分岔组成，也不要让故事回到开头。",
     "环形城墙、针线笸箩（已用）"),
    ("cycle", "v ⇝ v（回到起点）",
     "环要长在因果里：甲来自乙，乙来自丙，丙又来自甲，追问下去没有第一个。"
     "禁止用「故事复读自己的开头」来表现环——环是事物之间的因果闭合，"
     "不是叙述文字的回环。",
     "鸡和蛋、老面馒头（已用）"),
    ("symmetry", "σ(x) = x（镜像不变）",
     "两半互为镜像：一边发生什么，另一边同时以相反的方向发生同样的事；"
     "两边合起来恰好完整。故事本身不要回到开头。",
     "织布（已用）"),
    ("alternation", "f∘f = id（交替）",
     "两种状态你来我往：甲引出乙，乙又引出甲，交替本身就是生活，谁也不是终点。"
     "用具体事物的交替表现，结尾不要复读开头的句子。",
     "染坊翻牌（已用）"),
    ("spiral", "r = a·e^{bθ}（旋转×生长）",
     "一边绕着同一个中心转，一边一圈比一圈更远（或更近）：既在重复又在前进，"
     "转回原方向时已不在原处。故事不要回到开头。",
     "陶窑（已用）"),
    ("convergence", "many → one（汇聚）",
     "许多各自出发、互不相识的东西，一路合并，最后归于同一处、成为一件事。",
     "百家粥（已用）"),
    ("bottleneck", "所有路径必经一点",
     "无论从哪里来、要到哪里去，都必须经过同一个地方；那个地方因此看尽了所有人。"
     "不要让故事复读自己的开头。",
     "酱园门前（已用）"),
    ("symmetry_breaking", "规则场中的一个异常",
     "处处相同、事事如常之中，恰恰有一处不同；正是这一处让整体有了方向或有了故事。",
     "染坊晾布（已用）"),
]

JUDGE_MENU = [
    ("self_reference", "整体原样出现在自身内部（故事包含这个故事本身）"),
    ("recursion", "整体由与自己同形的部分组成，部分再由更小的同形部分组成"),
    ("nesting", "一层套一层，每层里面是缩小的同一层"),
    ("cycle", "沿因果追问会回到起点，没有第一个"),
    ("symmetry", "两半互为镜像，反向对应"),
    ("alternation", "两种状态交替往复"),
    ("spiral", "绕着中心转，同时一圈比一圈远或近"),
    ("convergence", "许多来处汇入同一处"),
    ("bottleneck", "所有路径必经同一点"),
    ("symmetry_breaking", "处处相同之中恰有一处不同"),
]

SYSTEM = (
    "你是一位民间故事作者。你写的故事像口耳相传了几百年：句子短，词朴素，"
    "有呼吸感，读起来自然和谐，可以讲给孩子听。"
)

PROMPT = """请为下面这个抽象结构写一个中文小故事（80~160 字）。

结构（只给你看，绝不能出现在故事里）：
- 方程：{equation}
- 这个结构在故事里应有的形：{shape}

要求：
1. 结构要长在情节和讲法里，不是被人物说出来、也不是被叙述者点破。
2. 口吻自然和谐，民间故事的质朴韵律，可朗读。
3. 严禁出现任何数学、结构类词语（结构、循环、递归、对称、无限、规律、重复……）。
4. 不要精确计数（不要「恰好四层」「三条支流」这类数字）。
5. 选一个新鲜的生活域{avoid}。人、物、地方要有名字或质感，不要抽象代号。
6. 除非上面的「形」明确要求，故事不得以任何方式回到或复读自己的开头，
   结尾不得与第一句相同，故事里不得出现「有人讲起这个故事」。
7. 再写一个英文版，同样自然，不是直译腔。
{feedback}
只输出 JSON（不要代码块标记）：
{{"title": "四五个字的标题", "domain": "一两个词的域", "story_zh": "...", "story_en": "...", "why_shape": "一句话：故事的形怎么体现这个结构（审核用，可用术语）"}}"""

JUDGE_PROMPT = """读下面这个小故事，判断它的「形」——故事的讲法和情节骨架——最像哪一个结构。

故事：
{story}

选项：
{menu}

另外判断：故事是否在结尾回到／复读了自己的开头，或让「这个故事本身」出现在故事内部？

只输出 JSON（不要代码块标记）：
{{"motif_id": "选项里的一个id", "loops_back": true或false}}"""


def chat(model: str, system: str, user: str) -> str | None:
    body = json.dumps({"model": model, "messages": (
        [{"role": "system", "content": system}] if system else []) +
        [{"role": "user", "content": user}]})
    cmd = ["curl", "-s", "--max-time", "240", "-X", "POST",
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


def judge(story_zh: str) -> dict | None:
    menu = "\n".join(f"- {mid}: {desc}" for mid, desc in JUDGE_MENU)
    reply = chat(JUDGE_MODEL, "", JUDGE_PROMPT.format(story=story_zh, menu=menu))
    return parse_json(reply) if reply else None


def gen_one(spec) -> dict | None:
    mid, eq, shape, avoid = spec
    avoid_txt = f"（避免：{avoid}）" if avoid else ""
    feedback = ""
    item = None
    for attempt in range(3):
        reply = chat(WRITER_MODEL, SYSTEM, PROMPT.format(
            equation=eq, shape=shape, avoid=avoid_txt, feedback=feedback))
        item = parse_json(reply) if reply else None
        if not item or "story_zh" not in item:
            print(f"  retry {mid} (attempt {attempt + 1}: bad output)")
            continue
        item.update(motif_id=mid, equation=eq, model=WRITER_MODEL)
        v = judge(item["story_zh"])
        if v is None:
            print(f"  {mid}: judge unavailable, accepting unscreened")
            item["judge"] = None
            return item
        ok_motif = v.get("motif_id") == mid
        ok_loop = (not v.get("loops_back")) or mid == "self_reference"
        item["judge"] = {**v, "model": JUDGE_MODEL, "passed": ok_motif and ok_loop}
        if ok_motif and ok_loop:
            print(f"  ok {mid}: {item.get('title', '?')} [{item.get('domain', '?')}]"
                  + (f" (attempt {attempt + 1})" if attempt else ""))
            return item
        reasons = []
        if not ok_motif:
            reasons.append(f"盲测读者把故事的形认成了「{v.get('motif_id')}」")
        if not ok_loop:
            reasons.append("故事回到/复读了自己的开头（这属于另一个母题，禁止）")
        print(f"  rewrite {mid} (attempt {attempt + 1}): {'; '.join(reasons)}")
        feedback = ("\n上一版的问题：" + "；".join(reasons) +
                    "。请换写法修正，让形只属于目标结构。\n")
    print(f"  WARN {mid}: kept last version despite failed screening")
    return item


def main():
    OUT.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        items = [i for i in ex.map(gen_one, MOTIF_SPECS) if i]
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    passed = sum(1 for i in items if i.get("judge") and i["judge"].get("passed"))
    print(f"\n{len(items)} stories ({passed} passed screening) -> {OUT}")


if __name__ == "__main__":
    main()
