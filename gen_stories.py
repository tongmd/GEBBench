#!/usr/bin/env python3
"""GEB benchmark — story-voice generation via yunwu API (gpt-5.5).

One story per motif.  The requirement is 自然和谐: the story must be shaped
like the motif (the way 从前有座山 IS self-reference), told in a plain
folk-tale voice, with no structural vocabulary and no exact counting.

Usage:  python3 gen_stories.py [n_per_motif]
Output: data/stories_pilot.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "gpt-5.5"
OUT = Path(__file__).parent / "data" / "stories_pilot.json"

# (motif_id, equation, invariant, avoid-domains)
MOTIF_SPECS = [
    ("self_reference", "s = T(s)（f(x)=x，不动点）",
     "整体原封不动地出现在自身内部；没有底层", "山/庙/和尚（已有）"),
    ("recursion", "T = Node(T, …, T)",
     "任何一个部分单独拿出来看，形状都和整体一模一样", "河流与支流（已有）"),
    ("nesting", "S_λ(X) = X（缩放不变）",
     "每一层打开，里面是同一层的缩小版", "环形城墙（已有）"),
    ("cycle", "v ⇝ v（回到起点）",
     "沿着因果一路追问，会回到出发的地方；没有第一个", "鸡和蛋（已有）"),
    ("symmetry", "σ(x) = x（镜像不变）",
     "两半互为镜像，任何一边发生的事，另一边同时以相反的方向发生", ""),
    ("alternation", "f∘f = id（交替/周期）",
     "两种状态你来我往，谁也不是终点，交替本身永远继续", ""),
    ("spiral", "r = a·e^{bθ}（旋转×生长）",
     "一边绕着中心转，一边一圈比一圈更远（或更近），既在重复又在前进", ""),
    ("convergence", "many → one（汇聚）",
     "许多各自出发的东西，一路合并，最后归于同一处", ""),
    ("bottleneck", "所有路径必经一点",
     "无论从哪里来、到哪里去，都必须经过同一个地方", ""),
    ("symmetry_breaking", "规则场中的一个异常",
     "处处相同、事事如常之中，恰恰有一处不同，这一处让整体有了方向", ""),
]

SYSTEM = (
    "你是一位民间故事作者。你写的故事像口耳相传了几百年：句子短，词朴素，"
    "有呼吸感，读起来自然和谐，可以讲给孩子听。"
)

PROMPT = """请为下面这个抽象结构写一个小故事（中文 80~160 字）。

结构（只给你看，绝不能出现在故事里）：
- 方程：{equation}
- 不变式：{invariant}

要求：
1. 故事的「形」本身就是这个结构——像「从前有座山，山里有座庙，庙里有个老和尚在讲：从前有座山……」整个故事的讲法就是自指。结构要长在情节和讲法里，不是被人物说出来。
2. 口吻自然和谐，民间故事的质朴韵律，可朗读。
3. 严禁出现任何数学、结构类词语（如：结构、循环、递归、对称、无限、规律、重复……）。
4. 不要精确计数（不要"恰好四层""三条支流"这种数字）。
5. 选一个新鲜的生活域{avoid}。人、物、地方要有名字或质感，不要抽象代号。
6. 再写一个英文版，同样自然，不是直译腔。

只输出 JSON（不要代码块标记）：
{{"title": "四五个字的标题", "domain": "一两个词的域", "story_zh": "...", "story_en": "...", "why_shape": "一句话：故事的形在哪里体现了这个结构（审核用，可用术语）"}}"""


def call(spec) -> dict | None:
    mid, eq, inv, avoid = spec
    avoid_txt = f"（避免：{avoid}）" if avoid else ""
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(
                equation=eq, invariant=inv, avoid=avoid_txt)},
        ],
    })
    cmd = ["curl", "-s", "--max-time", "240", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/chat/completions",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json", "-d", body]
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            content = json.loads(raw)["choices"][0]["message"]["content"]
            content = re.sub(r"^```(json)?|```$", "", content.strip(),
                             flags=re.M).strip()
            item = json.loads(content)
            item["motif_id"] = mid
            item["equation"] = eq
            item["model"] = MODEL
            print(f"  ok {mid}: {item.get('title', '?')} [{item.get('domain', '?')}]")
            return item
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  retry {mid} (attempt {attempt + 1}): {e}")
    print(f"  FAIL {mid}")
    return None


def main():
    OUT.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        items = [i for i in ex.map(call, MOTIF_SPECS) if i]
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"\n{len(items)} stories -> {OUT}")


if __name__ == "__main__":
    main()
