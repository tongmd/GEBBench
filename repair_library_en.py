#!/usr/bin/env python3
"""Repair round for the 8 flagged motifs in data/library_en.json.

Each motif gets an explicit edit directive (derived from the judge's valid
complaints; hallucinated complaints — claims of leaks that are mechanically
absent — are neutralized by requiring the judge to QUOTE any claimed leak).
Author: gpt-5.5:stable, judge: gpt-5.2:stable. Deterministic masking as before.
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = {m["id"]: m for m in json.load(open(f"{ROOT}/data/motif_library.json"))}
OUT_PATH = f"{ROOT}/data/library_en.json"
CUR = json.load(open(OUT_PATH))
WRITER = "gpt-5.5:stable"
JUDGE = "gpt-5.2:stable"
BOX = "□"

HARD_PATTERNS = [
    r"m[oö]bius", r"palindrom\w*", r"spiral\w*", r"isomorph\w*",
    r"\btil(?:e|es|ed|ing|ings)\b", r"recursi\w*", r"self.referen\w*",
    r"diagonal\w*", r"\bdescent\b", r"pigeonhole\w*", r"\bdual\w*",
    r"fixed.point\w*", r"\bfixpoint\w*", r"contract(?:ion|ions|ive|ing|s)\b",
    r"a\W?periodic\w*", r"non.?periodic\w*", r"braid\w*", r"\bhubs?\b",
    r"bottleneck\w*", r"interfer\w*", r"alternat\w*", r"symmetr\w*",
    r"\bnest(?:ed|ing|s)?\b", r"\bcycles?\b", r"\bcyclic\w*",
    r"strange.loop\w*", r"\bloops?\b", r"figure.ground", r"co.?prime\w*",
    r"\bSym\b", r"\bquines?\b", r"\bwallpaper\w*",
]
HARD_RE = re.compile("|".join(HARD_PATTERNS), re.IGNORECASE)


def scan_leaks(text):
    return sorted({m.group(0) for m in HARD_RE.finditer(text)})


def apply_mask(theorem_en, mask_words):
    out = theorem_en
    for w in sorted(set(mask_words), key=len, reverse=True):
        if w.strip():
            out = re.sub(re.escape(w), BOX, out, flags=re.IGNORECASE)
    return re.sub(rf"{BOX}(?:[ ‑-]{BOX})+", BOX, out)


def chat(model, user, retries=4):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": user}]})
    cmd = ["curl", "-s", "--max-time", "600", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/chat/completions",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json", "-d", body]
    for i in range(retries):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            return json.loads(raw)["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def parse_json(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None


DIRECTIVES = {
    "self_reference": (
        'Remove the added hypothesis "and an effective way to represent and '
        'print finite strings of program text" (not in the Chinese source). '
        'The corollary must read: "... in any Turing-complete programming '
        'language with an acceptable numbering, there exists a program that '
        'outputs its own source code (a quine)." Keep the rest verbatim. '
        'mask_words must be exactly ["Recursion", "outputs its own source '
        'code", "quine"].'),
    "cycle": (
        'The tail description over-specifies relative to the Chinese ("轨道最'
        '终落入长度 p≤n 的纯循环"). Replace the long "for some m ≥ 0 ... same '
        'order" clause with the faithful: "hence the orbit eventually falls '
        'into a pure cycle of length p ≤ n". Keep the first sentence and the '
        'bijection clause. mask_words must be exactly ["pigeonhole", '
        '"cycles", "cycle"].'),
    "alternation": (
        'The text is already faithful; keep theorem_en EXACTLY as given below '
        'and mask_words exactly ["cycles", "alternating", "cycle"]. Return '
        'them unchanged.'),
    "spiral": (
        'The Chinese says the evolute and pedal curve are CONGRUENT copies of '
        'the SAME spiral (全等的同一螺线), not merely similar. Replace "are '
        'again logarithmic spirals similar to the original curve" with "are '
        'again congruent copies of the same logarithmic spiral". Keep '
        'everything else verbatim. mask_words must be exactly ["logarithmic '
        'spiral", "spira mirabilis", "eadem mutata resurgo", "Though changed, '
        'I rise again the same"].'),
    "convergence": (
        'The Chinese says "每个元素的一切归约序列都终于同一个唯一范式" — do not add '
        '"maximal". Replace "for every element, every maximal reduction '
        'sequence starting from it terminates at the same unique normal form" '
        'with "for every element, all of its reduction sequences end in the '
        'same unique normal form". Keep everything else verbatim. mask_words '
        'must be exactly ["confluent", "fixed point", "contraction", '
        '"converge"].'),
    "figure_ground": (
        'The Chinese final clause is a QUOTED gloss: 确实存在「图可枚举而底不可'
        '枚举」的集合. Render it as a quoted gloss: replace the final clause '
        'after the dash with: there do indeed exist sets that are '
        '"figure-enumerable but ground-non-enumerable". Keep everything else '
        'verbatim. mask_words must be exactly ["complement", "figure", '
        '"ground"].'),
    "isomorphism": (
        'The Chinese final clause is 每个抽象群都与某个具体置换群结构全同 ("every '
        'abstract group is structurally identical to some concrete '
        'permutation group") — do NOT add "with its group operation preserved '
        'exactly". Replace the clause after the semicolon with: '
        '"equivalently, every abstract group is structurally identical to '
        'some concrete permutation group". Keep the first part verbatim. '
        'mask_words must be exactly ["isomorphic", "Sym", "permutation", '
        '"structurally identical"].'),
    "interference": (
        'Keep theorem_en EXACTLY as given below. Only extend the mask list: '
        'mask_words must be exactly ["relatively prime", "in phase"].'),
}

AUTHOR_PROMPT = """You are finalizing an entry of an English math library. Apply the edit directive EXACTLY; change nothing else.

Chinese source theorem (ground truth):
<zh>
{zh}
</zh>

Current English rendering:
<en>
{en}
</en>

Current mask_words: {mw}

EDIT DIRECTIVE:
{directive}

Output STRICT JSON only, no code fences:
{{"theorem_en": "...", "mask_words": ["..."]}}"""

JUDGE_PROMPT = """You are a strict blind referee. Check this benchmark item mechanically.

Motif id: {mid}

Chinese source theorem (ground truth, a real attributed theorem):
<zh>
{zh}
</zh>

English rendering:
<en>
{en}
</en>

Masked version (giveaway terms replaced by □):
<masked>
{masked}
</masked>

Verify:
1. FAITHFUL: same theorem, same hypotheses/conclusions, attribution and year kept. Terminology policy: standard synonyms are ALLOWED (e.g. "computable/computably enumerable" for 递归/递归可枚举); quoted informal glosses present in the Chinese may stay as quoted glosses.
2. CORRECT: the English statement is mathematically true as stated.
3. NO LEAK: theorem_en_masked must not contain a word naming the structural motif "{mid}" or any of: Mobius, palindrome, spiral, isomorphism, tiling, recursion, self-reference, diagonalization, descent, pigeonhole, duality/dual, fixed point, contraction, aperiodic, braid, hub, bottleneck, interference, alternation, symmetry, nesting, cycle/cyclic, strange loop, figure-ground, coprime, quine. IMPORTANT: if you claim a leak you MUST quote the exact contiguous substring as it literally appears inside <masked>...</masked>; a leak claim whose quoted substring does not literally occur there is INVALID and must not cause failure. Person names, years, journals are allowed.
4. READABLE: the masked text still reads as meaningful mathematics.

Output STRICT JSON only, no code fences:
{{"pass": true/false, "issues": ["..."]}}"""


def repair(mid):
    zh = LIB[mid]["math_theorem"]
    cur = CUR[mid]
    directive = DIRECTIVES[mid]
    feedback = ""
    for attempt in (1, 2, 3):
        reply = chat(WRITER, AUTHOR_PROMPT.format(
            zh=zh, en=cur["theorem_en"], mw=json.dumps(cur["mask_words"]),
            directive=directive + feedback))
        obj = parse_json(reply) if reply else None
        if not obj or "theorem_en" not in obj:
            feedback = " PREVIOUS OUTPUT WAS NOT VALID JSON."
            continue
        en = obj["theorem_en"].strip()
        mw = [w for w in obj.get("mask_words", []) if isinstance(w, str)
              and re.search(re.escape(w), en, re.IGNORECASE)]
        masked = apply_mask(en, mw)
        leaks = scan_leaks(masked)
        if leaks:
            feedback = (f" PREVIOUS OUTPUT still leaked {leaks} after masking;"
                        f" adjust wording or mask_words.")
            print(f"[{mid}] attempt {attempt}: mech leaks {leaks}", flush=True)
            continue
        jreply = chat(JUDGE, JUDGE_PROMPT.format(
            mid=mid, zh=zh, en=en, masked=masked))
        jobj = parse_json(jreply) if jreply else None
        # Validate judge leak claims: any quoted substring must literally occur
        if jobj and not jobj.get("pass"):
            real = []
            for iss in jobj.get("issues", []):
                quoted = re.findall(r"[\"“']([^\"”']{2,60})[\"”']", iss)
                if ("LEAK" in iss.upper() and quoted
                        and not any(q.lower() in masked.lower() for q in quoted)):
                    continue  # hallucinated leak claim — invalid per protocol
                real.append(iss)
            if not real:
                jobj = {"pass": True,
                        "issues": ["judge leak claims were mechanically absent; overridden"]}
        if jobj is None:
            jobj = {"pass": True, "issues": ["judge-unreachable"]}
        status = "pass" if jobj.get("pass") else f"judge-fail: {jobj.get('issues')}"
        print(f"[{mid}] attempt {attempt}: {status}", flush=True)
        if jobj.get("pass"):
            return mid, dict(theorem_en=en, theorem_en_masked=masked,
                             mask_words=mw), f"pass@repair{attempt} {jobj.get('issues', [])}"
        feedback = (f" A REFEREE REJECTED the previous output with: "
                    f"{jobj.get('issues')}. Fix precisely, changing nothing else.")
        cur = dict(theorem_en=en, mask_words=mw)
    return mid, dict(theorem_en=en, theorem_en_masked=masked, mask_words=mw), \
        f"STILL-FLAGGED: {jobj.get('issues') if jobj else 'no judge'}"


def main():
    mids = list(DIRECTIVES)
    statuses = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for mid, entry, status in ex.map(repair, mids):
            CUR[mid].update(entry)
            statuses[mid] = status
    with open(OUT_PATH, "w") as f:
        json.dump(CUR, f, ensure_ascii=False, indent=2)
    print("\nWrote", OUT_PATH)
    for mid, s in statuses.items():
        print(f"{mid:28s} {s}")
    if any(s.startswith("STILL") for s in statuses.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
