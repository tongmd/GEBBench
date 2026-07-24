#!/usr/bin/env python3
"""Final mechanical validation of data/library_en.json."""
import json
import re
import sys

ROOT = "/Users/stone/GEB-benchmark"
lib = {m["id"]: m for m in json.load(open(f"{ROOT}/data/motif_library.json"))}
en = json.load(open(f"{ROOT}/data/library_en.json"))

HARD_PATTERNS = [
    r"m[oö]bius", r"palindrom\w*", r"spiral\w*", r"isomorph\w*",
    r"\btil(?:e|es|ed|ing|ings)\b", r"recursi\w*", r"self.referen\w*",
    r"diagonal\w*", r"\bdescent\b", r"pigeonhole\w*", r"\bdual\w*",
    r"fixed.point\w*", r"contract(?:ion|ions|ive|ing|s)\b",
    r"a\W?periodic\w*", r"braid\w*", r"\bhubs?\b",
    r"bottleneck\w*", r"interfer\w*", r"alternat\w*", r"symmetr\w*",
    r"\bnest(?:ed|ing|s)?\b", r"\bcycles?\b", r"\bcyclic\w*",
    r"strange.loop\w*", r"\bloops?\b", r"figure.ground", r"co.?prime\w*",
    r"\bSym\b", r"\bquines?\b",
]
HARD_RE = re.compile("|".join(HARD_PATTERNS), re.IGNORECASE)

ok = True
assert set(en) == set(lib), "id mismatch"
for mid, e in en.items():
    problems = []
    for k in ("invariant_en", "theorem_en", "theorem_en_masked", "mask_words"):
        if k not in e:
            problems.append(f"missing {k}")
    if not e["theorem_en"].strip():
        problems.append("empty theorem_en")
    # mask words all occur in theorem_en
    for w in e["mask_words"]:
        if not re.search(re.escape(w), e["theorem_en"], re.IGNORECASE):
            problems.append(f"mask word not in theorem_en: {w!r}")
    # reproduce masked deterministically
    m = e["theorem_en"]
    for w in sorted(set(e["mask_words"]), key=len, reverse=True):
        m = re.sub(re.escape(w), "□", m, flags=re.IGNORECASE)
    m = re.sub(r"□(?:[ ‑-]□)+", "□", m)
    if m != e["theorem_en_masked"]:
        problems.append("theorem_en_masked NOT reproducible from mask_words")
    leaks = sorted({x.group(0) for x in HARD_RE.finditer(e["theorem_en_masked"])})
    if leaks:
        problems.append(f"LEAKS in masked: {leaks}")
    zh_art = re.findall(r"[「」『』，。；：（）一-鿿]", e["theorem_en"] + e["invariant_en"])
    if zh_art:
        problems.append(f"ZH artifacts: {sorted(set(zh_art))}")
    if "□" not in e["theorem_en_masked"] and e["mask_words"]:
        problems.append("mask_words nonempty but no □ in masked")
    status = "OK " if not problems else "FAIL"
    if problems:
        ok = False
    print(f"{status} {mid:28s} masks={len(e['mask_words']):2d} "
          f"{'; '.join(problems)}")
print("\nALL OK" if ok else "\nPROBLEMS FOUND")
sys.exit(0 if ok else 1)
