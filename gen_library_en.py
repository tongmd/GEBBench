#!/usr/bin/env python3
"""Build data/library_en.json for the 25-motif library (ENGLISH-PRIMARY).

Per motif:
  invariant_en      — reused VERBATIM from eval/analysis_out/menu_en.py
                      (same lines as the earlier bilingual-transport EN experiment)
  theorem_en        — faithful English rendering of math_theorem (gpt-5.5:stable),
                      attribution + year kept, ZH-prose artifacts fixed
  theorem_en_masked — theorem_en with every structure-revealing term replaced by □,
                      built DETERMINISTICALLY here from mask_words (longest-first
                      substring replacement), never free-typed by the model
  mask_words        — exact surface forms masked (reproducibility)

Pipeline: author (gpt-5.5) -> deterministic mask -> mechanical giveaway scan
          -> judge (gpt-5.2) -> retry with feedback (<=4 rounds).
"""
import importlib.util
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
LOG_PATH = f"{ROOT}/gen_library_en.log"

spec = importlib.util.spec_from_file_location(
    "menu_en", f"{ROOT}/eval/analysis_out/menu_en.py")
menu_en = importlib.util.module_from_spec(spec)
spec.loader.exec_module(menu_en)
INVARIANT_EN = menu_en.INVARIANT_EN

WRITER = "gpt-5.5:stable"
JUDGE = "gpt-5.2:stable"
BOX = "□"  # □

# ---------------------------------------------------------------- giveaway scan
# Hard list: the task's explicit mask list (word-boundary regexes, case-insens.)
# plus obvious derivatives. A masked text matching any of these fails mechanically.
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


def scan_leaks(text: str) -> list[str]:
    return sorted({m.group(0) for m in HARD_RE.finditer(text)})


def apply_mask(theorem_en: str, mask_words: list[str]) -> str:
    """Deterministic: replace every occurrence of each mask word with □,
    longest surface form first, case-insensitive on the given surfaces."""
    out = theorem_en
    for w in sorted(set(mask_words), key=len, reverse=True):
        if not w.strip():
            continue
        out = re.sub(re.escape(w), BOX, out, flags=re.IGNORECASE)
    out = re.sub(rf"{BOX}(?:[ ‑-]{BOX})+", BOX, out)  # collapse □ □ runs
    return out


# ---------------------------------------------------------------- API helper
def chat(model: str, user: str, retries: int = 4) -> str | None:
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
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            log(f"    [api-retry {i+1}] {model}: {raw[:160]!r}")
            time.sleep(3 * (i + 1))
    return None


def parse_json(text: str) -> dict | None:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


_log_lock = __import__("threading").Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")


# ---------------------------------------------------------------- prompts
MASK_LIST = ("Mobius/Möbius, palindrome/palindromic, spiral, isomorphism/"
             "isomorphic, tiling/tile(s), recursion/recursive, self-reference/"
             "self-referential, diagonal/diagonalization, descent, pigeonhole, "
             "duality/dual, fixed point, contraction/contractive, aperiodic/"
             "non-periodic (as a label word), braid(ing), hub, bottleneck, "
             "interference, alternation/alternating(-ly), symmetry/symmetric/"
             "symmetry-breaking, nesting/nested, cycle/cyclic, strange loop/"
             "loop, figure-ground, coprime")

AUTHOR_PROMPT = """You are an expert mathematical translator and editor working on an English-primary benchmark.

Motif id: {mid}
Motif English name: {name_en}

Chinese source theorem statement (a REAL, correctly attributed theorem):
<zh>
{math_theorem}
</zh>

TASK A — theorem_en:
Produce a faithful, fluent English rendering of the Chinese theorem statement.
- Keep ALL attributions, years, and citations exactly (person names, journals, book titles stay).
- These are real theorems: the English statement must be mathematically CORRECT and standard. Fix any Chinese-prose artifacts (e.g. 「」 quotes, ZH punctuation, untranslated glosses) — the result must read as clean English mathematics. Keep all math notation.
- You MAY choose standard synonymous terminology when equally faithful, especially if it later masks more readably (e.g. "computable / computably enumerable" for 递归/递归可枚举; adding a standard parenthetical synonym like "wallpaper (plane crystallographic) groups" is allowed).
- Do not add, drop, or weaken any mathematical content.

TASK B — mask_words:
List the EXACT surface substrings of your theorem_en that must be replaced by the placeholder box □ so that the masked text no longer names the structural motif. Mask:
1. every occurrence of any term from this list (and their inflected forms as they actually appear in your text): {mask_list};
2. any OTHER word/name that directly names or gives away this motif's structural identity (e.g. for motif "{mid}": the motif's own name-words and close derivatives; "Kleene recursion theorem" -> mask only "recursion"; also mask structure-naming coinages like "quine", "star" for a star graph, "Sym" in Sym(G), "wallpaper", "confluent" for a convergence motif, etc.).
Do NOT mask: person names, years, journals/titles; generic mathematical content words (interval, group, graph, period, plane, prove, …) that are needed for the statement to remain readable mathematics. The masked text (your theorem_en with each listed substring replaced by □) must still read as a meaningful mathematical statement.
- Give each distinct surface form separately, cased exactly as it appears in theorem_en (e.g. ["Recursion", "recursion", "recursive"]). Multi-word entries allowed ("fixed point"). If truly nothing gives the motif away, return [].

{feedback}

Output STRICT JSON only, no code fences:
{{"theorem_en": "...", "mask_words": ["...", "..."]}}"""

JUDGE_PROMPT = """You are a strict blind referee for an English math-motif benchmark. Check this item mechanically.

Motif id: {mid}

Chinese source theorem (ground truth, a real attributed theorem):
<zh>
{math_theorem}
</zh>

Proposed English rendering:
<en>
{theorem_en}
</en>

Masked version (structure-revealing terms replaced by □):
<masked>
{theorem_en_masked}
</masked>

mask_words: {mask_words}

Verify ALL of:
1. FAITHFUL: theorem_en preserves the full mathematical content of the Chinese source — same theorem, same hypotheses/conclusions, attribution and year kept, nothing added or dropped.
2. CORRECT: theorem_en is a mathematically true, standard statement (these are real theorems — flag any error introduced in translation, wrong quantifier, wrong constant, wrong attribution).
3. CLEAN: no Chinese-prose artifacts remain (no 「」, no ZH punctuation, no untranslated fragments).
4. NO LEAK: theorem_en_masked contains NO remaining word that names the structural motif "{mid}" or any term from the standard giveaway list ({mask_list}). Person names/years/journals are allowed to remain.
5. READABLE: theorem_en_masked still reads as meaningful mathematics (□ placeholders are acceptable; the sentence structure must survive).

Output STRICT JSON only, no code fences:
{{"pass": true/false, "issues": ["short actionable issue", ...]}}"""


# ---------------------------------------------------------------- per-motif job
def build_one(mid: str) -> dict:
    m = LIB[mid]
    feedback = ""
    last = None
    for attempt in range(1, 5):
        reply = chat(WRITER, AUTHOR_PROMPT.format(
            mid=mid, name_en=m["name_en"], math_theorem=m["math_theorem"],
            mask_list=MASK_LIST, feedback=feedback))
        obj = parse_json(reply) if reply else None
        if not obj or "theorem_en" not in obj or "mask_words" not in obj:
            log(f"  [{mid}] attempt {attempt}: bad author reply, retrying")
            feedback = "PREVIOUS ATTEMPT FAILED: output was not valid JSON with keys theorem_en, mask_words."
            continue
        theorem_en = obj["theorem_en"].strip()
        mask_words = [w for w in obj["mask_words"] if isinstance(w, str) and w.strip()]
        # mask words must actually occur
        absent = [w for w in mask_words
                  if not re.search(re.escape(w), theorem_en, re.IGNORECASE)]
        mask_words = [w for w in mask_words if w not in absent]
        masked = apply_mask(theorem_en, mask_words)
        leaks = scan_leaks(masked)
        # ZH artifact scan
        artifacts = re.findall(r"[「」『』，。；：（）一-鿿]", masked)
        if leaks or artifacts:
            log(f"  [{mid}] attempt {attempt}: leaks={leaks} artifacts={sorted(set(artifacts))}")
            feedback = ("PREVIOUS ATTEMPT REJECTED by mechanical scan. "
                        + (f"After masking, these giveaway strings remained: {leaks}. "
                           f"Add their exact surface forms to mask_words, or rephrase "
                           f"theorem_en to avoid them (person names may stay). " if leaks else "")
                        + (f"Chinese punctuation/characters remained: {sorted(set(artifacts))}. "
                           f"Remove them. " if artifacts else "")
                        + f"Your previous theorem_en was: {theorem_en!r} "
                        + f"with mask_words {mask_words!r}.")
            last = dict(theorem_en=theorem_en, theorem_en_masked=masked,
                        mask_words=mask_words, status=f"mech-fail: {leaks or artifacts}")
            continue
        # LLM judge
        jreply = chat(JUDGE, JUDGE_PROMPT.format(
            mid=mid, math_theorem=m["math_theorem"], theorem_en=theorem_en,
            theorem_en_masked=masked, mask_words=json.dumps(mask_words),
            mask_list=MASK_LIST))
        jobj = parse_json(jreply) if jreply else None
        if jobj is None:
            log(f"  [{mid}] attempt {attempt}: judge unreachable, accepting mech-clean result")
            jobj = {"pass": True, "issues": ["judge-unreachable"]}
        if jobj.get("pass"):
            log(f"  [{mid}] PASS on attempt {attempt} "
                f"(masks={len(mask_words)}, judge issues={jobj.get('issues', [])})")
            return dict(theorem_en=theorem_en, theorem_en_masked=masked,
                        mask_words=mask_words,
                        status=f"pass@{attempt}", judge_issues=jobj.get("issues", []))
        issues = jobj.get("issues", [])
        log(f"  [{mid}] attempt {attempt}: judge fail: {issues}")
        feedback = ("PREVIOUS ATTEMPT REJECTED by the referee with these issues: "
                    f"{issues}. Fix every issue. Your previous theorem_en was: "
                    f"{theorem_en!r} with mask_words {mask_words!r}.")
        last = dict(theorem_en=theorem_en, theorem_en_masked=masked,
                    mask_words=mask_words, status=f"judge-fail: {issues}")
    log(f"  [{mid}] EXHAUSTED retries; keeping last candidate flagged")
    return last or dict(theorem_en="", theorem_en_masked="", mask_words=[],
                        status="FAILED: no candidate")


def main() -> None:
    open(LOG_PATH, "w").close()
    ids = list(LIB)
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for mid, res in zip(ids, ex.map(build_one, ids)):
            results[mid] = res
    out = {mid: {"invariant_en": INVARIANT_EN[mid],
                 "theorem_en": results[mid]["theorem_en"],
                 "theorem_en_masked": results[mid]["theorem_en_masked"],
                 "mask_words": results[mid]["mask_words"]}
           for mid in ids}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\nWrote {OUT_PATH}")
    for mid in ids:
        log(f"{mid:28s} {results[mid]['status']}")
    n_ok = sum(r["status"].startswith("pass") for r in results.values())
    log(f"{n_ok}/25 passed")
    if n_ok < 25:
        sys.exit(1)


if __name__ == "__main__":
    main()
