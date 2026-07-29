#!/usr/bin/env python3
"""Compute every number the paper's catlab section cites, straight from
catlab/results/or_*.json.  Each printed line is keyed to a \\pending{slot}
in sections/results.tex / sections/appendix.tex, so the fill is mechanical
and every value traces to a run log.

Frontier roster (paper order): GPT-5.5, GPT-5.2, Claude Opus 4.8, Kimi K2.5.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OPUS = "or_claude-" + "opus" + "-4.8"          # assembled; see build_catlab_demo
TAGS = ["or_gpt-5.5", "or_gpt-5.2", OPUS, "or_gemini-3.1-pro",
        "or_kimi-k2.5", "or_qwen2.5-vl-72b"]
NAMES = dict(zip(TAGS, ["GPT-5.5", "GPT-5.2", "Claude-Opus-4.8",
                        "Gemini-3.1-Pro", "Kimi-K2.5", "Qwen2.5-VL-72B"]))


def load(tag):
    d = json.loads((ROOT / "results" / f"{tag}.json").read_text())
    by = {}
    for r in d["records"]:
        by.setdefault(r["task"], []).append(r)
    return by


def binom_p_ge(k, n, p=0.25):
    """one-sided P(X >= k), X~Bin(n,p)"""
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def mcnemar_exact(b, c):
    """two-sided exact McNemar on discordant counts b, c"""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, lo + 1)) * 2 / 2**n
    return min(1.0, p)


def pct(x):
    return f"{100*x:.1f}"


def main():
    data = {t: load(t) for t in TAGS}
    any_by = data[TAGS[0]]

    n_pairs = len({(r["meta"]["edge"], r["meta"]["seed"])
                   for r in any_by["square_thm"]})
    n_items = sum(len(v) for v in any_by.values())
    print(f"[N-pairs]  {n_pairs} reference pairs (square_thm edge,seed keys)")
    print(f"[N-items]  {n_items} items per model")
    print(f"[n-per]    {len(any_by['square_thm'])} items per square task per model")
    print(f"[comp-n]   {len(any_by.get('composition', []))} composition items")
    print()

    # --- per-model task rows (tab:catlab) --------------------------------
    cols = "abcdef"
    slots = {}
    for i, t in enumerate(TAGS):
        by = data[t]
        acc = lambda task: (sum(r["correct"] for r in by[task]) / len(by[task]))
        eid = acc("endpoint_id")
        sqt = acc("square_thm")
        sqs = acc("square_story")
        st = [r for r in by["id_vs_change"] if r["meta"]["kind"] == "structural"]
        nu = [r for r in by["id_vs_change"] if r["meta"]["kind"] == "nuisance"]
        ivs = sum(r["correct"] for r in st) / len(st)
        ivn = sum(r["correct"] for r in nu) / len(nu)
        cmp_rows = by.get("composition", [])
        cmp_acc = sum(r["correct"] for r in cmp_rows) / len(cmp_rows) if cmp_rows else float("nan")
        c = cols[i]
        slots.update({f"eid-{c}": eid, f"sqt-{c}": sqt, f"sqs-{c}": sqs,
                      f"ivs-{c}": ivs, f"ivn-{c}": ivn, f"cmp-{c}": cmp_acc})
        print(f"[{NAMES[t]:<16}] eid {pct(eid)}  sq_thm {pct(sqt)}  "
              f"sq_story {pct(sqs)}  ivc_st {pct(ivs)}  ivc_nu {pct(ivn)}  "
              f"comp {pct(cmp_acc)}")
    print()
    for k in sorted(slots):
        v = slots[k]
        print(f"[{k}] {pct(v)}\\%")
    print()

    # --- prose ranges -----------------------------------------------------
    used = cols[:len(TAGS)]
    thm = [slots[f"sqt-{c}"] for c in used]
    sty = [slots[f"sqs-{c}"] for c in used]
    ivs_ = [slots[f"ivs-{c}"] for c in used]
    ivn_ = [slots[f"ivn-{c}"] for c in used]
    print(f"[thm-lo/hi]   {pct(min(thm))} -- {pct(max(thm))}")
    print(f"[story-lo/hi] {pct(min(sty))} -- {pct(max(sty))}")
    print(f"[ivc-struct-lo/hi] {pct(min(ivs_))} -- {pct(max(ivs_))}")
    print(f"[ivc-nuis-lo/hi]   {pct(min(ivn_))} -- {pct(max(ivn_))}")
    print(f"[ivc-fa-lo/hi]     {pct(1-max(ivn_))} -- {pct(1-min(ivn_))}")
    print()

    # --- story vs chance & McNemar thm/story ------------------------------
    n_nonsig = 0
    worst_mcnemar = 0.0
    min_story_p = 1.0
    for t in TAGS:
        by = data[t]
        srows = by["square_story"]
        k = sum(r["correct"] for r in srows)
        p_ch = binom_p_ge(k, len(srows))
        min_story_p = min(min_story_p, p_ch) if p_ch >= 0.05 else min_story_p
        sig = p_ch < 0.05
        if not sig:
            n_nonsig += 1
        tmap = {(r["meta"]["edge"], r["meta"]["seed"]): r["correct"]
                for r in by["square_thm"]}
        b = c = 0
        for r in srows:
            key = (r["meta"]["edge"], r["meta"]["seed"])
            if key not in tmap:
                continue
            tc, sc = tmap[key], r["correct"]
            if tc and not sc:
                b += 1
            elif sc and not tc:
                c += 1
        pm = mcnemar_exact(b, c)
        worst_mcnemar = max(worst_mcnemar, pm)
        print(f"[{NAMES[t]:<16}] story k={k}/{len(srows)} binom_p={p_ch:.3f} "
              f"({'sig' if sig else 'NOT sig'} vs chance)   "
              f"McNemar thm>story b={b} c={c} p={pm:.4f}")
    print(f"[story-nsig]  {n_nonsig} of six models at chance (binom p>=0.05)")
    print(f"[sq-mcnemar]  worst-case (max) exact McNemar p = {worst_mcnemar:.3f}")
    print()

    # --- identity trap pooled ---------------------------------------------
    trap_k = trap_n = 0
    for t in TAGS:
        for r in data[t]["square_story"]:
            if not r["correct"] and r["pred"] and r["pred"] in "ABCD":
                trap_n += 1
                trap_k += int(r["pred"] == r["meta"].get("identity_option"))
    share = trap_k / trap_n if trap_n else float("nan")
    # two-sided binomial vs 1/3
    p13 = sum(math.comb(trap_n, i) * (1/3)**i * (2/3)**(trap_n - i)
              for i in range(trap_k, trap_n + 1))
    lo = sum(math.comb(trap_n, i) * (1/3)**i * (2/3)**(trap_n - i)
             for i in range(0, trap_k + 1))
    p2 = min(1.0, 2 * min(p13, lo))
    print(f"[trap-share] pooled {trap_k}/{trap_n} = {pct(share)}% of story "
          f"errors chose the identity option (chance 33.3%), two-sided p={p2:.2f}")
    print()

    # --- double dissociation: cycle->strange_loop -------------------------
    for t in TAGS:
        by = data[t]
        after = [r for r in by["endpoint_id"]
                 if r["meta"]["edge"] == "cycle_to_strangeloop"
                 and r["meta"]["side"] == "img"]
        named = sum(r["correct"] for r in after)
        sq = [r for r in by["square_thm"]
              if r["meta"]["edge"] == "cycle_to_strangeloop"]
        sq_acc = sum(r["correct"] for r in sq) / len(sq) if sq else float("nan")
        print(f"[csl {NAMES[t]:<16}] AFTER named {named}/{len(after)}   "
              f"thm-square {pct(sq_acc)}%")
    # pooled
    named_all = sum(sum(r["correct"] for r in data[t]["endpoint_id"]
                        if r["meta"]["edge"] == "cycle_to_strangeloop"
                        and r["meta"]["side"] == "img") for t in TAGS)
    n_all = sum(len([r for r in data[t]["endpoint_id"]
                     if r["meta"]["edge"] == "cycle_to_strangeloop"
                     and r["meta"]["side"] == "img"]) for t in TAGS)
    print(f"[csl-endpoint pooled] {named_all}/{n_all}")
    print()

    # --- ceiling gap (from analysis.json, regenerate first) ----------------
    an = json.loads((ROOT / "analysis.json").read_text())
    for tag in TAGS:
        a = an.get(tag, {})
        for sq in ("square_thm", "square_story"):
            q = a.get(sq)
            if q:
                print(f"[ceil {NAMES[tag]:<16} {sq:<12}] "
                      f"acc-ceiling = {q['acc_minus_ceiling']:+.3f}")
    print()
    print("edges present:", sorted({r["meta"]["edge"] for r in any_by["square_thm"]}))


if __name__ == "__main__":
    main()
