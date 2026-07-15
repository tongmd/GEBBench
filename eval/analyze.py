#!/usr/bin/env python3
"""GEB benchmark — aggregate eval results into LaTeX tables + confusion data.

Reads eval/results/*.json (skips smoke_*), emits:
  paper/tables/main_results.tex   per-model per-task accuracy
  paper/tables/confusion.tex      most-confused motif pairs (pooled over models)
  eval/results/_aggregate.json    machine-readable summary

Usage:  python3 eval/analyze.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
RES = ROOT / "eval" / "results"
LIB = {m["id"]: m for m in
       json.loads((ROOT / "data" / "motif_library.json").read_text())}

TASKS = ["scene2motif", "skeleton2motif", "story2motif",
         "theorem2motif", "crossvoice"]
TASK_SHORT = {"scene2motif": "Scene", "skeleton2motif": "Skel",
              "story2motif": "Story", "theorem2motif": "Thm",
              "crossvoice": "Cross"}
# nicer display names
DISPLAY = {"api_gpt-5.5": "GPT-5.5", "api_gpt-5.2": "GPT-5.2",
           "qwen2-vl-7b": "Qwen2-VL-7B", "qwen2.5-7b": "Qwen2.5-7B",
           "qwen2.5-14b": "Qwen2.5-14B", "qwen2.5-coder-32b": "Qwen2.5-Coder-32B"}
ORDER = ["api_gpt-5.5", "api_gpt-5.2", "qwen2.5-coder-32b",
         "qwen2.5-14b", "qwen2.5-7b", "qwen2-vl-7b"]


def load() -> dict:
    out = {}
    for f in RES.glob("*.json"):
        if f.stem.startswith(("smoke", "_")):
            continue
        out[f.stem] = json.loads(f.read_text())
    return out


def main():
    runs = load()
    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)

    # ---- main results table ----
    tags = [t for t in ORDER if t in runs] + [t for t in runs if t not in ORDER]
    lines = [r"\begin{tabular}{l" + "c" * (len(TASKS) + 1) + "}", r"\toprule",
             "Model & " + " & ".join(TASK_SHORT[t] for t in TASKS)
             + r" & \textbf{Avg} \\", r"\midrule"]
    agg = {}
    for tag in tags:
        summ = runs[tag].get("summary", {})
        cells, accs = [], []
        for t in TASKS:
            if t in summ:
                a = summ[t]["acc"]
                accs.append(a)
                cells.append(f"{a*100:.1f}")
            else:
                cells.append("--")
        avg = sum(accs) / len(accs) if accs else 0
        name = DISPLAY.get(tag, tag)
        lines.append(f"{name} & " + " & ".join(cells)
                     + f" & \\textbf{{{avg*100:.1f}}} " + r"\\")
        agg[tag] = {"tasks": summ, "avg": round(avg, 4)}
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "main_results.tex").write_text("\n".join(lines))

    # ---- per-task random baseline (18-way, crossvoice 4-way) ----
    n_motifs = len(LIB)
    baseline = {t: (1 / 4 if t == "crossvoice" else 1 / n_motifs) for t in TASKS}

    # ---- confusion: pooled predictions on the *motif-id* tasks ----
    confuse = Counter()
    per_motif = defaultdict(lambda: [0, 0])  # motif -> [correct, total]
    for tag, run in runs.items():
        for r in run.get("records", []):
            if r["task"] == "crossvoice":
                continue
            ans = r["answer"]
            per_motif[ans][1] += 1
            if r["correct"]:
                per_motif[ans][0] += 1
            elif r["pred"] in LIB:
                confuse[(ans, r["pred"])] += 1

    top = confuse.most_common(12)
    clines = [r"\begin{tabular}{llc}", r"\toprule",
              r"True motif & Mistaken for & \# \\", r"\midrule"]
    for (a, b), n in top:
        clines.append(f"{a.replace('_',' ')} & {b.replace('_',' ')} & {n} " + r"\\")
    clines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "confusion.tex").write_text("\n".join(clines))

    # hardest motifs (lowest recall pooled)
    recall = {m: c / t for m, (c, t) in per_motif.items() if t}
    hardest = sorted(recall.items(), key=lambda kv: kv[1])[:6]

    out = {"models": agg, "baseline": baseline,
           "top_confusions": [{"true": a, "pred": b, "n": n}
                              for (a, b), n in top],
           "hardest_motifs": [{"motif": m, "recall": round(r, 3)}
                             for m, r in hardest],
           "n_motifs": n_motifs}
    (RES / "_aggregate.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"models": {DISPLAY.get(k, k): v["avg"]
                                 for k, v in agg.items()},
                      "hardest": hardest,
                      "top_confusions": top[:6]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
