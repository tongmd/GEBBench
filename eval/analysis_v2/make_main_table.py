#!/usr/bin/env python3
"""Emit paper/tables/main_results.tex from eval/results_v2/*.json.

Design:
  rows    = models (frontier first, then open weights by size), chance row last
  columns = Scene, Skel, Story, Thm (25-way, 4% chance)
            Cross, XvSc, XvSt, Fam4 (4-way, 25% chance)
            Avg (macro average over attempted tasks)
  text-only models attempt Story, Thm, XvSt and show -- elsewhere.
"""
import json
import os

ROOT = "/Users/stone/GEB-benchmark"
RES = os.path.join(ROOT, "eval", "results_v2")
OUT_TEX = os.path.join(ROOT, "paper", "tables", "main_results.tex")

# (results-file stem, display label)
MODELS = [
    ("api_gpt-5.5", "GPT-5.5"),
    ("api_gpt-5.2", "GPT-5.2"),
    ("api_claude-opus-4-8", "Claude Opus 4.8"),
    ("qwen2.5-vl-32b", "Qwen2.5-VL-32B"),
    ("deepseek-coder-v2-lite", "DeepSeek-Coder-V2-Lite"),
    ("qwen2.5-14b", "Qwen2.5-14B"),
    ("qwen2.5-7b", "Qwen2.5-7B"),
    ("qwen2-vl-7b", "Qwen2-VL-7B"),
]
FRONTIER_ROWS = 3  # midrule after these

TASKS = ["scene2motif", "skeleton2motif", "story2motif", "theorem2motif",
         "crossvoice", "xv_scene_thm", "xv_story_thm", "scene2motif_family4"]
EXPECTED_N = {"scene2motif": 210, "skeleton2motif": 49, "story2motif": 121,
              "theorem2motif": 25, "crossvoice": 210, "xv_scene_thm": 210,
              "xv_story_thm": 121, "scene2motif_family4": 210}
CHANCE = {"scene2motif": 4.0, "skeleton2motif": 4.0, "story2motif": 4.0,
          "theorem2motif": 4.0, "crossvoice": 25.0, "xv_scene_thm": 25.0,
          "xv_story_thm": 25.0, "scene2motif_family4": 25.0}


def load_accs(stem):
    path = os.path.join(RES, stem + ".json")
    d = json.load(open(path))
    hits, tot = {}, {}
    for r in d["records"]:
        t = r["task"]
        hits[t] = hits.get(t, 0) + (1 if r["correct"] else 0)
        tot[t] = tot.get(t, 0) + 1
    for t, n in tot.items():
        assert n == EXPECTED_N[t], (stem, t, n)
    return {t: 100.0 * hits[t] / tot[t] for t in tot}


def cell(accs, t):
    return f"{accs[t]:.1f}" if t in accs else "--"


def main():
    lines = []
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c}{$25$-way identification}"
                 r" & \multicolumn{4}{c}{$4$-way} & \\")
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    lines.append(r"Model & Scene & Skel & Story & Thm & Cross & XvSc & XvSt"
                 r" & Fam4 & \textbf{Avg} \\")
    lines.append(r"\midrule")
    summary = {}
    for i, (stem, label) in enumerate(MODELS):
        accs = load_accs(stem)
        avg = sum(accs.values()) / len(accs)
        summary[stem] = {"accs": accs, "avg": avg, "n_tasks": len(accs)}
        row = [label] + [cell(accs, t) for t in TASKS] + \
              [r"\textbf{%.1f}" % avg]
        lines.append(" & ".join(row) + r" \\")
        if i == FRONTIER_ROWS - 1:
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.append(r"\textit{Random chance} & "
                 + " & ".join(f"{CHANCE[t]:.0f}" for t in TASKS)
                 + r" & -- \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT_TEX)
    with open(os.path.join(ROOT, "eval", "analysis_v2",
                           "main_table_numbers.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for stem, s in summary.items():
        print(f"{stem:26s} avg={s['avg']:.1f} over {s['n_tasks']} tasks")


if __name__ == "__main__":
    main()
