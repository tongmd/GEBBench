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

# (results-file stem, display label) — all API models via OpenRouter (or_*),
# open-weight local models via kiti vLLM (unchanged tags)
MODELS = [
    ("or_gpt-5.5", "GPT-5.5"),
    ("or_gpt-5.2", "GPT-5.2"),
    ("or_claude-opus-4.8", "Claude Opus 4.8"),
    ("or_gemini-3.1-pro", "Gemini 3.1 Pro"),
    ("or_kimi-k2.5", "Kimi K2.5"),
    ("or_qwen2.5-vl-72b", "Qwen2.5-VL-72B"),
    ("or_deepseek-v3.2", "DeepSeek-V3.2"),
    ("qwen2.5-vl-32b", "Qwen2.5-VL-32B"),
    ("deepseek-coder-v2-lite", "DeepSeek-Coder-V2-Lite"),
    ("qwen2.5-14b", "Qwen2.5-14B"),
    ("qwen2.5-7b", "Qwen2.5-7B"),
    ("qwen2-vl-7b", "Qwen2-VL-7B"),
]
FRONTIER_ROWS = 4   # midrule after the proprietary frontier block
OPEN_API_ROWS = 7   # second midrule after the open-weights-via-API block

# Open-model extension (2026-08, tom 7x4090, local vLLM, identical protocol):
# the three strongest open vision models plus one representative per remaining
# vendor family enter the main table; everything else goes to the appendix.
EXT_MAIN = [
    ("glm-4.1v-9b", "GLM-4.1V-9B"),
    ("qwen3-vl-8b", "Qwen3-VL-8B"),
    ("gemma-3-27b", "Gemma-3-27B"),
    ("internvl3.5-8b", "InternVL3.5-8B"),
    ("kimi-vl-a3b", "Kimi-VL-A3B"),
    ("pixtral-12b", "Pixtral-12B"),
    ("minicpm-v-4", "MiniCPM-V-4"),
    ("nemotron-nano-vl-8b", "Nemotron-Nano-VL-8B"),
]
EXT_APPENDIX = [
    ("gemma-3-12b", "Gemma-3-12B"),
    ("internvl3-8b", "InternVL3-8B"),
    ("glm-4-9b", "GLM-4-9B"),
    ("qwen3-4b", "Qwen3-4B"),
    ("gemma-2-9b", "Gemma-2-9B"),
    ("llama-3.1-8b", "Llama-3.1-8B"),
    ("mistral-7b-v0.3", "Mistral-7B-v0.3"),
    ("olmo-2-7b", "OLMo-2-7B"),
]
OUT_EXT_TEX = os.path.join(ROOT, "paper", "tables", "extended_models.tex")

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
    empty = 0
    for r in d["records"]:
        t = r["task"]
        hits[t] = hits.get(t, 0) + (1 if r["correct"] else 0)
        tot[t] = tot.get(t, 0) + 1
        if not str(r.get("raw") or "").strip():
            empty += 1
    for t, n in tot.items():
        assert n == EXPECTED_N[t], (stem, t, n)
    # An empty reply is a dropped connection, not a wrong answer: every frontier
    # endpoint here is reached through one SOCKS tunnel, and a saturated tunnel
    # drops requests. Scoring cannot tell the two apart -- it counts an
    # unparsable reply as wrong -- so a transport failure would otherwise land
    # in the table as a real accuracy. Two runs were lost to exactly this
    # (90/90 empty, acc=0.0000); see eval/analysis_out/quarantine/README.md.
    # A result whose json carries an explicit "note" documenting its empties
    # (e.g. kimi-vl-a3b: one reply lost to a server OOM window, provenance in
    # the note field) is allowed up to 2 of them.
    allowed = 2 if d.get("note") else 0
    assert empty <= allowed, (
        f"{stem}: {empty}/{sum(tot.values())} records have an empty reply. "
        "These are failed calls, not wrong answers -- re-run the model before "
        "regenerating the table.")
    if empty:
        print(f"  note: {stem} has {empty} documented empty repl"
              f"{'y' if empty == 1 else 'ies'} (see its note field)")
    return {t: 100.0 * hits[t] / tot[t] for t in tot}


def cell(accs, t):
    return f"{accs[t]:.1f}" if t in accs else "--"


HEADER = [
    r"\toprule",
    r" & \multicolumn{4}{c}{$25$-way identification}"
    r" & \multicolumn{4}{c}{$4$-way} & \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"Model & Scene & Skel. & Story & Thm. & Sc$\to$St & Sc$\to$Th & St$\to$Th"
    r" & Adv. & \textbf{Avg} \\",
    r"\midrule",
]


def rows_for(models, summary):
    lines = []
    for stem, label in models:
        accs = load_accs(stem)
        avg = sum(accs.values()) / len(accs)
        summary[stem] = {"accs": accs, "avg": avg, "n_tasks": len(accs)}
        row = [label] + [cell(accs, t) for t in TASKS] + \
              [r"\textbf{%.1f}" % avg]
        lines.append(" & ".join(row) + r" \\")
    return lines


def chance_row():
    return [r"\midrule",
            r"\textit{Random chance} & "
            + " & ".join(f"{CHANCE[t]:.0f}" for t in TASKS)
            + r" & -- \\"]


def main():
    summary = {}
    # ---- main table: core roster + open-model extension block ----
    lines = [r"\begin{tabular}{lccccccccc}"] + HEADER
    core_rows = rows_for(MODELS, summary)
    for i, row in enumerate(core_rows):
        lines.append(row)
        if i in (FRONTIER_ROWS - 1, OPEN_API_ROWS - 1):
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.extend(rows_for(EXT_MAIN, summary))
    lines.extend(chance_row())
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT_TEX)

    # ---- appendix table: remaining extension arms ----
    lines = [r"\begin{tabular}{lccccccccc}"] + HEADER
    lines.extend(rows_for(EXT_APPENDIX, summary))
    lines.extend(chance_row())
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(OUT_EXT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT_EXT_TEX)

    with open(os.path.join(ROOT, "eval", "analysis_v2",
                           "main_table_numbers.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for stem, s in summary.items():
        print(f"{stem:26s} avg={s['avg']:.1f} over {s['n_tasks']} tasks")


if __name__ == "__main__":
    main()
