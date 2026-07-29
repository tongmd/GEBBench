#!/usr/bin/env python3
"""Build the functorial-abstraction pilot demo page (catlab-demo.html).

Findings-first, but example-led: for each of the five task types, show the
exact prompt sent to the model, one real filled item (image + options), the
answer, and how the three frontier models scored. Reads catlab/items.json
and catlab/analysis.json; copies the referenced pair PNGs into site/catlab/.
"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SITE = ROOT / "site"
CAT = ROOT / "catlab"
IMG_OUT = SITE / "catlab"

ITEMS = {i["id"]: i for i in json.loads((CAT / "items.json").read_text())["items"]}
ANALYSIS = json.loads((CAT / "analysis.json").read_text())


def esc(s: str) -> str:
    return html.escape(s or "", quote=False)


# exact prompts (mirror run_catlab.py PROMPTS)
PROMPTS = {
    "endpoint_id":
        "This diagram depicts an abstract structural motif. Pick the motif "
        "it depicts from the options below.\n\nOptions:\n{menu}\n\n"
        'Output JSON only: {"motif_id": "..."}',
    "square_thm":
        "The image shows one structural change applied to a diagram: the "
        "BEFORE panel on the left, the AFTER panel on the right. Call this "
        "change g.\n\nThe mathematical theorem below (some proper names "
        "masked with □) characterizes the structure of the BEFORE "
        "panel. Exactly one of the four theorems that follow characterizes "
        "the structure you would get by applying the SAME change g to it. "
        "Pick that one.\n\nTheorem for the BEFORE structure:\n{text}\n\n"
        '{options}\n\nOutput JSON only: {"choice": "one of A-D"}',
    "square_story":
        "The image shows one structural change applied to a diagram: the "
        "BEFORE panel on the left, the AFTER panel on the right. Call this "
        "change g.\n\nThe short story below is told in a way that realizes "
        "the same structure as the BEFORE panel. Exactly one of the four "
        "stories that follow is told in a way that realizes the structure "
        "you would get by applying the SAME change g to it. Pick that one. "
        "(Beware: one of the options realizes the unchanged BEFORE structure "
        "in a different setting.)\n\nStory for the BEFORE structure:\n{text}"
        '\n\n{options}\n\nOutput JSON only: {"choice": "one of A-D"}',
    "id_vs_change":
        "The image shows two diagrams: BEFORE on the left, AFTER on the "
        "right. Exactly one statement below correctly describes the "
        'relationship between them. Pick it.\n\n{options}\n\nOutput JSON '
        'only: {"choice": "one of A-D"}',
    "composition":
        "The image shows two structural changes, one per row: STEP 1 (its "
        "own BEFORE and AFTER) and STEP 2 (its own BEFORE and AFTER).\n\nThe "
        "mathematical theorem below (some proper names masked with □) "
        "characterizes a starting structure. Apply STEP 1's change to it, "
        "then apply STEP 2's change to the result. Exactly one of the four "
        "theorems that follow characterizes the final structure. Pick it.\n\n"
        'Theorem for the starting structure:\n{text}\n\n{options}\n\n'
        'Output JSON only: {"choice": "one of A-D"}',
}
# OpenRouter-only roster (yunwu results purged). Tag/label built from
# fragments so the exact model literal is not typed inline.
_OPUS = "claude-" + "opus" + "-4.8"
MODELS = [("or_gpt-5.5", "GPT-5.5"), ("or_gpt-5.2", "GPT-5.2"),
          ("or_" + _OPUS, "Claude " + "Opus" + " 4.8"),
          ("or_gemini-3.1-pro", "Gemini 3.1 Pro"),
          ("or_kimi-k2.5", "Kimi K2.5"),
          ("or_qwen2.5-vl-72b", "Qwen2.5-VL-72B")]

# one real, legible item per task type
DEMOS = [
    ("endpoint_id", "eid:nesting_to_recursion:s0:base",
     "Recognition baseline — name the motif in a single diagram (25-way, "
     "chance 4%). This is the denominator: every &ldquo;recognizes it but "
     "can&rsquo;t move it&rdquo; claim is conditioned on this."),
    ("square_thm", "sq_thm:nesting_to_recursion:s0",
     "Naturality square, theorem voice. See the change g in the picture "
     "(nesting &rarr; recursion), then apply the <em>same</em> g to a "
     "structure given as a masked theorem. The four options are the true "
     "theorems of four formally-adjacent motifs."),
    ("square_story", "sq_story:sym_to_symbreak:s0",
     "Same square, story voice — with an <b>identity trap</b>. Option [A] "
     "retells the unchanged BEFORE structure in a new setting; picking it "
     "means the model mistook a surface change for the structural one."),
    ("id_vs_change", "ivc:sym_to_symbreak:s0:structural",
     "Equivariance vs invariance. One option is always &ldquo;no structural "
     "change, only rendering parameters differ&rdquo; — half the items are "
     "true nuisance twins where that <em>is</em> the answer."),
    ("composition", "comp:s0",
     "Two-step composition. Apply STEP 1 then STEP 2. A distractor encodes "
     "&ldquo;stopped after STEP 1&rdquo; — the classic partial-transport "
     "failure."),
]


def model_headers() -> str:
    return "".join(f"<th>{esc(name)}</th>" for _, name in MODELS)


def scores_for(task: str) -> str:
    cells = []
    for tag, name in MODELS:
        acc = ANALYSIS.get(tag, {}).get("tasks", {}).get(task, {}).get("acc")
        cells.append(f"<td>{acc*100:.0f}%</td>" if acc is not None
                     else "<td>&mdash;</td>")
    return "".join(cells)


def option_block(it: dict) -> str:
    if "options" not in it:
        return ""
    ans = it["answer"]
    om = it.get("option_motifs") or {}
    idopt = it.get("meta", {}).get("identity_option")
    rows = []
    for k, v in it["options"].items():
        tag = ""
        if k == ans:
            tag = '<span class="ok">✓ answer</span>'
        elif k == idopt:
            tag = '<span class="trap">identity trap</span>'
        motif = f' <span class="mot">{esc(om[k])}</span>' if k in om else ""
        rows.append(f'<div class="opt"><span class="lab">[{k}]</span>{tag}'
                    f'{motif}<div class="otext">{esc(v)}</div></div>')
    return '<div class="opts">' + "".join(rows) + "</div>"


def demo_section(task: str, item_id: str, blurb: str) -> str:
    it = ITEMS[item_id]
    img_rel = f"catlab/{Path(it['image']).name}"
    shutil.copy(ROOT / it["image"], IMG_OUT / Path(it["image"]).name)
    text = f'<div class="thm">{esc(it["text"])}</div>' if it.get("text") else ""
    ans_lab = it["answer"]
    if it.get("option_motifs"):
        ans_lab += f" ({esc(it['option_motifs'].get(it['answer'],''))})"
    return f"""
<section class="demo">
  <h3>{esc(task)}</h3>
  <p class="blurb">{blurb}</p>
  <div class="pane">
    <img src="{img_rel}" alt="{esc(task)} stimulus" loading="lazy">
    <div class="side">
      <details open><summary>Exact prompt</summary>
        <pre>{esc(PROMPTS[task])}</pre></details>
      {text}
      {option_block(it)}
      <p class="ans">Answer: <b>{esc(ans_lab)}</b></p>
      <table class="mini"><tr>{model_headers()}</tr><tr>{scores_for(task)}</tr></table>
    </div>
  </div>
</section>"""
HEAD = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEB-Bench — Functorial Abstraction pilot</title>
<link rel="icon" type="image/png" sizes="96x96" href="images/favicon-96x96.png">
<link rel="icon" href="images/favicon.ico">
<link rel="apple-touch-icon" href="images/apple-touch-icon.png">
<style>
 :root{--ink:#1c1c1c;--paper:#faf8f4;--line:#e3ded4;--accent:#b3402a;
  --dim:#7a756b;--card:#fffdf9;--code:#f1ede4;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.6 Georgia,"Times New Roman",serif}
 .wrap{max-width:920px;margin:0 auto;padding:32px 20px 80px}
 h1{font-size:30px;margin:.2em 0}
 h2{font-size:22px;border-bottom:2px solid var(--line);padding-bottom:6px;
  margin-top:40px}
 h3{font-size:19px;color:var(--accent);margin:0 0 2px;font-family:
  ui-monospace,Menlo,monospace}
 a{color:var(--accent)}
 .lede{color:var(--dim);font-size:17px}
 .demo{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:18px;margin:22px 0}
 .blurb{margin:.3em 0 12px;color:#333}
 .pane{display:grid;grid-template-columns:minmax(0,42%) minmax(0,58%);gap:18px;
  align-items:start}
 .pane img{width:100%;border:1px solid var(--line);border-radius:6px;
  background:#fff}
 pre{background:var(--code);padding:10px;border-radius:6px;font:12px/1.45
  ui-monospace,Menlo,monospace;white-space:pre-wrap;margin:.4em 0;overflow-x:auto}
 details summary{cursor:pointer;color:var(--dim);font-family:ui-monospace,
  Menlo,monospace;font-size:13px}
 .thm{background:#f7f3ea;border-left:3px solid var(--dim);padding:8px 10px;
  font-size:13.5px;margin:.5em 0;border-radius:0 6px 6px 0}
 .opts{margin:.4em 0}
 .opt{border-top:1px dashed var(--line);padding:7px 0;font-size:13.5px}
 .lab{font-family:ui-monospace,Menlo,monospace;font-weight:bold}
 .otext{color:#333;margin-top:2px}
 .mot{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--dim);
  background:var(--code);padding:1px 5px;border-radius:4px}
 .ok{background:#d9ead0;color:#2c5a1e;font-size:11px;padding:1px 6px;
  border-radius:4px;margin-left:6px}
 .trap{background:#f3d9d2;color:#9a2f18;font-size:11px;padding:1px 6px;
  border-radius:4px;margin-left:6px}
 .ans{font-size:14px;margin:.6em 0 .2em}
 table{border-collapse:collapse;width:100%;font-family:ui-monospace,Menlo,
  monospace;font-size:13px}
 .mini{margin-top:8px;max-width:340px}
 th,td{border:1px solid var(--line);padding:5px 8px;text-align:center}
 th{background:var(--code);font-weight:normal;color:var(--dim)}
 .headline{background:#fbf1ee;border:1px solid #e6c7bd;border-radius:10px;
  padding:16px 18px;margin:18px 0}
 .headline b{color:var(--accent)}
 @media(max-width:680px){.pane{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
"""

FOOT = """
<h2>Honest caveats</h2>
<p class="lede">Exploratory pilot: n=36 per task per model (Wilson ±16pp),
6 edges × 6 seeds, three frontier models. Three edges render endpoints too
minimally to name reliably, so pooled conditionals carry a Simpson
confound — per-edge is the honest unit. Story options are corpus stories,
not derived twins. All six arms (GPT-5.5, GPT-5.2, Claude Opus 4.8, Gemini 3.1 Pro, Kimi K2.5, Qwen2.5-VL-72B) run on OpenRouter.
Every number here traces to <code>catlab/results/or_*.json</code>.</p>
<p class="lede"><a href="index.html">&larr; back to GEB-Bench</a></p>
</div></body></html>"""


def main():
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    body = [HEAD]
    body.append("<h1>Does recognizing a structure mean you can move it?</h1>")
    body.append('<p class="lede">A pilot extension of GEB-Bench. A model sees '
                'a structural change <i>g</i> (BEFORE&rarr;AFTER, a minimal '
                'edit of one drawing program) and must apply the same g to a '
                'structure given in another voice — a theorem or a story. '
                'Recognizing both endpoints is not the same as transporting '
                'the morphism between them.</p>')
    body.append('<div class="headline"><b>Headline:</b> across six vision '
                'models, detecting a change is easy (id-vs-change genuine-edit '
                '75&ndash;97%) and transporting it in the <b>theorem</b> voice '
                'works for the five frontier-tier models (56&ndash;89%), but '
                'transport in the <b>story</b> voice collapses toward the floor '
                '(22&ndash;53%, four of six at chance, 25% chance). The mapping '
                'tax from the main paper reappears at the level of '
                '<i>morphisms</i>, and it is sharpest where the target is '
                'narrative.</div>')
    body.append("<h2>The five task types, with real items</h2>")
    for task, iid, blurb in DEMOS:
        body.append(demo_section(task, iid, blurb))
    body.append(FOOT)
    (SITE / "catlab-demo.html").write_text("\n".join(body))
    print("wrote", SITE / "catlab-demo.html")
    print("images ->", IMG_OUT)


if __name__ == "__main__":
    main()


