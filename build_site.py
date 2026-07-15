#!/usr/bin/env python3
"""GEB benchmark — build the static preview site from motifs/.

Copies scene/skeleton images into site/assets/ and generates site/index.html.

Usage:  python3 build_site.py
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
MOTIFS_DIR = ROOT / "motifs"
STORIES_FILE = ROOT / "data" / "stories_pilot.json"
SITE = ROOT / "site"

MOTIF_ORDER = ["self_reference", "recursion", "nesting", "cycle"]

# story-first motifs: stories exist, skeletons/scenes not yet generated
NEW_MOTIF_ORDER = ["symmetry", "alternation", "spiral",
                   "convergence", "bottleneck", "symmetry_breaking"]
NEW_MOTIF_NAMES = {
    "symmetry": "Symmetry — two halves mirror each other",
    "alternation": "Alternation — two states take turns",
    "spiral": "Spiral — turning while growing",
    "convergence": "Convergence — many become one",
    "bottleneck": "Bottleneck — every path passes one point",
    "symmetry_breaking": "Symmetry breaking — one exception gives direction",
}

SCENE_LABEL = {
    "mountain_temple": "山中庙 · ink wash",
    "droste_study": "study within study · photo",
    "river_delta": "river delta · aerial",
    "lightning": "lightning · long exposure",
    "winter_tree": "winter tree · photo",
    "walled_city": "walled city · aerial",
    "pond_ripples": "pond ripples · photo",
    "ouroboros": "ouroboros · ink & gold",
    "canal_town": "canal town · aerial",
}

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEB Benchmark — One Structure, Many Voices</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});"></script>
<style>
  :root {
    --ink: #1c1c1c; --paper: #faf8f4; --line: #e3ded4;
    --accent: #b3402a; --dim: #7a756b; --code-bg: #f1ede4;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink);
         font: 16px/1.65 Georgia, "Times New Roman", serif; }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 0 24px 96px; }
  header { padding: 72px 0 24px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 44px; margin: 0 0 8px; letter-spacing: -0.5px; }
  h1 .geb { color: var(--accent); }
  .tagline { font-size: 20px; font-style: italic; color: var(--dim); margin: 0 0 18px; }
  .chain { font-family: "SF Mono", Menlo, monospace; font-size: 14px;
           background: var(--code-bg); border: 1px solid var(--line);
           border-radius: 8px; padding: 14px 18px; overflow-x: auto; white-space: nowrap; }
  .chain b { color: var(--accent); }
  .intro { max-width: 760px; }
  .fam-badge { display: inline-block; font: 12px/1 Menlo, monospace; color: #fff;
               background: var(--accent); border-radius: 4px; padding: 4px 8px;
               margin-right: 8px; vertical-align: middle; }
  section.sample { margin-top: 64px; }
  h2 { font-size: 26px; margin: 0 0 4px; }
  .strip { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 6px; }
  figure { margin: 0; flex: 0 0 auto; }
  figure img { height: 220px; width: auto; display: block; border: 1px solid var(--line);
               border-radius: 6px; background: #fff; }
  figcaption { font: 12px/1.5 Menlo, monospace; color: var(--dim); padding-top: 5px;
               text-align: center; }
  figure.skel img { height: 120px; opacity: 0.75; }
  details { border: 1px solid var(--line); border-radius: 6px; margin-top: 10px;
            background: #fffdf9; }
  summary { cursor: pointer; padding: 10px 16px; font-family: Menlo, monospace;
            font-size: 13px; color: var(--ink); user-select: none; }
  summary:hover { color: var(--accent); }
  .voice { padding: 4px 20px 18px; }
  pre { background: var(--code-bg); border: 1px solid var(--line); border-radius: 6px;
        padding: 14px 16px; overflow-x: auto; font: 13px/1.55 "SF Mono", Menlo, monospace; }
  .story p { margin: 8px 0; }
  .story .dom { font-family: Menlo, monospace; font-size: 12px; color: var(--accent); }
  .mathv { font-size: 15px; }
  .motif-eq { font-size: 18px; margin: 4px 0 2px; }
  .invariant { color: var(--dim); font-style: italic; margin: 2px 0 16px; }
  .nuisance { font: 12px/1.6 Menlo, monospace; color: var(--dim); margin-top: 8px; }
  .tale { border-left: 3px solid var(--accent); background: #fffdf9;
          padding: 14px 20px; margin: 16px 0 6px; border-radius: 0 6px 6px 0; }
  .tale .t-title { font-weight: bold; margin: 0 0 6px; }
  .tale .t-domain { color: var(--dim); font: 12px/1 Menlo, monospace; font-weight: normal; }
  .tale p.zh { margin: 6px 0; }
  .tale p.en { margin: 6px 0; color: var(--dim); font-style: italic; font-size: 14.5px; }
  .tale .t-why { font: 12px/1.6 Menlo, monospace; color: var(--dim); margin: 8px 0 0; }
  .fam-badge.pending { background: var(--dim); }
  .motif-eq.mono { font-family: "SF Mono", Menlo, monospace; font-size: 15px; }
  h2.part { margin: 72px 0 -28px; font-size: 21px; color: var(--dim);
            font-weight: normal; font-style: italic; border-bottom: 1px solid var(--line);
            padding-bottom: 10px; }
  footer { margin-top: 88px; padding-top: 20px; border-top: 1px solid var(--line);
           color: var(--dim); font-size: 14px; }
  a { color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1><span class="geb">GEB</span> Benchmark</h1>
  <p class="tagline">See a scene, grasp a structure.</p>
  <div class="chain">motif&nbsp;z &nbsp;&rarr;&nbsp; skeleton (vis-graph&nbsp;ref)
    &nbsp;&rarr;&nbsp; { natural / artistic <b>scenes</b>, <b>story</b> shaped like it,
    <b>math</b> fixed by it }</div>
  <p class="intro">A benchmark of <b>abstract structures</b>, in the spirit of
  <em>Gödel, Escher, Bach</em>. The unit is a structural <b>motif</b> — self-reference,
  recursion, nesting, cycle — not any particular graph: node counts, depths and angles
  are nuisance variables. Each motif speaks with several voices: a folk-tale shaped like
  it, an equation fixed by it, and <b>natural or artistic images whose composition
  <em>is</em> the structure</b> — a vis-graph skeleton is used as the reference for scene
  generation, so you look at a landscape and the structure dawns on you.</p>
</header>
@MOTIFS@
<footer>GEB Benchmark · see a scene, grasp a structure ·
  scenes: gpt-image with programmatic skeletons as references</footer>
</div>
</body>
</html>
"""

MOTIF_TPL = """<section class="sample" id="motif-@MID@">
  <h2><span class="fam-badge">motif</span>@NAME@</h2>
  <p class="motif-eq">$$@EQ@$$</p>
  <p class="invariant">invariant: @INVARIANT@</p>
  <div class="strip">@FIGURES@</div>
  @TALE@
  <details><summary>story voice — a tale with this shape</summary><div class="voice story">@STORY@</div></details>
  <details><summary>mathematics voice</summary><div class="voice mathv">@MATH@</div></details>
  <details><summary>motif schema</summary><div class="voice"><pre>@SCHEMA@</pre></div></details>
  <p class="nuisance">nuisance (deliberately varied, never scored): @NUISANCE@</p>
</section>
"""

TALE_TPL = """<div class="tale">
  <p class="t-title">《@TITLE@》 <span class="t-domain">@DOMAIN@ · @MODEL@@SCREEN@</span></p>
  <p class="zh">@ZH@</p>
  <p class="en">@EN@</p>
  <p class="t-why">why this shape — @WHY@</p>
</div>
"""

NEW_MOTIF_TPL = """<section class="sample" id="motif-@MID@">
  <h2><span class="fam-badge pending">motif</span>@NAME@</h2>
  <p class="motif-eq mono">@EQ@</p>
  @TALE@
  <p class="nuisance">scenes: pending — skeleton renderer + gpt-image next</p>
</section>
"""


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def story_html(text: str) -> str:
    out = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if block.startswith("[domain:"):
            dom, _, rest = block.partition("]")
            out.append(f'<p><span class="dom">{esc(dom)}]</span></p>')
            if rest.strip():
                out.append(f"<p>{esc(rest.strip())}</p>")
        else:
            out.append(f"<p>{esc(block)}</p>")
    return "\n".join(out)


def math_html(md: str) -> str:
    out = []
    for block in md.strip().split("\n\n"):
        block = block.strip()
        if block.startswith("#"):
            continue
        block = esc(block)
        while "**" in block:
            block = block.replace("**", "<b>", 1).replace("**", "</b>", 1)
        out.append(f"<p>{block}</p>")
    return "\n".join(out)


def load_stories() -> dict:
    if not STORIES_FILE.exists():
        return {}
    by_motif = {}
    for s in json.loads(STORIES_FILE.read_text()):
        by_motif.setdefault(s["motif_id"], []).append(s)
    return by_motif


def tale_html(story: dict) -> str:
    j = story.get("judge")
    screen = " · screened ✓" if j and j.get("passed") else ""
    out = TALE_TPL
    for k, v in {
        "@TITLE@": esc(story.get("title", "")),
        "@DOMAIN@": esc(story.get("domain", "")),
        "@MODEL@": esc(story.get("model", "")),
        "@SCREEN@": screen,
        "@ZH@": esc(story.get("story_zh", "")),
        "@EN@": esc(story.get("story_en", "")),
        "@WHY@": esc(story.get("why_shape", "")),
    }.items():
        out = out.replace(k, v)
    return out


def build_new_motif(mid: str, stories: dict) -> str:
    ss = stories.get(mid, [])
    eq = ss[0].get("equation", "") if ss else ""
    tale = "\n".join(tale_html(s) for s in ss)
    out = NEW_MOTIF_TPL
    for k, v in {"@MID@": mid, "@NAME@": esc(NEW_MOTIF_NAMES[mid]),
                 "@EQ@": esc(eq), "@TALE@": tale}.items():
        out = out.replace(k, v)
    return out


def build_motif(d: Path, stories: dict) -> str:
    schema = json.loads((d / "schema.json").read_text())
    mid = schema["motif_id"]
    assets = SITE / "assets" / f"motif_{mid}"
    assets.mkdir(parents=True, exist_ok=True)

    figures = []
    for png in sorted(d.glob("scene_*.png")) + sorted(d.glob("skeleton_*.png")):
        shutil.copy2(png, assets / png.name)
        rel = f"assets/motif_{mid}/{png.name}"
        if png.stem.startswith("scene_"):
            label = SCENE_LABEL.get(png.stem.removeprefix("scene_"),
                                    png.stem.removeprefix("scene_"))
            cls = ""
        else:
            label = "skeleton (generation ref)"
            cls = ' class="skel"'
        figures.append(f'<figure{cls}><a href="{rel}"><img loading="lazy" src="{rel}" '
                       f'alt="{mid} {png.stem}"></a>'
                       f"<figcaption>{esc(label)}</figcaption></figure>")

    out = MOTIF_TPL
    for k, v in {
        "@MID@": mid,
        "@NAME@": esc(schema["name"]),
        "@EQ@": schema["equation"],
        "@INVARIANT@": esc(schema["invariant"]),
        "@FIGURES@": "\n".join(figures),
        "@TALE@": "\n".join(tale_html(s) for s in stories.get(mid, [])),
        "@STORY@": story_html((d / "story.txt").read_text()),
        "@MATH@": math_html((d / "math.md").read_text()),
        "@SCHEMA@": esc(json.dumps(schema, indent=2, ensure_ascii=False)),
        "@NUISANCE@": esc(", ".join(schema["nuisance_variables"])),
    }.items():
        out = out.replace(k, v)
    return out


def main():
    if (SITE / "assets").exists():
        shutil.rmtree(SITE / "assets")
    SITE.mkdir(exist_ok=True)
    stories = load_stories()
    motifs = [build_motif(MOTIFS_DIR / mid, stories) for mid in MOTIF_ORDER
              if (MOTIFS_DIR / mid / "schema.json").exists()]
    new = [build_new_motif(mid, stories) for mid in NEW_MOTIF_ORDER
           if stories.get(mid)]
    if new:
        motifs.append('<h2 class="part">New motifs — story voice first, '
                      'scenes coming</h2>')
        motifs.extend(new)
    (SITE / "index.html").write_text(PAGE.replace("@MOTIFS@", "\n".join(motifs)))
    n_stories = sum(len(v) for v in stories.values())
    print(f"site/index.html built with {len(motifs) - (1 if new else 0)} motif "
          f"sections, {n_stories} stories")


if __name__ == "__main__":
    main()
