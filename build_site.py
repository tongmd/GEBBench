#!/usr/bin/env python3
"""GEB benchmark — build the static preview site from examples/.

Reads every examples/<sample_id>/ directory, copies preview images into
site/assets/, and generates site/index.html showing all six voices per sample.

Usage:  python3 build_site.py
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
EXAMPLES = ROOT / "examples"
SITE = ROOT / "site"

FAMILY_LABEL = {
    "graph_world": ("Graph Worlds", "topology"),
    "recursive_garden": ("Recursive Gardens", "hierarchy"),
    "symmetry_lab": ("Symmetry Labs", "geometry"),
}

IMAGE_LABEL = {
    "canonical": "canonical",
    "style_blueprint": "blueprint · renderer",
    "style_pencil": "pencil · renderer",
    "style_gpt_chalkboard": "chalkboard · gpt-image",
    "style_gpt_papercut": "paper cut-out · gpt-image",
    "style_gpt_botanical_ink": "botanical ink · gpt-image",
    "style_gpt_woodcut": "woodcut · gpt-image",
    "style_gpt_ceramic_tile": "ceramic tile · gpt-image",
    "style_gpt_neon_sign": "neon sign · gpt-image",
}

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEB Benchmark — One Structure, Six Voices</title>
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
  .qa { color: var(--dim); font-size: 15px; margin: 2px 0 16px; }
  .qa b { color: var(--ink); }
  .strip { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 6px; }
  figure { margin: 0; flex: 0 0 auto; }
  figure img { height: 210px; width: auto; display: block; border: 1px solid var(--line);
               border-radius: 6px; background: #fff; }
  figcaption { font: 12px/1.5 Menlo, monospace; color: var(--dim); padding-top: 5px;
               text-align: center; }
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
  .mathv h1 { display: none; }
  footer { margin-top: 88px; padding-top: 20px; border-top: 1px solid var(--line);
           color: var(--dim); font-size: 14px; }
  a { color: var(--accent); }
  h2.part { margin: 72px 0 -28px; font-size: 21px; color: var(--dim);
            font-weight: normal; font-style: italic; border-bottom: 1px solid var(--line);
            padding-bottom: 10px; }
  .motif-eq { font-size: 18px; margin: 4px 0 2px; }
  .invariant { color: var(--dim); font-style: italic; margin: 2px 0 16px; }
  figure.skel img { height: 120px; opacity: 0.75; }
  .nuisance { font: 12px/1.6 Menlo, monospace; color: var(--dim); margin-top: 8px; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1><span class="geb">GEB</span> Benchmark</h1>
  <p class="tagline">One structure, six voices.</p>
  <div class="chain">z&nbsp;(latent schema) &nbsp;&rarr;&nbsp; { <b>S</b>tory, <b>M</b>athematics,
    <b>P</b>rogram, <b>D</b>rawing&nbsp;Program, canonical&nbsp;<b>I</b>mage,
    stylized&nbsp;<b>I</b><sub>style</sub> }</div>
  <p class="intro">A benchmark of <b>abstract structures</b>, in the spirit of
  <em>Gödel, Escher, Bach</em>. The unit is a structural <b>motif</b> — self-reference,
  recursion, nesting, cycle — not any particular graph: node counts, depths and angles
  are nuisance variables. Each motif speaks with several voices: a folk-tale shaped like
  it, an equation fixed by it, and <b>natural or artistic images whose composition
  <em>is</em> the structure</b> — a vis-graph skeleton is used as the reference for scene
  generation, so you look at a landscape and the structure dawns on you.</p>
</header>
<h2 class="part">Ⅰ · Motifs — see a scene, grasp a structure</h2>
@MOTIFS@
<h2 class="part">Ⅱ · Diagram families (v0 pilot: verifiable instances)</h2>
@SAMPLES@
<footer>GEB Benchmark · One Structure, Six Voices · examples v0 ·
  images: deterministic SVG renderer + gpt-image stylization</footer>
</div>
</body>
</html>
"""

SAMPLE_TPL = """<section class="sample" id="@SID@">
  <h2><span class="fam-badge">@FAM_SHORT@</span>@SID@</h2>
  <p class="qa">@QUERY@ &nbsp;·&nbsp; answer: <b>@ANSWER@</b></p>
  <div class="strip">@FIGURES@</div>
  <details><summary>story voice</summary><div class="voice story">@STORY@</div></details>
  <details><summary>mathematics voice</summary><div class="voice mathv">@MATH@</div></details>
  <details><summary>program voice (self-checking)</summary><div class="voice"><pre>@PROGRAM@</pre></div></details>
  <details><summary>drawing-program voice</summary><div class="voice"><pre>@DRAWING@</pre></div></details>
  <details><summary>latent schema (ground truth)</summary><div class="voice"><pre>@SCHEMA@</pre></div></details>
</section>
"""

MOTIF_TPL = """<section class="sample" id="motif-@MID@">
  <h2><span class="fam-badge">motif</span>@NAME@</h2>
  <p class="motif-eq">$$@EQ@$$</p>
  <p class="invariant">invariant: @INVARIANT@</p>
  <div class="strip">@FIGURES@</div>
  <details><summary>story voice — a tale with this shape</summary><div class="voice story">@STORY@</div></details>
  <details><summary>mathematics voice</summary><div class="voice mathv">@MATH@</div></details>
  <details><summary>motif schema</summary><div class="voice"><pre>@SCHEMA@</pre></div></details>
  <p class="nuisance">nuisance (deliberately varied, never scored): @NUISANCE@</p>
</section>
"""

MOTIF_ORDER = ["self_reference", "recursion", "nesting", "cycle"]

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
    """Keep it dependency-free: bold + paragraphs; $$ blocks left for KaTeX."""
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


def describe_query(schema: dict) -> tuple[str, str]:
    fam, task = schema["family"], schema["task_type"]
    a = schema["answer"]
    if fam == "graph_world" and task == "reachability":
        q = schema["query"]
        label = f"reachability: {q['source']} &rarr; {q['target']} ?"
        ans = "yes — " + " → ".join(a["witness_path"]) if a["reachable"] else "no"
    elif fam == "graph_world":
        label = "cycle detection: does the graph contain a directed cycle?"
        ans = "yes — " + " → ".join(a["witness_cycle"]) if a["has_cycle"] else "no"
    elif fam == "recursive_garden":
        label = f"leaf count after depth-{schema['depth']} binary recursion?"
        ans = str(a["leaf_count"])
    else:
        label = "order of the rotation acting on the asymmetric shape?"
        ans = f"C{a['rotation_order']} (order {a['rotation_order']})"
    return label, ans


def build_sample(d: Path) -> str:
    schema = json.loads((d / "schema.json").read_text())
    sid = schema["sample_id"]
    fam = schema["family"]
    assets = SITE / "assets" / sid
    assets.mkdir(parents=True, exist_ok=True)

    figures = []
    order = ["canonical"] + [k for k in IMAGE_LABEL if k.startswith("style_gpt")] + \
            ["style_blueprint", "style_pencil"]
    for stem in order:
        png = d / f"{stem}.png"
        if not png.exists():
            continue
        shutil.copy2(png, assets / png.name)
        rel = f"assets/{sid}/{png.name}"
        figures.append(f'<figure><a href="{rel}"><img loading="lazy" src="{rel}" '
                       f'alt="{sid} {stem}"></a>'
                       f"<figcaption>{IMAGE_LABEL[stem]}</figcaption></figure>")

    label, ans = describe_query(schema)
    out = SAMPLE_TPL
    for k, v in {
        "@SID@": sid,
        "@FAM_SHORT@": FAMILY_LABEL[fam][0],
        "@QUERY@": label,
        "@ANSWER@": esc(ans),
        "@FIGURES@": "\n".join(figures),
        "@STORY@": story_html((d / "story.txt").read_text()),
        "@MATH@": math_html((d / "math.md").read_text()),
        "@PROGRAM@": esc((d / "program.py").read_text()),
        "@DRAWING@": esc(json.dumps(json.loads((d / "drawing.json").read_text()), indent=2)),
        "@SCHEMA@": esc(json.dumps(schema, indent=2)),
    }.items():
        out = out.replace(k, v)
    return out


def build_motif(d: Path) -> str:
    schema = json.loads((d / "schema.json").read_text())
    mid = schema["motif_id"]
    assets = SITE / "assets" / f"motif_{mid}"
    assets.mkdir(parents=True, exist_ok=True)

    figures = []
    scenes = sorted(d.glob("scene_*.png"))
    skels = sorted(d.glob("skeleton_*.png"))
    for png in scenes + skels:
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
        "@STORY@": story_html((d / "story.txt").read_text()),
        "@MATH@": math_html((d / "math.md").read_text()),
        "@SCHEMA@": esc(json.dumps(schema, indent=2, ensure_ascii=False)),
        "@NUISANCE@": esc(", ".join(schema["nuisance_variables"])),
    }.items():
        out = out.replace(k, v)
    return out


def main():
    SITE.mkdir(exist_ok=True)
    motif_root = ROOT / "motifs"
    motifs = [build_motif(motif_root / mid) for mid in MOTIF_ORDER
              if (motif_root / mid / "schema.json").exists()]
    sections = [build_sample(d) for d in sorted(EXAMPLES.iterdir())
                if d.is_dir() and (d / "schema.json").exists()
                and json.loads((d / "schema.json").read_text()).get("family")
                in FAMILY_LABEL]
    page = PAGE.replace("@MOTIFS@", "\n".join(motifs)) \
               .replace("@SAMPLES@", "\n".join(sections))
    (SITE / "index.html").write_text(page)
    print(f"site/index.html built with {len(motifs)} motifs + {len(sections)} samples")


if __name__ == "__main__":
    main()
