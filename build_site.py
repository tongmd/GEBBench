#!/usr/bin/env python3
"""GEB benchmark — build the static preview site.

Sources:
  data/motif_library.json   verified motif library (math + form devices)
  motifs/<id>/              skeletons + scenes (gen_motifs.py)
  data/stories_v2.json      form-device stories (gen_stories.py)
  data/stories_pilot.json   v1 pilot stories (kept as extra tales)

Usage:  python3 build_site.py
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
MOTIFS_DIR = ROOT / "motifs"
SITE = ROOT / "site"

LIBRARY = {m["id"]: m for m in
           json.loads((ROOT / "data" / "motif_library.json").read_text())}

# conceptual arc: self-reference family -> loops & twists -> mirrors &
# correspondences -> weaves & periods -> fields -> flows
MOTIF_ORDER = [
    "self_reference", "strange_loop", "recursion", "nesting",
    "cycle", "mobius", "spiral",
    "symmetry", "figure_ground", "isomorphism",
    "alternation", "braiding", "interference", "tiling",
    "symmetry_breaking", "convergence", "bottleneck", "hub",
]

SCENE_LABEL = {
    "mountain_temple": "山中庙 · ink wash",
    "droste_study": "study within study · photo",
    "river_delta": "river delta · aerial",
    "lightning": "lightning · long exposure",
    "winter_tree": "winter tree · photo",
    "walled_city": "walled city · aerial",
    "pond_ripples": "pond ripples · photo",
    "moon_gates": "moon gates · photo",
    "ouroboros": "ouroboros · ink & gold",
    "canal_town": "canal town · aerial",
    "butterfly": "butterfly · macro",
    "temple_facade": "temple gate · woodblock",
    "lantern_row": "lantern row · dusk",
    "zebra": "zebra · photo",
    "nautilus": "nautilus · macro",
    "typhoon": "typhoon · satellite",
    "watershed": "watershed · aerial",
    "homing_birds": "homing birds · long exposure",
    "gorge_bridge": "gorge bridge · aerial",
    "hourglass": "hourglass · macro",
    "orchard_anomaly": "orchard anomaly · aerial",
    "red_plum": "one red blossom · photo",
    "impossible_stairs": "endless stairs · photo",
    "drawing_hands": "drawing hands · photo",
    "day_night_birds": "birds interlocked · woodcut",
    "papercut": "papercut & remnant · photo",
    "constellations": "two constellations · night sky",
    "tree_and_roots": "crown & roots · ink wash",
    "silk_band": "silk band · studio",
    "ant_band": "ants on the band · studio",
    "hair_braid": "braid · close-up",
    "interchange": "interchange · aerial",
    "two_ripples": "two ripples · overhead",
    "moire_curtains": "moiré curtains · photo",
    "honeycomb": "honeycomb · macro",
    "salt_pans": "salt pans · aerial",
    "firework": "firework · long exposure",
    "starfish": "starfish · macro",
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
  .toc { font-family: Menlo, monospace; font-size: 12.5px; line-height: 2.1;
         margin-top: 14px; }
  .toc a { text-decoration: none; border: 1px solid var(--line); border-radius: 4px;
           padding: 3px 8px; margin-right: 6px; white-space: nowrap; }
  .toc a:hover { border-color: var(--accent); }
  .fam-badge { display: inline-block; font: 12px/1 Menlo, monospace; color: #fff;
               background: var(--accent); border-radius: 4px; padding: 4px 8px;
               margin-right: 8px; vertical-align: middle; }
  section.sample { margin-top: 72px; }
  h2 { font-size: 25px; margin: 0 0 4px; }
  h2 .zh { color: var(--dim); font-weight: normal; }
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
  .voice p { margin: 8px 0; }
  pre { background: var(--code-bg); border: 1px solid var(--line); border-radius: 6px;
        padding: 14px 16px; overflow-x: auto; font: 13px/1.55 "SF Mono", Menlo, monospace;
        white-space: pre-wrap; }
  .mathv { font-size: 15px; }
  .mathv .lbl { font-family: Menlo, monospace; font-size: 11.5px; color: var(--accent);
                text-transform: uppercase; letter-spacing: 1px; }
  .motif-eq { font-size: 18px; margin: 4px 0 2px; }
  .invariant { color: var(--dim); font-style: italic; margin: 2px 0 16px; }
  .nuisance { font: 12px/1.6 Menlo, monospace; color: var(--dim); margin-top: 8px; }
  .tale { border-left: 3px solid var(--accent); background: #fffdf9;
          padding: 14px 20px; margin: 16px 0 6px; border-radius: 0 6px 6px 0; }
  .tale .t-title { font-weight: bold; margin: 0 0 6px; }
  .tale .t-domain { color: var(--dim); font: 12px/1 Menlo, monospace; font-weight: normal; }
  .tale p.zh { margin: 6px 0; white-space: pre-line; }
  .tale p.en { margin: 6px 0; color: var(--dim); font-style: italic; font-size: 14.5px; }
  .tale .t-why { font: 12px/1.6 Menlo, monospace; color: var(--dim); margin: 8px 0 0; }
  .device { font-size: 14.5px; color: var(--ink); }
  .geb-echo { font-size: 13.5px; color: var(--dim); font-style: italic; margin-top: 10px; }
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
    &nbsp;&rarr;&nbsp; { natural / artistic <b>scenes</b>, <b>story</b> whose telling
    enacts it, <b>math</b> with a real theorem }</div>
  <p class="intro">A benchmark of <b>abstract structures</b>, in the spirit of
  <em>Gödel, Escher, Bach</em>. The unit is a structural <b>motif</b> — @NMOTIFS@ of
  them, from self-reference and strange loops to Möbius bands and interference —
  never any particular graph: node counts, depths and angles are nuisance variables.
  Each motif speaks with several voices. Its <b>mathematics</b> is real: a signature
  equation, a formal definition, a classical home and a true theorem. Its
  <b>story</b> is built the way GEB's dialogues are built — the Crab Canon is a
  palindrome, the Sloth Canon a transformed voice — each tale's <em>telling</em>
  implements a mechanically checkable form device. Its <b>images</b> are natural or
  artistic scenes whose composition <em>is</em> the structure: a vis-graph skeleton
  guides generation, so you look at a landscape and the structure dawns on you.</p>
  <nav class="toc">@TOC@</nav>
</header>
@MOTIFS@
<footer>GEB Benchmark · see a scene, grasp a structure ·
  scenes: gpt-image with programmatic skeletons as references ·
  stories: form-device constrained, dual-screened</footer>
</div>
</body>
</html>
"""

MOTIF_TPL = """<section class="sample" id="motif-@MID@">
  <h2><span class="fam-badge">motif</span>@NAME@ <span class="zh">· @NAME_ZH@</span></h2>
  <p class="motif-eq">$$@EQ@$$</p>
  <p class="invariant">invariant: @INVARIANT@</p>
  <div class="strip">@FIGURES@</div>
  @TALES@
  <details><summary>mathematics voice — definition · home · theorem</summary>
    <div class="voice mathv">
      <p><span class="lbl">定义</span><br>@MATH_FORMAL@</p>
      <p><span class="lbl">归属</span><br>@MATH_HOME@</p>
      <p><span class="lbl">定理</span><br>@MATH_THEOREM@</p>
    </div></details>
  <details><summary>story form device — the telling enacts the structure</summary>
    <div class="voice device"><p>@DEVICE@</p>@GEB_ECHO@</div></details>
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


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def load_stories(name: str) -> dict:
    f = ROOT / "data" / name
    if not f.exists():
        return {}
    by_motif = {}
    for s in json.loads(f.read_text()):
        by_motif.setdefault(s["motif_id"], []).append(s)
    return by_motif


def tale_html(story: dict, badge: str) -> str:
    j = story.get("judge")
    screen = f" · {badge}" if j and j.get("passed") else ""
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


def build_motif(mid: str, v2: dict, pilot: dict) -> str:
    lib = LIBRARY[mid]
    d = MOTIFS_DIR / mid
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

    tales = [tale_html(s, "device ✓ + blind ✓") for s in v2.get(mid, [])]
    tales += [tale_html(s, "screened ✓ (v1 pilot)") for s in pilot.get(mid, [])]

    echo = lib.get("geb_echo", "")
    echo_html = f'<p class="geb-echo">GEB echo — {esc(echo)}</p>' if echo else ""

    schema = {
        "motif_id": mid, "name": lib["name_en"], "name_zh": lib["name_zh"],
        "equation": lib["equation_katex"], "invariant": lib["invariant_zh"],
        "math_home": lib["math_home"],
        "nuisance_variables": ["node count", "depth", "angles", "layout",
                               "scale", "domain", "medium"],
    }
    out = MOTIF_TPL
    for k, v in {
        "@MID@": mid,
        "@NAME@": esc(lib["name_en"].split(" — ")[0]),
        "@NAME_ZH@": esc(lib["name_zh"]),
        "@EQ@": lib["equation_katex"],
        "@INVARIANT@": esc(lib["invariant_zh"]),
        "@FIGURES@": "\n".join(figures),
        "@TALES@": "\n".join(tales),
        "@MATH_FORMAL@": esc(lib["math_formal"]),
        "@MATH_HOME@": esc(lib["math_home"]),
        "@MATH_THEOREM@": esc(lib["math_theorem"]),
        "@DEVICE@": esc(lib["story_form_device"]),
        "@GEB_ECHO@": echo_html,
        "@SCHEMA@": esc(json.dumps(schema, indent=2, ensure_ascii=False)),
        "@NUISANCE@": esc(", ".join(schema["nuisance_variables"])),
    }.items():
        out = out.replace(k, v)
    return out


def main():
    if (SITE / "assets").exists():
        shutil.rmtree(SITE / "assets")
    SITE.mkdir(exist_ok=True)
    v2 = load_stories("stories_v2.json")
    pilot = load_stories("stories_pilot.json")
    order = [mid for mid in MOTIF_ORDER if mid in LIBRARY]
    sections = [build_motif(mid, v2, pilot) for mid in order]
    toc = " ".join(f'<a href="#motif-{mid}">{LIBRARY[mid]["name_zh"]}</a>'
                   for mid in order)
    page = (PAGE.replace("@MOTIFS@", "\n".join(sections))
                .replace("@NMOTIFS@", str(len(order)))
                .replace("@TOC@", toc))
    (SITE / "index.html").write_text(page)
    n_scenes = len(list(MOTIFS_DIR.glob("*/scene_*.png")))
    n_tales = sum(len(v) for v in v2.values()) + sum(len(v) for v in pilot.values())
    print(f"site/index.html built: {len(order)} motifs, {n_scenes} scenes, "
          f"{n_tales} tales")


if __name__ == "__main__":
    main()
