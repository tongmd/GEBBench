#!/usr/bin/env python3
"""GEB benchmark — build the static site (findings-first).

Not a gallery: one "same structure, many voices" showcase strip, then the
findings from the v2 evaluation (1,156 items, 8 models) as the main content,
with the two headline result figures.

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
ANALYSIS = ROOT / "eval" / "analysis_v2"

LIBRARY = {m["id"]: m for m in
           json.loads((ROOT / "data" / "motif_library.json").read_text())}


def esc(s: str) -> str:
    return html.escape(s, quote=False)


# --- showcase: a handful of scenes that make "one structure, many voices"
# land at a glance. motif_id -> [(scene_file_stem, caption), ...] ---
SHOWCASE = [
    ("recursion", [("scene_river_delta", "river delta · aerial"),
                   ("scene_lightning", "lightning · long exposure")]),
    ("self_reference", [("scene_droste_study", "study within study")]),
    ("cycle", [("scene_ouroboros", "ouroboros · ink & gold")]),
    ("mobius", [("scene_silk_band", "silk band · one half-twist")]),
]

PLACEHOLDER = "@BODY@"

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEB Benchmark — See a scene, grasp a structure</title>
<link rel="icon" type="image/svg+xml" href="images/favicon.svg">
<link rel="icon" type="image/png" sizes="96x96" href="images/favicon-96x96.png">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});"></script>
<style>
  :root { --ink:#1c1c1c; --paper:#faf8f4; --line:#e3ded4; --accent:#b3402a;
          --dim:#7a756b; --code-bg:#f1ede4; --card:#fffdf9; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--paper); color:var(--ink);
         font:16px/1.65 Georgia,"Times New Roman",serif; }
  .wrap { max-width:940px; margin:0 auto; padding:0 24px 96px; }
  header { padding:72px 0 20px; border-bottom:1px solid var(--line); }
  h1 { font-size:44px; margin:0 0 8px; letter-spacing:-0.5px; }
  h1 .geb { color:var(--accent); }
  .tagline { font-size:20px; font-style:italic; color:var(--dim); margin:0 0 18px; }
  .intro { max-width:780px; }
  .chain { font-family:"SF Mono",Menlo,monospace; font-size:13.5px;
           background:var(--code-bg); border:1px solid var(--line);
           border-radius:8px; padding:13px 17px; overflow-x:auto; white-space:nowrap;
           margin:16px 0; }
  .chain b { color:var(--accent); }
  .stat-row { display:flex; flex-wrap:wrap; gap:10px; margin:22px 0 4px; }
  .stat { flex:1 1 150px; border:1px solid var(--line); border-radius:8px;
          background:var(--card); padding:12px 16px; }
  .stat .n { font-size:26px; font-weight:bold; color:var(--accent); }
  .stat .l { font:12px/1.4 Menlo,monospace; color:var(--dim); }
  h2 { font-size:27px; margin:64px 0 6px; letter-spacing:-0.3px; }
  h2 .k { color:var(--accent); font:13px/1 Menlo,monospace; text-transform:uppercase;
          letter-spacing:1.5px; display:block; margin-bottom:8px; }
  h3 { font-size:19px; margin:34px 0 4px; }
  p.lead { font-size:17px; color:var(--ink); }
  .finding { border-left:3px solid var(--accent); background:var(--card);
             padding:14px 22px; margin:22px 0; border-radius:0 8px 8px 0; }
  .finding .headline { font-weight:bold; font-size:17px; margin:0 0 6px; }
  .finding p { margin:6px 0; }
  .num { font-family:Menlo,monospace; color:var(--accent); font-size:0.95em; }
  figure { margin:26px 0; }
  figure img { max-width:100%; display:block; border:1px solid var(--line);
               border-radius:8px; background:#fff; margin:0 auto; }
  figcaption { font:13px/1.55 Georgia,serif; color:var(--dim); padding-top:8px;
               text-align:center; font-style:italic; }
  .showcase { display:flex; gap:12px; overflow-x:auto; padding:6px 0 4px; }
  .showcase figure { margin:0; flex:0 0 auto; }
  .showcase img { height:180px; width:auto; border-radius:6px; }
  .showcase figcaption { font:12px/1.4 Menlo,monospace; }
  .sk { opacity:.7; }
  .sk img { height:110px; }
  table { border-collapse:collapse; width:100%; font-size:14px; margin:18px 0;
          font-family:Menlo,monospace; }
  th,td { border-bottom:1px solid var(--line); padding:7px 9px; text-align:right; }
  th:first-child,td:first-child { text-align:left; }
  thead th { color:var(--dim); font-weight:normal; font-size:12px;
             text-transform:uppercase; letter-spacing:.5px; }
  tbody tr:hover { background:var(--card); }
  .hi { color:var(--accent); font-weight:bold; }
  .dim { color:var(--dim); }
  footer { margin-top:80px; padding-top:20px; border-top:1px solid var(--line);
           color:var(--dim); font-size:14px; }
  a { color:var(--accent); }
  code { font-family:Menlo,monospace; font-size:.9em; background:var(--code-bg);
         padding:1px 5px; border-radius:4px; }
</style>
</head>
<body>
<div class="wrap">
""" + PLACEHOLDER + """
<footer>GEB&nbsp;Benchmark · see a scene, grasp a structure ·
  25 motifs &times; 4 voices, 1,156 items · scenes: gpt-image with programmatic
  skeleton references · stories: form-device constrained, dual-screened ·
  code MIT, data CC-BY-4.0</footer>
</div>
</body>
</html>
"""


def copy_showcase() -> str:
    """Copy the handful of showcase scenes and return the strip HTML."""
    dst = SITE / "assets" / "showcase"
    dst.mkdir(parents=True, exist_ok=True)
    figs = []
    for mid, scenes in SHOWCASE:
        skel = sorted((MOTIFS_DIR / mid).glob("skeleton_*.png"))
        if skel:
            shutil.copy2(skel[0], dst / f"{mid}__{skel[0].name}")
            rel = f"assets/showcase/{mid}__{skel[0].name}"
            figs.append(f'<figure class="sk"><img loading="lazy" src="{rel}" '
                        f'alt="{mid} skeleton"><figcaption>{esc(mid)}<br>skeleton'
                        f'</figcaption></figure>')
        for stem, cap in scenes:
            src = MOTIFS_DIR / mid / f"{stem}.png"
            if not src.exists():
                continue
            shutil.copy2(src, dst / f"{mid}__{stem}.png")
            rel = f"assets/showcase/{mid}__{stem}.png"
            figs.append(f'<figure><img loading="lazy" src="{rel}" alt="{cap}">'
                        f'<figcaption>{esc(cap)}</figcaption></figure>')
    return '<div class="showcase">' + "\n".join(figs) + "</div>"


def hero() -> str:
    return f"""<header>
  <h1><span class="geb">GEB</span> Benchmark</h1>
  <p class="tagline">See a scene, grasp a structure.</p>
  <p class="intro">A benchmark of <b>abstract structural motifs</b> in the spirit
  of <em>G&ouml;del, Escher, Bach</em>. The unit of a sample is a structure &mdash;
  self-reference, strange loop, recursion, M&ouml;bius, interference &mdash; not
  any particular graph: node counts, depths and angles are nuisance variables,
  recorded but never scored. Each motif is presented in four <b>voices</b>, and
  the question is always the same: <em>which structure is this?</em></p>
  <div class="chain">motif&nbsp;z &rarr; <b>skeleton</b> (vis-graph)
    &rarr; <b>scene</b> (image; the composition <i>is</i> the structure)
    &rarr; <b>story</b> (the telling enacts it)
    &rarr; <b>theorem</b> (a real, name-masked theorem)</div>
  {copy_showcase()}
  <p class="intro dim" style="font-size:14px">One recursive-branching skeleton
  becomes a river delta and a lightning bolt &mdash; no pixel in common, one
  structure. Same for the others above.</p>
  <div class="stat-row">
    <div class="stat"><div class="n">25&times;4</div>
      <div class="l">motifs &times; voices</div></div>
    <div class="stat"><div class="n">1,156</div>
      <div class="l">items (946 main + 210 adversarial)</div></div>
    <div class="stat"><div class="n">8</div>
      <div class="l">models, 4 vendors</div></div>
    <div class="stat"><div class="n">2</div>
      <div class="l">languages (EN primary, ZH mirror)</div></div>
  </div>
</header>"""


def load(name: str) -> dict:
    return json.loads((ANALYSIS / name).read_text())


def copy_fig(stem: str) -> None:
    dst = SITE / "assets" / "figs"
    dst.mkdir(parents=True, exist_ok=True)
    src = ANALYSIS / f"{stem}.png"
    if src.exists():
        shutil.copy2(src, dst / f"{stem}.png")


def pct(x: float) -> str:
    return f"{x * 100:.0f}"


def findings() -> str:
    g = load("gradient_v2.json")
    m = load("mapping_v2.json")
    c = load("confusion_v2.json")
    copy_fig("gradient_kappa_v2")
    copy_fig("mapping_gap_v2")

    # numbers
    gap = m["matched_arity_gap"]
    g55 = gap["api_gpt-5.5"]
    g52 = gap["api_gpt-5.2"]
    sc = m["scale_interaction"]
    s_story = sc["story2motif"]
    s_map = sc["xv_story_thm"]
    d1 = c["distance_share"]
    xm = c["cross_model"]["api_gpt-5.5__api_gpt-5.2"]
    hardest = c["hardest_bottom5"]

    out = ['<h2><span class="k">what we found</span>Reading structure, and failing to carry it</h2>',
           '<p class="lead">Frontier models can <em>recognize</em> an abstract '
           'structure in any single voice, yet fail &mdash; systematically &mdash; '
           'to <em>carry</em> it from one voice to another. The failure is lawful, '
           'not random.</p>']

    # Finding 1: recognition vs mapping (headline)
    out.append(f"""<h3>1 &nbsp;Recognition is not mapping</h3>
<div class="finding">
  <p class="headline">On the very same scenes, at the very same 4-way odds,
  identifying the structure beats carrying it across voices by
  <span class="num">21&ndash;44 points</span>.</p>
  <p>Our adversarial split asks a model to pick a scene's motif from its three
  nearest confusables (4-way). A separate task asks it to match the same scene
  to the one story (of four) that shares its structure &mdash; same odds, same
  images. Every model does far better at the first:
  GPT-5.5 <span class="num">{pct(g55['acc_family4'])}% &rarr; {pct(g55['acc_crossvoice'])}%</span>,
  GPT-5.2 <span class="num">{pct(g52['acc_family4'])}% &rarr; {pct(g52['acc_crossvoice'])}%</span>
  (McNemar <span class="num">p&lt;10<sup>-8</sup></span>). The bottleneck is not
  fine discrimination; it is cross-voice transport.</p>
</div>
<figure><img src="assets/figs/mapping_gap_v2.png" alt="recognition vs mapping gap">
  <figcaption>Chance-corrected &kappa;: within-family identification (recognition)
  vs cross-voice matching (mapping), on the same scenes.</figcaption></figure>""")

    # Finding 2: scale helps recognition not mapping
    out.append(f"""<h3>2 &nbsp;Scale feeds recognition, not mapping</h3>
<div class="finding">
  <p class="headline">Growing a model 7B&nbsp;&rarr;&nbsp;14B lifts recognition
  by <span class="num">+{pct(s_story['delta'])} points</span> and moves mapping
  by <span class="num">exactly zero</span>.</p>
  <p>In the Qwen2.5 family, story&nbsp;&rarr;&nbsp;motif identification climbs
  <span class="num">{pct(s_story['acc_qwen2.5-7b'])}% &rarr; {pct(s_story['acc_qwen2.5-14b'])}%</span>
  (p&nbsp;=&nbsp;2&times;10<sup>-9</sup>), while story&nbsp;&rarr;&nbsp;theorem
  matching stays pinned at
  <span class="num">{pct(s_map['acc_qwen2.5-7b'])}% &rarr; {pct(s_map['acc_qwen2.5-14b'])}%</span>
  (14 items flip each way; p&nbsp;=&nbsp;1.0). A bigger model reads structure
  better and transports it no better at all.</p>
</div>""")

    # Finding 3: errors live in conceptual geometry
    out.append(f"""<h3>3 &nbsp;Errors live in conceptual geometry</h3>
<div class="finding">
  <p class="headline">When a model is wrong, it is wrong toward the
  formally-nearest motif <span class="num">2.2&times;</span> more often than chance.</p>
  <p>Across <span class="num">{d1['n_errors']:,}</span> identification errors,
  <span class="num">{pct(d1['share']['1'])}%</span> land on a designed
  nearest-neighbor motif (chance <span class="num">{pct(d1['chance_share']['1'])}%</span>;
  permutation p&nbsp;&lt;&nbsp;10<sup>-4</sup>). An <em>independently</em> derived
  graph-edit distance &mdash; read straight off the released skeleton programs,
  never used to build the benchmark &mdash; reproduces the same concentration.
  Three perceptual embeddings (DINOv2, CLIP, SigLIP) do not explain it: errors
  track the designed <em>formal</em> geometry more than any measured perceptual one.</p>
</div>""")

    # Finding 4: gradient + figure
    pf = g["pooled_frontier"]
    _kmap = {"theorem": "theorem2motif", "story": "story2motif",
             "skeleton": "skeleton2motif", "scene": "scene2motif",
             "crossvoice": "crossvoice"}

    def kmean(t):
        return pf[_kmap[t]]["kappa"]
    out.append(f"""<h3>4 &nbsp;A voice gradient</h3>
<div class="finding">
  <p class="headline">Structure is easiest stated as a theorem, hardest carried
  across voices &mdash; a clean ordering once you correct for chance.</p>
  <p>Because the tasks differ in difficulty floor (25-way vs 4-way), we score
  every cell as &kappa;&nbsp;=&nbsp;(acc&nbsp;&minus;&nbsp;chance)/(1&nbsp;&minus;&nbsp;chance).
  Pooled over the three frontier models, &kappa; falls monotonically:
  theorem <span class="num">{kmean('theorem'):.2f}</span> &gt;
  story <span class="num">{kmean('story'):.2f}</span> &gt;
  skeleton <span class="num">{kmean('skeleton'):.2f}</span> &gt;
  scene <span class="num">{kmean('scene'):.2f}</span> &gt;
  cross-voice <span class="num">{kmean('crossvoice'):.2f}</span>. Cross-voice is
  the lowest for every vision model in &gt;99.9% of bootstrap resamples.</p>
</div>
<figure><img src="assets/figs/gradient_kappa_v2.png" alt="voice gradient by kappa">
  <figcaption>Chance-corrected accuracy (&kappa;) by voice, five vision-capable
  models, with bootstrap 95% intervals. Chance sits at &kappa;&nbsp;=&nbsp;0.</figcaption></figure>""")

    # Finding 5: cross-vendor same traps
    out.append(f"""<h3>5 &nbsp;Different models fall into the same traps</h3>
<div class="finding">
  <p class="headline">When two frontier models miss the same item, they pick the
  <em>same</em> wrong motif <span class="num">{pct(xm['same_wrong_share'])}%</span> of the time.</p>
  <p>Two systems from different vendors, with different training data, converge
  on the same conceptual confusions on the same stimuli. The hard boundaries are
  properties of the structures, not quirks of one model &mdash; which is what a
  benchmark should measure.</p>
</div>""")

    # Finding 6: bilingual
    out.append("""<h3>6 &nbsp;Structure survives translation</h3>
<div class="finding">
  <p class="headline">The same stories in English and Chinese score
  <span class="num">identically</span> (77.3% each), and when models err in both
  languages they err toward the same motif far above chance.</p>
  <p>What the benchmark probes is structural, not lexical: the shape, not the
  words that carry it. This is why the dataset is English-primary with a Chinese
  mirror rather than tied to either language.</p>
</div>""")

    # hardest motifs strip
    hard_items = ", ".join(
        f'<span class="num">{esc(h["motif"].replace("_", " "))}</span> ({pct(h["recall"])}%)'
        for h in hardest[:5])
    out.append(f"""<h2><span class="k">the hard edge</span>Where every model struggles</h2>
<p>Pooled recall is lowest on the motifs at the conceptual core of GEB &mdash;
{hard_items}. Strikingly, these are also the structures the image generator
itself could least often render faithfully (rank correlation
&rho;&nbsp;=&nbsp;&minus;0.48): the strange loops and figure&ndash;grounds that
are hard to <em>draw</em> are hard to <em>read</em>.</p>""")

    return "\n".join(out)


def closing() -> str:
    return """<h2><span class="k">get it</span>Dataset &amp; code</h2>
<p>All 1,156 items, the 25-motif library (real theorems + form devices), the
generation and evaluation harness, and the analysis scripts are open. Code is
MIT; the dataset is CC-BY-4.0.</p>
<p><a href="https://github.com/tongmd/GEB-benchmark">github.com/tongmd/GEB-benchmark</a></p>"""


def main():
    if (SITE / "assets").exists():
        shutil.rmtree(SITE / "assets")
    SITE.mkdir(exist_ok=True)
    body = "\n".join([hero(), findings(), closing()])
    page = PAGE.replace(PLACEHOLDER, body)
    (SITE / "index.html").write_text(page)
    kb = len(page) // 1024
    print(f"site/index.html built: findings-first, {kb} KB, "
          f"showcase + 6 findings + 2 figures")


if __name__ == "__main__":
    main()
