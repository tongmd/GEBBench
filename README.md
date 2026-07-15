# GEB Benchmark

**See a scene, grasp a structure.** A benchmark of *abstract structures* in the
spirit of *Gödel, Escher, Bach*. The unit of a sample is a structural **motif**
— self-reference, recursion, nesting, cycle — not any particular graph: node
counts, depths and angles are **nuisance variables**, recorded but never scored.

```
motif z → skeleton (programmatic vis-graph rendering, several embeddings)
        → natural/artistic scenes (gpt-image, skeleton as reference —
          the composition IS the structure; not a style transfer)
        → story voice (a tale shaped like the motif, e.g. 从前有座山…)
        → math voice (the motif's equation, e.g. s = T(s), f(x)=x)
```

Same skeleton, different worlds: one recursive-branching skeleton becomes a
river delta and a lightning bolt — no pixel in common, one structure.

## Motifs (v0)

| Motif | Equation | Scenes |
| --- | --- | --- |
| self-reference | `s = T(s)`, `f(x) = x` | 山中庙 (ink wash) · Droste study |
| recursion | `T = Node(T,…,T)` | river delta · lightning · winter tree |
| nesting | `S_λ(X) = X` | walled city · pond ripples |
| cycle | `v ⇝ v` | ouroboros · canal town |

## Generate

```bash
python3 gen_motifs.py     # motifs/: skeletons (SVG→PNG) + scenes (gpt-image)
python3 build_site.py     # site/: static preview page
```

`motifs/<motif_id>/`: `schema.json` (motif, equation, invariant, nuisance),
`story.txt`, `math.md`, `skeleton_*.svg/png`, `scene_*.png`.

Design docs: [`proposal/design-v1.md`](proposal/design-v1.md) (superseded
diagram-family draft, kept for history) ·
[`proposal/scaling.md`](proposal/scaling.md) (how to scale motif items).

## Homepage

`site/` → <https://geb.stonezhang.com> (Cloudflare Pages).
