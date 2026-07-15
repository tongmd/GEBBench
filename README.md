# GEB Benchmark

**See a scene, grasp a structure.** A benchmark of *abstract structures* in the
spirit of *Gödel, Escher, Bach*. The unit of a sample is a structural **motif**
— self-reference, strange loop, recursion, … Möbius, interference — not any
particular graph: node counts, depths and angles are **nuisance variables**,
recorded but never scored.

```
motif z → skeleton (programmatic vis-graph rendering, several embeddings)
        → natural/artistic scenes (gpt-image, skeleton as reference —
          the composition IS the structure; not a style transfer)
        → story voice (GEB-style FORM DEVICE: the telling itself enacts the
          structure, mechanically checkable — Crab Canon = palindrome)
        → math voice (signature equation + formal definition + classical
          home + a true theorem, adversarially verified)
```

Same skeleton, different worlds: one recursive-branching skeleton becomes a
river delta and a lightning bolt — no pixel in common, one structure.

## Motif library (v2 — 25 motifs)

`data/motif_library.json` — designed by a propose→curate→verify agent
workflow; every entry's mathematics and pairwise distinguishability was
adversarially reviewed (`data/motif_library_review_notes.json`).

| Motif | Signature | Story form device (mechanically checked) |
| --- | --- | --- |
| self_reference 自指 | `⊢ ψ ↔ φ(⌜ψ⌝)` | title asserts the acrostic; sentence initials spell the title |
| strange_loop 怪圈 | layered graph + one back edge | 3 narration levels; innermost hero = outer narrator; last line = first |
| recursion 递归 | `T ::= • \| Node(T,T)`, Catalan | 话说/且说回来 bracket pairs, 0-or-2 children, base-case line |
| nesting 嵌套 | Cantor nested intervals | opening lines only, outer names never return, ends innermost |
| cycle 因果闭环 | `⟨r \| rⁿ=e⟩ ≅ Z/nZ` | 于是-chained segments; last result = first line verbatim |
| mobius 莫比乌斯带 | `(0,t) ∼ (1, 1−t)` | two laps, direction words flipped by an involution; coda = line 1 |
| spiral 螺旋副歌 | `r = a e^{bθ}` | refrain ×4, segment lengths 6/12/24/48 (ratio 2) |
| symmetry 回文对称 | `w = w^R`, `D_n` | Crab Canon: 2k+1 sentences, lower half = upper half reversed |
| figure_ground 图底互换 | `∃A (A ∈ Σ⁰₁ ∧ Ā ∉ Σ⁰₁)` | same events retold with subject/object roles complemented |
| isomorphism 同构 | `φ(a·b) = φ(a)∘φ(b)` | two worlds + a bijective dictionary; line k ↔ line k |
| alternation 严格交替 | `χ(G) ≤ 2 ⇔` no odd cycle | script dialogue, speakers strictly alternate |
| braiding 辫结 | `σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁` | seat order announced per scene; swaps compose to final order |
| interference 干涉 | beat / `lcm(p,q)` | period-3 and period-4 sentence rules; double hits at 12 & 24 |
| tiling 平移密铺 | `T + λ = T`, 17 wallpaper groups | one syntactic template ×8, only three slots vary |
| symmetry_breaking 对称破缺 | Poincaré–Hopf `Σ ind = χ(M)` | 12 sentences obey one rule; exactly sentence 7 breaks it |
| convergence 万流归一 | in-tree, Newman's lemma | 3 disjoint threads; final line unites all names at a new place |
| bottleneck 咽喉之点 | Menger: min cut = max disjoint paths | 3 journeys, each passes 独木桥 exactly once mid-way |
| hub 星型枢纽 | `G ≅ K_{1,n}` | villagers never speak directly; every quote pair contains the miller |
| diagonalization 对角线遁逸 | Cantor: `d(n) = 1 − f(n)(n)` | n tellers × n binary traits; the traveler's trait k flips teller k's trait k |
| infinite_descent 无穷递降 | Fermat descent | 6 citations, strictly decreasing numbers; verdict: the first claim never existed |
| coprime_orbit 互素跳环 | `⟨k⟩ = Z_n ⇔ gcd(k,n)=1` | lamp at stride 3 around 8: roll-call 0,3,6,1,4,7,2,5; stride-2 control strands half |
| pigeonhole 鸽笼必撞 | `\|A\| > \|B\| ⇒` collision | 13 guests, 12 slots; the elder asserts a collision before any check |
| projective_duality 点线对偶 | `σ² = id`, `⊢P ⇔ ⊢P^σ` | told twice; the opening dictionary maps telling 2 back to telling 1 verbatim |
| contraction_fixed_point 收缩不动点 | Banach fixed point | chant halves each round (17→9→5→3→2→1) to a word that maps to itself |
| aperiodic_tiling 非周期密铺 | Penrose; fat/thin → φ | 21 sentences typed A/B by the Fibonacci substitution A→AB, B→A |

## Evaluation

```bash
python3 eval/build_items.py   # eval/items.json: 165 judge-free MC items
                              #   (scene/skeleton/story/theorem→motif, crossvoice)
python3 eval/run_eval.py --base <openai-compatible> --model <id> --tag <tag> [--vision]
python3 eval/analyze.py       # paper/tables/{main_results,confusion}.tex
```

## Generate

```bash
python3 gen_motifs.py     # motifs/: skeletons (SVG→PNG) + scenes (gpt-image)
python3 gen_stories.py    # data/stories_v2.json: form-device stories,
                          #   dual-screened (device judge + blind motif judge)
python3 build_site.py     # site/: static preview page
```

`motifs/<motif_id>/`: `schema.json`, `math.md` (definition · home · theorem),
`skeleton_*.svg/png`, `scene_*.png`.

Design docs: [`proposal/design-v1.md`](proposal/design-v1.md) (superseded
diagram-family draft, kept for history) ·
[`proposal/scaling.md`](proposal/scaling.md) (how to scale motif items).

## Homepage

`site/` → <https://geb.stonezhang.com> (Cloudflare Pages).
