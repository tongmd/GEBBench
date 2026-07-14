# GEB Benchmark

**One structure, six voices.** Every sample starts from a single latent schema
`z` and is rendered — in parallel, never by chained translation — into six
representations:

```
z  →  { Story, Mathematics, Program, Drawing Program, Canonical Image, Stylized Image }
```

Inspired by *Gödel, Escher, Bach*: the same abstract structure expressed as a
narrative, a formal statement, executable code, graphics code, a picture, and
a stylized picture. The schema is the only ground truth; every voice is
generated from it and machine-checkable against it.

## V1 families

| Family | Structure | Example properties |
| --- | --- | --- |
| A. Graph Worlds | topology | reachability, shortest path, cycles, dependency order |
| B. Recursive Gardens | hierarchy | recursion depth, branching factor, self-similarity |
| C. Symmetry Labs | geometry | rotation order, chirality, group composition |

Full design: [`proposal/design-v1.md`](proposal/design-v1.md).

## Examples

```bash
python3 generate_examples.py            # 5 samples -> examples/
python3 gen_stylized.py                 # gpt-image stylized versions
```

Each `examples/<sample_id>/` contains:

| File | Voice |
| --- | --- |
| `schema.json` | latent schema `z` (ground truth, incl. answer) |
| `story.txt` | story voice, 2 domains |
| `math.md` | mathematics voice |
| `program.py` | program voice — standalone, self-checking |
| `drawing.json` | drawing-program voice (renderer input) |
| `canonical.svg/.png` | canonical image |
| `style_blueprint`, `style_pencil` | stylized, structure-preserving **by construction** (renderer parameters) |
| `style_gpt_*` | stylized via gpt-image (`images/edits` on the canonical image) |
| `signature.json` | machine-checkable structure signature |

Every generated `program.py` is executed at generation time and its result is
asserted against the schema answer.

## Homepage

`site/` — static preview page (Cloudflare Pages).
