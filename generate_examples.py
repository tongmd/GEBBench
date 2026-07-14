#!/usr/bin/env python3
"""GEB benchmark — example generator (v0).

One latent schema -> six voices, generated in parallel from the schema (hub-and-spokes):

    schema.json        latent schema z (ground truth, incl. answer)
    story.txt          Story voice (2 domains, same structure)
    math.md            Mathematics voice (statement + solution)
    program.py         Program voice (standalone, self-checking)
    drawing.json       Drawing-program voice (renderer input)
    canonical.svg      Canonical image
    style_*.svg        Stylized images (structure-preserving by construction:
                       style = renderer parameters, same primitives)
    signature.json     Structure signature (machine-checkable)

Families:
    A  graph_world       reachability (yes / no), cycle detection
    B  recursive_garden  binary recursive tree
    C  symmetry_lab      cyclic group action (rotation order)

Every generated program.py is executed and its result asserted against the
schema answer before the sample is written (executable ground truth).

Usage:  python3 generate_examples.py [outdir]      (default: ./examples)
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import deque
from pathlib import Path

CANVAS_W, CANVAS_H = 768, 512

# ---------------------------------------------------------------------------
# styles: a style is just a renderer parameter set -> structure preserved
# by construction (same primitives, same coordinates, same topology)
# ---------------------------------------------------------------------------

STYLES = {
    "canonical": dict(bg="#ffffff", ink="#1a1a1a", node_fill="#f4f4f4",
                      accent="#c0392b", faint="#9a9a9a", text="#1a1a1a",
                      jitter=0.0, grid=False),
    "blueprint": dict(bg="#123a6d", ink="#e8f0fa", node_fill="#123a6d",
                      accent="#ffd166", faint="#4a6fa5", text="#e8f0fa",
                      jitter=0.0, grid=True, grid_color="#2a5493"),
    "pencil":    dict(bg="#faf6ee", ink="#4a4a4a", node_fill="#faf6ee",
                      accent="#8c4646", faint="#b0a898", text="#4a4a4a",
                      jitter=1.7, grid=False),
}

# per-recursion-level colors, canonical style only
LEVEL_COLORS = ["#5d4023", "#7a6a2f", "#4f7d33", "#2f9e44", "#37b26b"]


class SVG:
    """Tiny SVG builder with optional hand-drawn jitter."""

    def __init__(self, w: int, h: int, style: dict, seed: int = 0):
        self.w, self.h, self.s = w, h, style
        self.parts: list[str] = []
        self.defs: list[str] = []
        self._markers: dict[str, str] = {}
        self.rng = random.Random(seed)
        if style.get("grid"):
            self._grid()

    # -- primitives ---------------------------------------------------------

    def _grid(self, step: int = 48):
        c = self.s["grid_color"]
        for x in range(step, self.w, step):
            self.parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{self.h}" '
                              f'stroke="{c}" stroke-width="1"/>')
        for y in range(step, self.h, step):
            self.parts.append(f'<line x1="0" y1="{y}" x2="{self.w}" y2="{y}" '
                              f'stroke="{c}" stroke-width="1"/>')

    def _marker(self, color: str) -> str:
        if color not in self._markers:
            mid = f"arrow{len(self._markers)}"
            self._markers[color] = mid
            self.defs.append(
                f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>')
        return self._markers[color]

    def _path_d(self, x1, y1, x2, y2) -> str:
        j = self.s.get("jitter", 0.0)
        d = f"M {x1:.1f} {y1:.1f}"
        if j:
            for t in (0.33, 0.66):
                mx, my = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                d += (f" L {mx + self.rng.uniform(-j, j):.1f}"
                      f" {my + self.rng.uniform(-j, j):.1f}")
        return d + f" L {x2:.1f} {y2:.1f}"

    def line(self, x1, y1, x2, y2, color=None, width=2.0, arrow=False, dash=None):
        color = color or self.s["ink"]
        attrs = f'stroke="{color}" stroke-width="{width}" fill="none" stroke-linecap="round"'
        if dash:
            attrs += f' stroke-dasharray="{dash}"'
        if arrow:
            attrs += f' marker-end="url(#{self._marker(color)})"'
        self.parts.append(f'<path d="{self._path_d(x1, y1, x2, y2)}" {attrs}/>')

    def circle(self, x, y, r, fill=None, stroke=None, width=2.0, dash=None):
        fill = fill or self.s["node_fill"]
        stroke = stroke or self.s["ink"]
        attrs = f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"'
        if dash:
            attrs += f' stroke-dasharray="{dash}"'
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {attrs}/>')

    def polygon(self, pts, fill="none", stroke=None, width=2.0):
        stroke = stroke or self.s["ink"]
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.parts.append(f'<polygon points="{p}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>')

    def text(self, x, y, s, size=16, color=None, anchor="middle", family="Menlo, monospace"):
        color = color or self.s["text"]
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                          f'font-family="{family}" fill="{color}" '
                          f'text-anchor="{anchor}" dominant-baseline="central">{s}</text>')

    def doc(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">'
                f'<rect width="100%" height="100%" fill="{self.s["bg"]}"/>'
                f'<defs>{"".join(self.defs)}</defs>'
                + "".join(self.parts) + "</svg>")


# ---------------------------------------------------------------------------
# graph algorithms (generator-side ground truth)
# ---------------------------------------------------------------------------

def bfs_path(adj: dict, src: str, tgt: str):
    """Shortest path src->tgt as a node list, or None."""
    parent, queue, seen = {src: None}, deque([src]), {src}
    while queue:
        u = queue.popleft()
        if u == tgt:
            path = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                queue.append(v)
    return None


def find_cycle(adj: dict, nodes: list):
    """One directed cycle as a node list (closed), or None."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in nodes}
    stack: list = []

    def dfs(u):
        color[u] = GRAY
        stack.append(u)
        for v in adj.get(u, ()):
            if color[v] == GRAY:
                return stack[stack.index(v):] + [v]
            if color[v] == WHITE:
                found = dfs(v)
                if found:
                    return found
        color[u] = BLACK
        stack.pop()
        return None

    for v in nodes:
        if color[v] == WHITE:
            c = dfs(v)
            if c:
                return c
    return None


def random_digraph(rng: random.Random, n: int, m: int):
    nodes = [chr(ord("A") + i) for i in range(n)]
    pool = [(u, v) for u in nodes for v in nodes if u != v]
    edges = sorted(rng.sample(pool, m))
    adj: dict = {v: [] for v in nodes}
    for u, v in edges:
        adj[u].append(v)
    return nodes, edges, adj


def graph_signature(nodes, edges, adj) -> dict:
    indeg = {v: 0 for v in nodes}
    outdeg = {v: 0 for v in nodes}
    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "directed_edges": [list(e) for e in edges],
        "degree_sequence": {"in": [indeg[v] for v in nodes],
                            "out": [outdeg[v] for v in nodes]},
        "has_cycle": find_cycle(adj, nodes) is not None,
    }


# ---------------------------------------------------------------------------
# Family A — Graph Worlds
# ---------------------------------------------------------------------------

GRAPH_STORY = {
    "city-roads": ("There is a one-way road from city {u} to city {v}.",
                   {"reachability": "A traveler starts in city {s}. Can they reach city {t} "
                                    "by following the roads?",
                    "cycle": "Is there a route that leaves some city and returns to it "
                             "by following the roads?"}),
    "messaging": ("User {u} can forward messages to user {v}.",
                  {"reachability": "A message originates with user {s}. Can it eventually "
                                   "reach user {t}?",
                   "cycle": "Can a message ever return to a user who already forwarded it?"}),
    "software":  ("Module {u} imports module {v}.",
                  {"cycle": "Is there a circular import among these modules?"}),
}


def render_graph_svg(drawing: dict, style_name: str, seed: int) -> str:
    s = STYLES[style_name]
    svg = SVG(CANVAS_W, CANVAS_H, s, seed=seed)
    pos = {n["id"]: (n["x"], n["y"]) for n in drawing["nodes"]}
    r = 22
    for e in drawing["edges"]:
        (x1, y1), (x2, y2) = pos[e["source"]], pos[e["target"]]
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        svg.line(x1 + ux * r, y1 + uy * r,
                 x2 - ux * (r + 6), y2 - uy * (r + 6),
                 width=2.2, arrow=True)
    for n in drawing["nodes"]:
        accent = n.get("role") in ("start", "goal")
        svg.circle(n["x"], n["y"], r,
                   stroke=s["accent"] if accent else s["ink"],
                   width=3.0 if accent else 2.2)
        svg.text(n["x"], n["y"], n["id"], size=18)
    return svg.doc()


def gen_graph_sample(sample_id: str, task: str, want: bool, seed: int) -> dict:
    rng = random.Random(seed)
    n, m = 6, 8
    while True:
        nodes, edges, adj = random_digraph(rng, n, m)
        cyc = find_cycle(adj, nodes)
        if task == "cycle":
            if (cyc is not None) == want and (cyc is None or len(cyc) >= 4):
                src = tgt = None
                answer = {"has_cycle": want,
                          "witness_cycle": cyc if want else None}
                break
        else:  # reachability
            src, tgt = "A", nodes[-1]
            path = bfs_path(adj, src, tgt)
            ok = (path is not None) == want
            nontrivial = (path is None) or (len(path) >= 4)
            live = all(len(adj[v]) > 0 or any(v == b for _, b in edges) for v in nodes)
            if ok and nontrivial and live:
                answer = {"reachable": want,
                          "witness_path": path if want else None}
                break

    # --- schema ---
    schema = {
        "sample_id": sample_id,
        "family": "graph_world",
        "task_type": task,
        "directed": True,
        "nodes": [{"id": v,
                   "role": ("start" if v == src else
                            "goal" if v == tgt else "ordinary")}
                  for v in nodes],
        "edges": [list(e) for e in edges],
        "query": ({"source": src, "target": tgt} if task == "reachability"
                  else {"property": "has_cycle"}),
        "answer": answer,
    }
    signature = graph_signature(nodes, edges, adj)
    schema["structural_properties"] = {
        "num_nodes": len(nodes), "num_edges": len(edges),
        "has_cycle": signature["has_cycle"],
    }
    if task == "reachability":
        p = answer["witness_path"]
        signature["shortest_path"] = {f"{src}_to_{tgt}": (len(p) - 1 if p else None)}

    # --- story voice (edge sentences shuffled: mention order carries nothing) ---
    domains = ["city-roads", "messaging"] if task == "reachability" else ["software", "messaging"]
    blocks = []
    for dom in domains:
        tmpl, queries = GRAPH_STORY[dom]
        sentences = [tmpl.format(u=u, v=v) for u, v in edges]
        rng.shuffle(sentences)
        q = queries[task].format(s=src, t=tgt)
        blocks.append(f"[domain: {dom}]\n" + " ".join(sentences) + f"\n\nQuestion: {q}\n")
    story = "\n".join(blocks)

    # --- math voice ---
    idx = {v: i for i, v in enumerate(nodes)}
    M = [[0] * n for _ in range(n)]
    for u, v in edges:
        M[idx[u]][idx[v]] = 1
    rows = " \\\\\n".join(" & ".join(str(x) for x in row) for row in M)
    edge_set = ", ".join(f"({u},{v})" for u, v in edges)
    lines = [f"# Mathematics voice — {sample_id}", "",
             f"Node set $V = \\{{{', '.join(nodes)}\\}}$, edge set",
             f"$$E = \\{{{edge_set}\\}}.$$",
             f"Adjacency matrix (rows/columns ordered ${', '.join(nodes)}$):",
             f"$$M = \\begin{{bmatrix}}\n{rows}\n\\end{{bmatrix}}.$$", ""]
    if task == "reachability":
        lines += [f"**Question.** Does there exist $k \\ge 1$ with "
                  f"$(M^k)_{{{src},{tgt}}} > 0$?", ""]
        if answer["reachable"]:
            p = answer["witness_path"]
            k = len(p) - 1
            lines += [f"**Solution.** Yes; the minimal such $k$ is ${k}$, witnessed by",
                      f"$${' \\to '.join(p)}.$$"]
        else:
            lines += [f"**Solution.** No; $(M^k)_{{{src},{tgt}}} = 0$ for all $k \\ge 1$ "
                      f"(no directed path from ${src}$ to ${tgt}$)."]
    else:
        lines += ["**Question.** Does there exist $k \\ge 1$ with "
                  "$\\operatorname{tr}(M^k) > 0$?", ""]
        if answer["has_cycle"]:
            c = answer["witness_cycle"]
            lines += [f"**Solution.** Yes; a witness cycle is $${' \\to '.join(c)}.$$"]
        else:
            lines += ["**Solution.** No; the graph is a DAG, so every $M^k$ is nilpotent."]
    math_md = "\n".join(lines) + "\n"

    # --- program voice ---
    graph_repr = "{\n" + "".join(f'    "{v}": {adj[v]!r},\n' for v in nodes) + "}"
    if task == "reachability":
        program = PROGRAM_REACH
        subs = {"@SID@": sample_id, "@GRAPH@": graph_repr,
                "@SRC@": repr(src), "@TGT@": repr(tgt),
                "@EXPECTED@": repr(answer["reachable"])}
    else:
        program = PROGRAM_CYCLE
        subs = {"@SID@": sample_id, "@GRAPH@": graph_repr,
                "@EXPECTED@": repr(answer["has_cycle"])}
    for k, v in subs.items():
        program = program.replace(k, v)

    # --- drawing voice: nodes on a circle, order shuffled (layout carries nothing) ---
    order = nodes[:]
    rng.shuffle(order)
    cx, cy, rad = CANVAS_W / 2, CANVAS_H / 2, 190
    drawing = {
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "nodes": [{"id": v,
                   "x": round(cx + rad * math.cos(2 * math.pi * i / n - math.pi / 2), 1),
                   "y": round(cy + rad * math.sin(2 * math.pi * i / n - math.pi / 2), 1),
                   "shape": "circle",
                   "role": ("start" if v == src else "goal" if v == tgt else "ordinary")}
                  for i, v in enumerate(order)],
        "edges": [{"source": u, "target": v, "directed": True} for u, v in edges],
    }

    return dict(schema=schema, signature=signature, story=story, math=math_md,
                program=program, drawing=drawing,
                render=lambda style: render_graph_svg(drawing, style, seed))


PROGRAM_REACH = '''"""GEB benchmark — program voice. sample: @SID@ (graph_world / reachability)"""

from collections import deque

GRAPH = @GRAPH@

SOURCE, TARGET = @SRC@, @TGT@
EXPECTED = @EXPECTED@


def reachable(graph, start, target):
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for nxt in graph.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


if __name__ == "__main__":
    result = reachable(GRAPH, SOURCE, TARGET)
    assert result == EXPECTED, (result, EXPECTED)
    print(f"reachable({SOURCE} -> {TARGET}) = {result}  [matches schema]")
'''

PROGRAM_CYCLE = '''"""GEB benchmark — program voice. sample: @SID@ (graph_world / cycle detection)"""

GRAPH = @GRAPH@

EXPECTED = @EXPECTED@


def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}

    def dfs(u):
        color[u] = GRAY
        for v in graph.get(u, ()):
            if color[v] == GRAY or (color[v] == WHITE and dfs(v)):
                return True
        color[u] = BLACK
        return False

    return any(color[v] == WHITE and dfs(v) for v in list(graph))


if __name__ == "__main__":
    result = has_cycle(GRAPH)
    assert result == EXPECTED, (result, EXPECTED)
    print(f"has_cycle = {result}  [matches schema]")
'''


# ---------------------------------------------------------------------------
# Family B — Recursive Gardens (binary recursive tree)
# ---------------------------------------------------------------------------

def tree_segments(depth: int, angle: float, scale: float, length: float):
    """Recursive binary tree as [(x1,y1,x2,y2,level)] with root at origin, growing up."""
    segs = []

    def branch(x, y, heading, ln, level):
        x2 = x + ln * math.cos(math.radians(heading))
        y2 = y + ln * math.sin(math.radians(heading))
        segs.append((x, y, x2, y2, level))
        if level < depth:
            branch(x2, y2, heading - angle, ln * scale, level + 1)
            branch(x2, y2, heading + angle, ln * scale, level + 1)

    branch(0.0, 0.0, -90.0, length, 0)
    return segs


def render_tree_svg(drawing: dict, style_name: str, seed: int) -> str:
    s = STYLES[style_name]
    svg = SVG(CANVAS_W, CANVAS_H, s, seed=seed)
    rec = drawing["recursion"]
    segs = tree_segments(rec["depth"], rec["children"][1]["angle_delta"],
                         rec["children"][0]["scale"], drawing["root"]["length"])
    x0, y0 = drawing["root"]["x"], drawing["root"]["y"]
    for x1, y1, x2, y2, level in segs:
        color = LEVEL_COLORS[level] if style_name == "canonical" else s["ink"]
        svg.line(x0 + x1, y0 + y1, x0 + x2, y0 + y2,
                 color=color, width=max(1.8, 7.0 - 1.6 * level))
    return svg.doc()


def gen_tree_sample(sample_id: str, depth: int, seed: int) -> dict:
    angle, scale, length = 34, 0.62, 175
    leaves, total = 2 ** depth, 2 ** (depth + 1) - 1
    schema = {
        "sample_id": sample_id,
        "family": "recursive_garden",
        "task_type": "binary_tree",
        "depth": depth,
        "branching_factor": 2,
        "parameters": {"angle": angle, "scale_ratio": scale, "initial_length": length},
        "query": {"property": "leaf_count"},
        "answer": {"leaf_count": leaves},
        "structural_properties": {"recursion_depth": depth, "branching_factor": 2,
                                  "leaf_count": leaves, "node_count": total,
                                  "self_similar": True, "bilateral_symmetry": True},
    }
    signature = {"depth": depth, "branching_factor": 2, "node_count": total,
                 "leaf_count": leaves, "angle_sequence": [-angle, angle],
                 "scale_ratio": scale, "bilateral_symmetry": True}

    story = (
        "[domain: plant-growth]\n"
        "A plant begins as a single stem. Each season, every growing tip sprouts exactly "
        "two new shoots — one angled to the left and one to the right — each shorter than "
        f"the branch it grew from. This repeats for {depth} seasons.\n\n"
        f"Question: after {depth} seasons, how many growing tips does the plant have?\n"
        "\n[domain: task-decomposition]\n"
        "A project starts as one task. Every task that is still too large is split into "
        "exactly two smaller subtasks, and the splitting is applied again to each new "
        f"subtask, {depth} times in total.\n\n"
        f"Question: after {depth} rounds of splitting, how many smallest-level subtasks exist?\n")

    math_md = (
        f"# Mathematics voice — {sample_id}\n\n"
        "Recursive definition:\n"
        "$$T_0 = \\operatorname{Leaf}, \\qquad "
        "T_d = \\operatorname{Node}(T_{d-1}, T_{d-1}).$$\n"
        f"Leaf count: $L_d = 2^d$. Total node count: $N_d = 2^{{d+1}} - 1$.\n"
        f"Branch length recurrence with ratio $r = {scale}$: "
        "$\\ell_{d+1} = r\\,\\ell_d$.\n\n"
        f"**Question.** $L_{{{depth}}} = ?$\n\n"
        f"**Solution.** $L_{{{depth}}} = 2^{{{depth}}} = {leaves}$ "
        f"(and $N_{{{depth}}} = {total}$).\n")

    program = PROGRAM_TREE.replace("@SID@", sample_id) \
                          .replace("@DEPTH@", str(depth)) \
                          .replace("@EXPECTED@", str(leaves))

    drawing = {
        "primitive": "branch",
        "root": {"x": CANVAS_W / 2, "y": CANVAS_H - 30, "angle": -90, "length": length},
        "recursion": {"depth": depth,
                      "children": [{"angle_delta": -angle, "scale": scale},
                                   {"angle_delta": angle, "scale": scale}]},
    }

    return dict(schema=schema, signature=signature, story=story, math=math_md,
                program=program, drawing=drawing,
                render=lambda style: render_tree_svg(drawing, style, seed))


PROGRAM_TREE = '''"""GEB benchmark — program voice. sample: @SID@ (recursive_garden / binary_tree)"""

from dataclasses import dataclass

DEPTH = @DEPTH@
EXPECTED_LEAVES = @EXPECTED@


@dataclass
class Node:
    left: "Node | None" = None
    right: "Node | None" = None


def build_tree(depth: int) -> Node:
    if depth == 0:
        return Node()
    return Node(left=build_tree(depth - 1), right=build_tree(depth - 1))


def count_leaves(node: Node) -> int:
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


if __name__ == "__main__":
    result = count_leaves(build_tree(DEPTH))
    assert result == EXPECTED_LEAVES, (result, EXPECTED_LEAVES)
    print(f"leaf_count(depth={DEPTH}) = {result}  [matches schema]")
'''


# ---------------------------------------------------------------------------
# Family C — Symmetry Labs (cyclic group action, rotation order)
# ---------------------------------------------------------------------------

BASE_SHAPE = [(0, 0), (2, 0), (2, -1), (4, 1), (2, 3), (2, 2), (0, 2)]  # asymmetric arrow


def rotate_pts(pts, angle_deg, center=(0.0, 0.0)):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    cx, cy = center
    return [(cx + ca * (x - cx) - sa * (y - cy),
             cy + sa * (x - cx) + ca * (y - cy)) for x, y in pts]


def render_symmetry_svg(drawing: dict, style_name: str, seed: int) -> str:
    s = STYLES[style_name]
    size = 768
    svg = SVG(size, size, s, seed=seed)
    cx, cy = drawing["canvas_center"]
    orbit_r = 175

    svg.circle(cx, cy, orbit_r, fill="none", stroke=s["faint"], width=1.5, dash="6 6")
    svg.line(cx - 14, cy, cx + 14, cy, color=s["faint"], width=2)
    svg.line(cx, cy - 14, cx, cy + 14, color=s["faint"], width=2)

    # base shape scaled to pixels, centroid placed on the orbit circle at angle 0
    scale = 30
    pts = [(x * scale, y * scale) for x, y in drawing["base_primitive"]["points"]]
    gx = sum(p[0] for p in pts) / len(pts)
    gy = sum(p[1] for p in pts) / len(pts)
    pts = [(x - gx + cx + orbit_r, y - gy + cy) for x, y in pts]

    for t in drawing["transform_set"]:
        ang = t["angle"]
        rot = rotate_pts(pts, ang, center=(cx, cy))
        base = ang == 0
        svg.polygon(rot, fill=s["accent"] if base else "none",
                    stroke=s["accent"] if base else s["ink"],
                    width=3.0 if base else 2.2)
        lx, ly = rotate_pts([(cx + orbit_r + 98, cy)], ang, center=(cx, cy))[0]
        svg.text(lx, ly, f"{ang}&#176;", size=15, color=s["faint"])
    return svg.doc()


def gen_symmetry_sample(sample_id: str, order: int, seed: int) -> dict:
    angle = 360 // order
    schema = {
        "sample_id": sample_id,
        "family": "symmetry_lab",
        "task_type": "rotation_order",
        "base_shape": {"type": "asymmetric_arrow", "points": [list(p) for p in BASE_SHAPE]},
        "group_action": {"type": "rotation", "angle_degrees": angle, "center": [0, 0]},
        "repetitions": order,
        "query": {"property": "rotation_order"},
        "answer": {"rotation_order": order},
        "structural_properties": {"rotation_order": order,
                                  "reflection_symmetric": False,
                                  "chirality_preserved": True},
    }
    signature = {"group_type": "cyclic", "group_order": order,
                 "transformations": [f"rotate_{angle * k}" for k in range(order)],
                 "reflection_axes": [], "chirality": "preserved", "orbit_size": order}

    story = (
        "[domain: mechanism]\n"
        "A mechanical pointer is mounted on a central pivot. Each activation of the "
        f"mechanism turns the pointer by exactly {angle} degrees about the pivot, and the "
        "pointer itself is a rigid asymmetric piece.\n\n"
        "Question: what is the smallest number of activations after which the pointer "
        "returns exactly to its original position and orientation?\n"
        "\n[domain: dance]\n"
        f"Four dancers stand around the center of a stage. Every beat, the whole formation "
        f"rotates by one step of {angle} degrees about the stage center, keeping each "
        "dancer's pose fixed relative to the formation.\n\n"
        "Question: what is the smallest number of beats after which the formation is "
        "exactly back to its starting configuration?\n")

    math_md = (
        f"# Mathematics voice — {sample_id}\n\n"
        "Rotation matrix:\n"
        "$$R_\\theta = \\begin{bmatrix}\\cos\\theta & -\\sin\\theta\\\\ "
        "\\sin\\theta & \\cos\\theta\\end{bmatrix}, \\qquad "
        f"\\theta = \\frac{{2\\pi}}{{{order}}}.$$\n"
        f"**Question.** What is the order of $R_\\theta$ in $SO(2)$, i.e. the smallest "
        "$k \\ge 1$ with $R_\\theta^k = I$?\n\n"
        f"**Solution.** $R_\\theta^{{{order}}} = I$ and $R_\\theta^k \\ne I$ for "
        f"$1 \\le k < {order}$, so the order is ${order}$; the orbit of the (asymmetric) "
        f"base shape has exactly ${order}$ elements and generates the cyclic group "
        f"$C_{{{order}}}$.\n")

    program = PROGRAM_SYM.replace("@SID@", sample_id) \
                         .replace("@POINTS@", repr(BASE_SHAPE)) \
                         .replace("@ANGLE@", str(float(angle))) \
                         .replace("@EXPECTED@", str(order))

    drawing = {
        "base_primitive": {"type": "polygon", "points": [list(p) for p in BASE_SHAPE]},
        "transform_set": [{"type": "rotate", "angle": angle * k} for k in range(order)],
        "composition": "union",
        "canvas_center": [384, 384],
    }

    return dict(schema=schema, signature=signature, story=story, math=math_md,
                program=program, drawing=drawing,
                render=lambda style: render_symmetry_svg(drawing, style, seed))


PROGRAM_SYM = '''"""GEB benchmark — program voice. sample: @SID@ (symmetry_lab / rotation_order)"""

import math

POINTS = @POINTS@
ANGLE_DEG = @ANGLE@
EXPECTED_ORDER = @EXPECTED@


def rotate(points, angle_deg, center=(0.0, 0.0)):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    cx, cy = center
    return [(cx + ca * (x - cx) - sa * (y - cy),
             cy + sa * (x - cx) + ca * (y - cy)) for x, y in points]


def close(ps, qs, tol=1e-9):
    return all(abs(px - qx) < tol and abs(py - qy) < tol
               for (px, py), (qx, qy) in zip(ps, qs))


def rotation_order(points, angle_deg, max_k=360):
    pts = points
    for k in range(1, max_k + 1):
        pts = rotate(pts, angle_deg)
        if close(pts, points):
            return k
    raise ValueError("no finite order found")


if __name__ == "__main__":
    result = rotation_order(POINTS, ANGLE_DEG)
    assert result == EXPECTED_ORDER, (result, EXPECTED_ORDER)
    print(f"rotation_order(angle={ANGLE_DEG}) = {result}  [matches schema]")
'''


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def verify_program(code: str):
    """Execute a generated program voice; its own assert is the check."""
    exec(compile(code, "<program-voice>", "exec"), {"__name__": "__main__"})


def write_sample(outdir: Path, sample: dict):
    sid = sample["schema"]["sample_id"]
    d = outdir / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "schema.json").write_text(json.dumps(sample["schema"], indent=2) + "\n")
    (d / "signature.json").write_text(json.dumps(sample["signature"], indent=2) + "\n")
    (d / "story.txt").write_text(sample["story"])
    (d / "math.md").write_text(sample["math"])
    (d / "program.py").write_text(sample["program"])
    (d / "drawing.json").write_text(json.dumps(sample["drawing"], indent=2) + "\n")
    (d / "canonical.svg").write_text(sample["render"]("canonical"))
    for style in ("blueprint", "pencil"):
        (d / f"style_{style}.svg").write_text(sample["render"](style))


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "examples"
    samples = [
        gen_graph_sample("gw_reach_yes_0001", "reachability", True, seed=11),
        gen_graph_sample("gw_reach_no_0002", "reachability", False, seed=23),
        gen_graph_sample("gw_cycle_0003", "cycle", True, seed=37),
        gen_tree_sample("rg_btree_0004", depth=4, seed=51),
        gen_symmetry_sample("sym_c4_0005", order=4, seed=67),
    ]
    index = []
    for s in samples:
        verify_program(s["program"])          # executable ground truth check
        write_sample(outdir, s)
        sch = s["schema"]
        index.append({"sample_id": sch["sample_id"], "family": sch["family"],
                      "task_type": sch["task_type"], "answer": sch["answer"]})
        print(f"  wrote {sch['sample_id']:<22} {sch['family']:<18} "
              f"{sch['task_type']:<15} answer={sch['answer']}")
    (outdir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\n{len(samples)} samples -> {outdir}")


if __name__ == "__main__":
    main()
