#!/usr/bin/env python3
"""Exp1 — FORMAL predictor graphs, hand-coded from skeleton_spec.

Each motif's skeleton_spec (data/motif_library.json) is reduced BY HAND to a
small directed graph (<=10 nodes) capturing its topology.  Reductions are
approximate and documented inline; the goal is relative graph-edit distance,
not faithful rendering.  Node attribute ``c`` (small int) encodes a state /
role where the motif's identity depends on it (alternation colors, deviant
cell, tile species, dual roles ...); substitution of differing ``c`` costs 1.

Conventions
- DiGraph.  Symmetric relations get both directions; oriented relations
  (recursion parent->child, convergence source->sink, cycles, descent) one.
- Repetition counts are nuisance and reduced to the smallest count that
  preserves the motif (e.g. 10-node alternation chain -> 6, K5 clusters ->
  K3), so GED reflects structure, not size of the drawn instance.
"""
from __future__ import annotations

import networkx as nx


def _bi(g: nx.DiGraph, a, b):
    g.add_edge(a, b)
    g.add_edge(b, a)


def _nodes(g: nx.DiGraph, spec):
    for n, c in spec:
        g.add_node(n, c=c)


def build_graphs() -> dict[str, nx.DiGraph]:
    G = {}

    # self_reference: one node, one self-loop — the minimal fixed point.
    g = nx.DiGraph(); g.add_node(0, c=0); g.add_edge(0, 0)
    G["self_reference"] = g

    # recursion: out-tree, depth 2, branching 2 (7 nodes), root->leaves.
    g = nx.DiGraph()
    for n in range(7):
        g.add_node(n, c=0)
    for p, ch in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        g.add_edge(p, ch)
    G["recursion"] = g

    # nesting: containment chain, arrows pointing only inward (spec: 5 circles,
    # inward arrows).  Directed path of 5.
    g = nx.DiGraph()
    for n in range(5):
        g.add_node(n, c=0)
    for n in range(4):
        g.add_edge(n, n + 1)
    G["nesting"] = g

    # cycle: directed 6-cycle (spec: 6 points, oriented arcs).
    g = nx.DiGraph()
    for n in range(6):
        g.add_node(n, c=0)
    for n in range(6):
        g.add_edge(n, (n + 1) % 6)
    G["cycle"] = g

    # symmetry: 7 nodes on a line, mirror-pair arcs (1,7)(2,6)(3,5),
    # center unpaired.  Pair arcs symmetric.
    g = nx.DiGraph()
    for n in range(7):
        g.add_node(n, c=0)
    for a, b in [(0, 6), (1, 5), (2, 4)]:
        _bi(g, a, b)
    G["symmetry"] = g

    # alternation: chain with strictly alternating node state (10 -> 6 nodes).
    g = nx.DiGraph()
    for n in range(6):
        g.add_node(n, c=n % 2)
    for n in range(5):
        _bi(g, n, n + 1)
    G["alternation"] = g

    # spiral: open ordered chain expanding outward (13 -> 7 points).  Same
    # chain topology as nesting by design — the two ARE formally close;
    # length difference + no shared containment semantics is all GED sees.
    g = nx.DiGraph()
    for n in range(7):
        g.add_node(n, c=0)
    for n in range(6):
        g.add_edge(n, n + 1)
    G["spiral"] = g

    # convergence: in-tree funnel 4 sources -> 2 -> 1 sink (7 nodes), edges
    # oriented toward the sink (mirror image of recursion's orientation).
    g = nx.DiGraph()
    for n in range(7):
        g.add_node(n, c=0)
    for s, t in [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2)]:
        g.add_edge(s, t)
    G["convergence"] = g

    # bottleneck: two K3 clusters (K5 reduced) joined ONLY through one cut
    # vertex, 2 spokes per side.
    g = nx.DiGraph()
    for n in range(7):
        g.add_node(n, c=0)
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        _bi(g, a, b)
    for x in (0, 1, 3, 4):
        _bi(g, x, 6)  # 6 = cut vertex
    G["bottleneck"] = g

    # symmetry_breaking: regular field (7x7 grid reduced to a 6-path) with
    # exactly one deviant cell (c=1) mid-chain.
    g = nx.DiGraph()
    for n in range(6):
        g.add_node(n, c=1 if n == 3 else 0)
    for n in range(5):
        _bi(g, n, n + 1)
    G["symmetry_breaking"] = g

    # strange_loop: upward chain of 4 levels + back edge top->bottom, i.e.
    # a directed 4-cycle.  (Formally a cycle through a hierarchy — its
    # closeness to `cycle` is real and intended by the family design.)
    g = nx.DiGraph()
    for n in range(4):
        g.add_node(n, c=0)
    for n in range(3):
        g.add_edge(n, n + 1)
    g.add_edge(3, 0)
    G["strange_loop"] = g

    # figure_ground: two rows of complementary-colored cells, aligned by
    # pairing edges; colors bitwise inverted (8 -> 4 columns).
    g = nx.DiGraph()
    pat = [1, 0, 1, 0]
    for i, b in enumerate(pat):
        g.add_node(("t", i), c=b)
        g.add_node(("b", i), c=1 - b)
        _bi(g, ("t", i), ("b", i))
    G["figure_ground"] = g

    # isomorphism: same graph drawn twice + vertex correspondence.
    # C5 reduced to K3; two copies, 3 pairing edges, uniform color.
    g = nx.DiGraph()
    for s in ("L", "R"):
        for i in range(3):
            g.add_node((s, i), c=0)
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            _bi(g, (s, a), (s, b))
    for i in range(3):
        _bi(g, ("L", i), ("R", i))
    G["isomorphism"] = g

    # mobius: ONE closed directed orbit of length 8 whose first half is
    # "outer" (c=0), second half "inner" (c=1): one loop, two sides swapped.
    g = nx.DiGraph()
    for n in range(8):
        g.add_node(n, c=0 if n < 4 else 1)
    for n in range(8):
        g.add_edge(n, (n + 1) % 8)
    G["mobius"] = g

    # braiding: 3 strands, endpoints connected by a 3-cycle permutation
    # (composite of the sigma-swaps; terminal order differs from initial).
    g = nx.DiGraph()
    for i in range(3):
        g.add_node(("s", i), c=0)
        g.add_node(("e", i), c=1)
    perm = {0: 1, 1: 2, 2: 0}
    for i in range(3):
        g.add_edge(("s", i), ("e", perm[i]))
    G["braiding"] = g

    # interference: two independent sources, each with its own wavefronts,
    # plus a shared beat node fed by BOTH — the pattern neither makes alone.
    g = nx.DiGraph()
    for s in ("A", "B"):
        g.add_node(s, c=0)
        for k in range(2):
            g.add_node((s, k), c=0)
            g.add_edge(s, (s, k))
    g.add_node("beat", c=1)
    g.add_edge("A", "beat")
    g.add_edge("B", "beat")
    G["interference"] = g

    # tiling: periodic unit cell — one hexagon, undirected (honeycomb reduced
    # to a single translated tile).
    g = nx.DiGraph()
    for n in range(6):
        g.add_node(n, c=0)
    for n in range(6):
        _bi(g, n, (n + 1) % 6)
    G["tiling"] = g

    # hub: star K_{1,5} (K_{1,8} reduced), no rim edges, symmetric spokes.
    g = nx.DiGraph()
    g.add_node("c", c=1)
    for i in range(5):
        g.add_node(i, c=0)
        _bi(g, "c", i)
    G["hub"] = g

    # diagonalization: diagonal cells (bits 1,0,1) each mapped to an escape-
    # row cell with the OPPOSITE bit (6x6 reduced to 3 diagonal positions).
    g = nx.DiGraph()
    bits = [1, 0, 1]
    for i, b in enumerate(bits):
        g.add_node(("d", i), c=b)
        g.add_node(("e", i), c=1 - b)
        g.add_edge(("d", i), ("e", i))
    G["diagonalization"] = g

    # infinite_descent: strictly descending chain hitting a hard floor
    # (c=1 terminal) — the blocked next step.
    g = nx.DiGraph()
    for n in range(5):
        g.add_node(n, c=0)
    g.add_node("floor", c=1)
    for n in range(4):
        g.add_edge(n, n + 1)
    g.add_edge(4, "floor")
    G["infinite_descent"] = g

    # coprime_orbit: a single closed orbit visiting all 8 points ({8/3} star
    # is, as a graph, an 8-cycle; the coprimality lives in the embedding).
    g = nx.DiGraph()
    for n in range(8):
        g.add_node(n, c=0)
    for n in range(8):
        g.add_edge(n, (n + 1) % 8)
    G["coprime_orbit"] = g

    # pigeonhole: items (c=0) mapped into fewer boxes (c=1); one box
    # necessarily receives two (5>4 reduced to 3>2).
    g = nx.DiGraph()
    for i in range(3):
        g.add_node(("i", i), c=0)
    for b in range(2):
        g.add_node(("b", b), c=1)
    g.add_edge(("i", 0), ("b", 0))
    g.add_edge(("i", 1), ("b", 1))
    g.add_edge(("i", 2), ("b", 1))
    G["pigeonhole"] = g

    # projective_duality: two isomorphic sides whose element ROLES are
    # swapped (points<->lines): left K3 of c=0, right K3 of c=1, paired.
    g = nx.DiGraph()
    for s, col in (("L", 0), ("R", 1)):
        for i in range(3):
            g.add_node((s, i), c=col)
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            _bi(g, (s, a), (s, b))
    for i in range(3):
        _bi(g, ("L", i), ("R", i))
    G["projective_duality"] = g

    # contraction_fixed_point: orbit chain collapsing INTO a fixed point that
    # maps to itself (self-loop terminal) — descent + self_reference hybrid.
    g = nx.DiGraph()
    for n in range(4):
        g.add_node(n, c=0)
    g.add_node("fix", c=1)
    for n in range(3):
        g.add_edge(n, n + 1)
    g.add_edge(3, "fix")
    g.add_edge("fix", "fix")
    G["contraction_fixed_point"] = g

    # aperiodic_tiling: two tile SPECIES around a center, non-uniform
    # neighborhood (sun patch reduced: center + 3 fat c=0 + 2 thin c=1,
    # thin tiles wedged between fats).
    g = nx.DiGraph()
    g.add_node("C", c=0)
    for i in range(3):
        g.add_node(("f", i), c=0)
        _bi(g, "C", ("f", i))
    for i in range(2):
        g.add_node(("t", i), c=1)
        _bi(g, ("f", i), ("t", i))
        _bi(g, ("f", i + 1), ("t", i))
    G["aperiodic_tiling"] = g

    return G


if __name__ == "__main__":
    for mid, g in build_graphs().items():
        print(f"{mid:26s} n={g.number_of_nodes():2d} e={g.number_of_edges():2d}")
