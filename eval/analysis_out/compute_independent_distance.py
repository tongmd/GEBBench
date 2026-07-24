#!/usr/bin/env python3
"""Independent formal distance, step 2: 25x25 distance matrices.

  ged      : typed graph edit distance between the relational graphs of
             motif_graphs.json (networkx graph_edit_distance, typed node/edge
             substitution costs per rubric R5/R6, timeout per pair),
             normalized per R7.
  eq_token : token-level normalized Levenshtein distance between
             equation_katex strings (KaTeX commands are single tokens).

Output: eval/analysis_out/independent_distance.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path("/Users/stone/GEB-benchmark")
OUT = ROOT / "eval" / "analysis_out"
TIMEOUT = 30.0

# ---- typed costs (rubric R5/R6) -------------------------------------------
def node_subst(u, v):
    a, b = u["t"], v["t"]
    if a == b:
        return 0.0
    if {a, b} == {"s0", "s1"}:
        return 0.25
    if "d" in (a, b):
        return 1.0
    return 0.5  # u vs s*


def edge_subst(e, f):
    a, b = e["t"], f["t"]
    if a == b:
        return 0.0
    if {a, b} == {"link", "contain"}:
        return 0.5
    return 1.0


def load_graphs():
    d = json.loads((OUT / "motif_graphs.json").read_text())
    G = {}
    for mid, spec in d["motifs"].items():
        g = nx.DiGraph()
        for n, t in spec["nodes"]:
            g.add_node(n, t=t)
        for a, b, t in spec["edges"]:
            g.add_edge(a, b, t=t)
        G[mid] = g
    return G


def ged_matrix(motifs, G):
    n = len(motifs)
    D = np.zeros((n, n))
    raw = np.zeros((n, n))
    timeouts = []
    for i in range(n):
        for j in range(i + 1, n):
            g1, g2 = G[motifs[i]], G[motifs[j]]
            t0 = time.time()
            best = None
            try:
                best = nx.graph_edit_distance(
                    g1, g2,
                    node_subst_cost=node_subst,
                    node_del_cost=lambda u: 1.0,
                    node_ins_cost=lambda u: 1.0,
                    edge_subst_cost=edge_subst,
                    edge_del_cost=lambda e: 1.0,
                    edge_ins_cost=lambda e: 1.0,
                    timeout=TIMEOUT)
                if time.time() - t0 > TIMEOUT:
                    timeouts.append((motifs[i], motifs[j]))
            except Exception as ex:
                print("ERR", motifs[i], motifs[j], ex)
            if best is None:
                best = (g1.number_of_nodes() + g2.number_of_nodes()
                        + g1.number_of_edges() + g2.number_of_edges())
            size = (g1.number_of_nodes() + g2.number_of_nodes()
                    + g1.number_of_edges() + g2.number_of_edges())
            raw[i, j] = raw[j, i] = best
            D[i, j] = D[j, i] = best / size
            print(f"GED {motifs[i]:24s} {motifs[j]:24s} raw={best:6.2f} "
                  f"norm={D[i, j]:.3f} ({time.time()-t0:.1f}s)", flush=True)
    return D, raw, timeouts


# ---- equation token distance ----------------------------------------------
TOKEN_RE = re.compile(
    r"\\[A-Za-z]+"          # \command
    r"|\\[^A-Za-z]"         # \{ \, etc.
    r"|[A-Za-z]+"           # identifier runs
    r"|\d+"                 # numbers
    r"|[^\sA-Za-z0-9]"      # single punctuation/symbol
)
# strip Chinese \text{...} content: purely presentational
CJK_RE = re.compile(r"[一-鿿]+")


def tokenize(eq):
    eq = CJK_RE.sub("T", eq)
    return TOKEN_RE.findall(eq)


def lev(a, b):
    m, n = len(a), len(b)
    dp = np.arange(n + 1, dtype=float)
    for i in range(1, m + 1):
        prev = dp.copy()
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1,
                        prev[j - 1] + (a[i - 1] != b[j - 1]))
    return dp[n]


def eq_matrix(motifs, lib):
    toks = {m: tokenize(lib[m]["equation_katex"]) for m in motifs}
    n = len(motifs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = toks[motifs[i]], toks[motifs[j]]
            D[i, j] = D[j, i] = lev(a, b) / max(len(a), len(b))
    return D, {m: len(t) for m, t in toks.items()}


def main():
    lib = {m["id"]: m for m in
           json.loads((ROOT / "data" / "motif_library.json").read_text())}
    motifs = list(lib)
    G = load_graphs()

    print("== equation token distance ==")
    EQ, tok_lens = eq_matrix(motifs, lib)
    print("token lengths:", tok_lens)

    print("== typed GED (300 pairs) ==")
    D, raw, timeouts = ged_matrix(motifs, G)

    out = {
        "motifs": motifs,
        "ged_normalized": D.tolist(),
        "ged_raw": raw.tolist(),
        "ged_timeout_pairs": timeouts,
        "ged_meta": {
            "algorithm": "networkx.optimize_graph_edit_distance, best value "
                         f"within {TIMEOUT}s per pair (anytime upper bound)",
            "node_subst": "0 same type; 0.25 s0<->s1; 0.5 u<->s*; 1.0 d<->*",
            "edge_subst": "0 same type; 0.5 link<->contain; 1.0 otherwise",
            "indel": "1.0 each node/edge",
            "normalization": "raw / (|V1|+|V2|+|E1|+|E2|)",
        },
        "eq_token_distance": EQ.tolist(),
        "eq_meta": {
            "tokenizer": "KaTeX commands, identifier runs, numbers, single "
                         "symbols; CJK text collapsed to one token",
            "distance": "Levenshtein / max(len_a, len_b)",
            "token_lengths": tok_lens,
        },
    }
    (OUT / "independent_distance.json").write_text(json.dumps(out))
    print("wrote", OUT / "independent_distance.json",
          f"({len(timeouts)} timeouts)")


if __name__ == "__main__":
    main()
