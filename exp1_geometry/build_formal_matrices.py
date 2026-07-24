#!/usr/bin/env python3
"""Exp1 — build the 25x25 predictor matrices that do NOT need image models.

  formal_family : binary distance, 0 if the two motifs share a confusable
                  family (eval/build_items.py FAMILIES — fixed before any
                  render existed), 1 otherwise.  PRIMARY formal predictor.
  formal_ged    : normalized graph edit distance between hand-coded skeleton
                  graphs (formal_graphs.py).  SECONDARY formal predictor.
  label_prior   : cosine distance between text-embedding-3-small embeddings
                  of "name_en + invariant_zh" per motif.

Output: exp1_geometry/matrices_formal.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formal_graphs import build_graphs

ROOT = Path(__file__).resolve().parent.parent
LIB = {m["id"]: m for m in
       json.loads((ROOT / "data" / "motif_library.json").read_text())}
MOTIFS = list(LIB)  # library order, canonical for every matrix in exp1

# confusable families — copied verbatim from eval/build_items.py:34 (fixed
# by skeleton specs before any render existed; circularity defense).
FAMILIES = [
    ["self_reference", "strange_loop", "recursion", "nesting"],
    ["cycle", "strange_loop", "mobius", "spiral", "alternation"],
    ["symmetry", "figure_ground", "isomorphism", "mobius"],
    ["tiling", "symmetry_breaking", "interference", "alternation", "braiding"],
    ["convergence", "bottleneck", "hub"],
    ["diagonalization", "self_reference", "symmetry_breaking"],
    ["infinite_descent", "recursion", "contraction_fixed_point", "convergence"],
    ["coprime_orbit", "cycle", "interference", "pigeonhole"],
    ["pigeonhole", "convergence", "bottleneck", "hub"],
    ["projective_duality", "isomorphism", "figure_ground", "symmetry"],
    ["contraction_fixed_point", "spiral", "convergence", "nesting"],
    ["aperiodic_tiling", "tiling", "interference", "alternation"],
]


def family_matrix() -> np.ndarray:
    n = len(MOTIFS)
    D = np.ones((n, n))
    np.fill_diagonal(D, 0.0)
    idx = {m: i for i, m in enumerate(MOTIFS)}
    for fam in FAMILIES:
        for a in fam:
            for b in fam:
                if a != b:
                    D[idx[a], idx[b]] = 0.0
    return D


def ged_matrix(timeout: float = 30.0) -> np.ndarray:
    graphs = build_graphs()

    def ncost(u, v):
        return 0.0 if u.get("c") == v.get("c") else 1.0

    n = len(MOTIFS)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            g1, g2 = graphs[MOTIFS[i]], graphs[MOTIFS[j]]
            t0 = time.time()
            d = nx.graph_edit_distance(
                g1, g2, timeout=timeout,
                node_subst_cost=ncost,
                node_del_cost=lambda u: 1.0, node_ins_cost=lambda u: 1.0,
                edge_del_cost=lambda e: 1.0, edge_ins_cost=lambda e: 1.0)
            if d is None:  # timeout with no estimate — upper bound fallback
                d = (g1.number_of_nodes() + g2.number_of_nodes()
                     + g1.number_of_edges() + g2.number_of_edges())
            # normalize by combined size so big-vs-big != automatically far
            size = (g1.number_of_nodes() + g2.number_of_nodes()
                    + g1.number_of_edges() + g2.number_of_edges())
            D[i, j] = D[j, i] = d / size
            print(f"  GED {MOTIFS[i]:24s} {MOTIFS[j]:24s} "
                  f"{d:5.1f} norm={D[i, j]:.3f} ({time.time()-t0:.1f}s)",
                  flush=True)
    return D


def embed_texts(texts: list[str], model="text-embedding-3-small",
                retries=3) -> np.ndarray:
    body = json.dumps({"model": model, "input": texts})
    cmd = ["curl", "-s", "--max-time", "120", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/embeddings",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json", "-d", body]
    for a in range(retries):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            resp = json.loads(raw)
            if "data" in resp:
                E = np.array([d["embedding"] for d in
                              sorted(resp["data"], key=lambda d: d["index"])])
                return E / np.linalg.norm(E, axis=1, keepdims=True)
        except json.JSONDecodeError:
            pass
        print(f"  embeddings retry {a+1}: {raw[:150]}", flush=True)
        time.sleep(3)
    raise RuntimeError("embeddings API failed")


def label_matrix() -> np.ndarray:
    texts = [f"{LIB[m]['name_en']}. Invariant: {LIB[m]['invariant_zh']}"
             for m in MOTIFS]
    E = embed_texts(texts)
    return 1.0 - E @ E.T


def main():
    print("family matrix ...", flush=True)
    fam = family_matrix()
    print("label-prior matrix ...", flush=True)
    lab = label_matrix()
    print("GED matrix (300 pairs) ...", flush=True)
    ged = ged_matrix()
    out = {"motifs": MOTIFS,
           "formal_family": fam.tolist(),
           "formal_ged": ged.tolist(),
           "label_prior": lab.tolist()}
    p = ROOT / "exp1_geometry" / "matrices_formal.json"
    p.write_text(json.dumps(out))
    print("wrote", p)


if __name__ == "__main__":
    main()
