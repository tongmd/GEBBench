#!/usr/bin/env python3
"""GEB benchmark — motif pipeline: abstract structure -> skeleton -> natural scene.

The unit of a sample is an ABSTRACT STRUCTURAL MOTIF (self-reference, recursion,
nesting, cycle ...), not a concrete graph.  Node counts, depths, branching
factors, angles are nuisance variables and are recorded as such.

Pipeline per motif:
    motif z
      -> skeleton(s): programmatic vis-graph renderings of the motif
      -> scene(s):    gpt-image with the skeleton as REFERENCE image, the scene
                      prompt reinterprets it as a natural / artistic image
                      (rivers, lightning, city walls, ouroboros ...), so the
                      composition IS the structure — not a style transfer
      -> story / math voices written at motif level (no counts, no sizes)

Usage:  python3 gen_motifs.py [motif_id ...]
"""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
from pathlib import Path

MODEL = "gpt-image-2"
ROOT = Path(__file__).parent
OUT = ROOT / "motifs"

NO_DIAGRAM = ("It must look like a real photograph or a piece of art, NOT a "
              "diagram: no visible diagram lines, no text, no letters.")


# ---------------------------------------------------------------------------
# skeleton renderers (SVG strings; plain, thin, reference-only)
# ---------------------------------------------------------------------------

def svg_doc(w: int, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
            f'<rect width="100%" height="100%" fill="#ffffff"/>{body}</svg>')


def skel_tree(depth: int = 4, angle: float = 34, scale: float = 0.62,
              length: float = 300, w: int = 1024, h: int = 1024) -> str:
    parts = []

    def branch(x, y, heading, ln, level):
        x2 = x + ln * math.cos(math.radians(heading))
        y2 = y + ln * math.sin(math.radians(heading))
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#1a1a1a" stroke-width="{max(2, 9 - 2 * level)}" '
                     'stroke-linecap="round"/>')
        if level < depth:
            branch(x2, y2, heading - angle, ln * scale, level + 1)
            branch(x2, y2, heading + angle, ln * scale, level + 1)

    branch(w / 2, h - 40, -90, length, 0)
    return svg_doc(w, h, "".join(parts))


def skel_concentric(n: int = 7, w: int = 1024, h: int = 1024) -> str:
    cx, cy, r_max = w / 2, h / 2, min(w, h) / 2 - 40
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{r_max * (k + 1) / n:.1f}" fill="none" '
             'stroke="#1a1a1a" stroke-width="4"/>' for k in range(n)]
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="#1a1a1a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_ring(w: int = 1024, h: int = 1024, segments: int = 8) -> str:
    cx, cy, r = w / 2, h / 2, min(w, h) / 2 - 90
    marker = ('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" '
              'markerWidth="8" markerHeight="8" orient="auto">'
              '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a"/></marker></defs>')
    parts = [marker]
    for k in range(segments):
        a1 = 2 * math.pi * k / segments
        a2 = a1 + 2 * math.pi / segments * 0.72
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
        parts.append(f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
                     'fill="none" stroke="#1a1a1a" stroke-width="5" marker-end="url(#a)"/>')
    return svg_doc(w, h, "".join(parts))


def skel_frames(levels: int = 4, scale: float = 0.62, w: int = 1536, h: int = 1024) -> str:
    parts, fw, fh = [], w - 160, h - 160
    for _ in range(levels):
        parts.append(f'<rect x="{(w - fw) / 2:.1f}" y="{(h - fh) / 2:.1f}" '
                     f'width="{fw:.1f}" height="{fh:.1f}" fill="none" '
                     'stroke="#1a1a1a" stroke-width="5"/>')
        fw, fh = fw * scale, fh * scale
    return svg_doc(w, h, "".join(parts))


def skel_mirror(w: int = 1536, h: int = 1024) -> str:
    """symmetry: an irregular half, exactly mirrored across a vertical axis."""
    parts = [f'<line x1="{w / 2}" y1="60" x2="{w / 2}" y2="{h - 60}" '
             'stroke="#1a1a1a" stroke-width="2" stroke-dasharray="10 12"/>']
    # one irregular polyline half + mirrored twin, plus matched node dots
    pts = [(0, 220), (-160, 320), (-110, 470), (-300, 560),
           (-210, 700), (-380, 790), (-260, 880)]
    for sgn in (-1, 1):
        path = " ".join(f"{'M' if i == 0 else 'L'} {w / 2 + sgn * abs(x):.0f} {y}"
                        for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="#1a1a1a" '
                     'stroke-width="6" stroke-linejoin="round"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{w / 2 + sgn * abs(x):.0f}" cy="{y}" r="11" '
                         'fill="#1a1a1a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_alternation(n: int = 10, w: int = 1536, h: int = 1024) -> str:
    """alternation: a chain whose nodes strictly alternate two states (A B A B ...)."""
    parts, margin = [], 130
    step = (w - 2 * margin) / (n - 1)
    for k in range(n - 1):
        x1, x2 = margin + k * step, margin + (k + 1) * step
        y1 = h / 2 + (-70 if k % 2 == 0 else 70)
        y2 = h / 2 + (-70 if (k + 1) % 2 == 0 else 70)
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     'stroke="#1a1a1a" stroke-width="5"/>')
    for k in range(n):
        x = margin + k * step
        y = h / 2 + (-70 if k % 2 == 0 else 70)
        if k % 2 == 0:
            parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="26" fill="#1a1a1a"/>')
        else:
            parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="26" fill="#ffffff" '
                         'stroke="#1a1a1a" stroke-width="6"/>')
    return svg_doc(w, h, "".join(parts))


def skel_spiral(turns: float = 3.4, b: float = 0.16, w: int = 1024, h: int = 1024) -> str:
    """spiral: logarithmic spiral r = a e^{b theta} polyline with dots on it."""
    cx, cy = w / 2, h / 2
    theta_max = turns * 2 * math.pi
    a = (min(w, h) / 2 - 60) / math.exp(b * theta_max)
    pts = []
    steps = 400
    for i in range(steps + 1):
        t = theta_max * i / steps
        r = a * math.exp(b * t)
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    path = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}"
                    for i, (x, y) in enumerate(pts))
    parts = [f'<path d="{path}" fill="none" stroke="#1a1a1a" stroke-width="6"/>',
             f'<circle cx="{cx}" cy="{cy}" r="7" fill="#1a1a1a"/>']
    # dots each time the spiral returns to the same heading: radius grows by e^{2 pi b}
    for k in range(int(turns) + 1):
        t = 2 * math.pi * k
        r = a * math.exp(b * t)
        parts.append(f'<circle cx="{cx + r:.1f}" cy="{cy:.1f}" r="13" fill="#1a1a1a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_converge(w: int = 1536, h: int = 1024) -> str:
    """convergence: many sources, paths merging pairwise until a single sink."""
    marker = ('<defs><marker id="c" viewBox="0 0 10 10" refX="8" refY="5" '
              'markerWidth="7" markerHeight="7" orient="auto">'
              '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a"/></marker></defs>')
    parts = [marker]
    # binary merge tree, left -> right: 8 sources -> 4 -> 2 -> 1 sink
    levels = 3
    xs = [140 + k * (w - 300) / levels for k in range(levels + 1)]
    ys = {3: [h * (i + 1) / 9 for i in range(8)]}
    for lv in (2, 1, 0):
        prev = ys[lv + 1]
        ys[lv] = [(prev[2 * i] + prev[2 * i + 1]) / 2 for i in range(len(prev) // 2)]
    for lv in (3, 2, 1):
        x1, x2 = xs[3 - lv], xs[3 - lv + 1]
        for i, y in enumerate(ys[lv]):
            y2 = ys[lv - 1][i // 2]
            parts.append(f'<path d="M {x1:.0f} {y:.0f} C {x1 + 90:.0f} {y:.0f} '
                         f'{x2 - 90:.0f} {y2:.0f} {x2:.0f} {y2:.0f}" fill="none" '
                         f'stroke="#1a1a1a" stroke-width="{3 + (3 - lv) * 2}" '
                         'marker-end="url(#c)"/>')
    for i, y in enumerate(ys[3]):
        parts.append(f'<circle cx="{xs[0]:.0f}" cy="{y:.0f}" r="12" fill="#1a1a1a"/>')
    parts.append(f'<circle cx="{xs[3]:.0f}" cy="{ys[0][0]:.0f}" r="22" fill="#1a1a1a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_bottleneck(w: int = 1536, h: int = 1024) -> str:
    """bottleneck: two clusters connected ONLY through one middle cut vertex."""
    cxl, cxr, cy = 300, w - 300, h / 2
    mid = (w / 2, cy)
    left = [(cxl + 150 * math.cos(a), cy + 260 * math.sin(a))
            for a in [k * 2 * math.pi / 7 + 0.4 for k in range(7)]]
    right = [(cxr + 150 * math.cos(a), cy + 260 * math.sin(a))
             for a in [k * 2 * math.pi / 7 + 1.1 for k in range(7)]]
    parts = []
    for cluster in (left, right):
        for i, (x1, y1) in enumerate(cluster):        # sparse intra-cluster edges
            x2, y2 = cluster[(i + 2) % len(cluster)]
            parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                         'stroke="#1a1a1a" stroke-width="3"/>')
    # every cross connection passes through the single middle node
    for x, y in left[:4] + right[:4]:
        parts.append(f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{mid[0]:.0f}" y2="{mid[1]:.0f}" '
                     'stroke="#1a1a1a" stroke-width="4"/>')
    for x, y in left + right:
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="13" fill="#1a1a1a"/>')
    parts.append(f'<circle cx="{mid[0]:.0f}" cy="{mid[1]:.0f}" r="24" fill="#ffffff" '
                 'stroke="#1a1a1a" stroke-width="7"/>')
    return svg_doc(w, h, "".join(parts))


def skel_broken(rows: int = 6, cols: int = 9, w: int = 1536, h: int = 1024) -> str:
    """symmetry_breaking: a perfectly regular field with exactly one deviant cell."""
    parts = []
    mx, my = 120, 110
    dx, dy = (w - 2 * mx) / (cols - 1), (h - 2 * my) / (rows - 1)
    odd = (2, 5)  # the one exception (row, col)
    for r in range(rows):
        for c in range(cols):
            x, y = mx + c * dx, my + r * dy
            if (r, c) == odd:
                parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="30" '
                             'fill="#1a1a1a" transform="rotate(0)"/>')
            else:
                parts.append(f'<rect x="{x - 24:.0f}" y="{y - 24:.0f}" width="48" '
                             'height="48" fill="none" stroke="#1a1a1a" '
                             'stroke-width="5"/>')
    return svg_doc(w, h, "".join(parts))


def skel_strange_loop(levels: int = 4, w: int = 1024, h: int = 1024) -> str:
    """strange loop: a ladder of levels climbing upward — and one edge from the
    top leading back to the bottom (a back edge across the hierarchy)."""
    marker = ('<defs><marker id="s" viewBox="0 0 10 10" refX="8" refY="5" '
              'markerWidth="7" markerHeight="7" orient="auto">'
              '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a"/></marker></defs>')
    parts = [marker]
    xs, ys = [], []
    for k in range(levels):
        x = 300 + (k % 2) * (w - 640)
        y = h - 160 - k * (h - 320) / (levels - 1)
        xs.append(x)
        ys.append(y)
    for k in range(levels):  # level platforms
        parts.append(f'<line x1="{xs[k] - 110:.0f}" y1="{ys[k]:.0f}" '
                     f'x2="{xs[k] + 110:.0f}" y2="{ys[k]:.0f}" '
                     'stroke="#1a1a1a" stroke-width="8"/>')
        parts.append(f'<circle cx="{xs[k]:.0f}" cy="{ys[k]:.0f}" r="14" fill="#1a1a1a"/>')
    for k in range(levels - 1):  # upward steps
        parts.append(f'<line x1="{xs[k]:.0f}" y1="{ys[k] - 10:.0f}" '
                     f'x2="{xs[k + 1]:.0f}" y2="{ys[k + 1] + 12:.0f}" '
                     'stroke="#1a1a1a" stroke-width="5" marker-end="url(#s)"/>')
    # the impossible return: top feeds straight back into bottom
    parts.append(f'<path d="M {xs[-1] + 60:.0f} {ys[-1]:.0f} '
                 f'C {w - 60:.0f} {ys[-1]:.0f} {w - 60:.0f} {ys[0]:.0f} '
                 f'{xs[0] + 120:.0f} {ys[0]:.0f}" fill="none" stroke="#1a1a1a" '
                 'stroke-width="6" stroke-dasharray="16 10" marker-end="url(#s)"/>')
    return svg_doc(w, h, "".join(parts))


def skel_figure_ground(rows: int = 4, cols: int = 7, w: int = 1536, h: int = 1024) -> str:
    """figure/ground: interlocking triangles — the white gaps between the black
    figures are exactly the same shape, pointing the other way."""
    parts = []
    tw, th = w / cols, h / rows
    for r in range(rows):
        for c in range(cols + 1):
            x0 = (c - 0.5) * tw if r % 2 else c * tw
            y0 = r * th
            up = f"M {x0:.0f} {y0 + th:.0f} L {x0 + tw / 2:.0f} {y0:.0f} " \
                 f"L {x0 + tw:.0f} {y0 + th:.0f} Z"
            parts.append(f'<path d="{up}" fill="#1a1a1a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_hub(n: int = 9, w: int = 1024, h: int = 1024) -> str:
    """hub: star K_{1,n} — one center touching all, no rim edges."""
    cx, cy, r = w / 2, h / 2, min(w, h) / 2 - 110
    parts = []
    for k in range(n):
        a = 2 * math.pi * k / n - math.pi / 2
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.0f}" y2="{y:.0f}" '
                     'stroke="#1a1a1a" stroke-width="5"/>')
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="16" fill="#1a1a1a"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="34" fill="#1a1a1a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_tiling(w: int = 1536, h: int = 1024, r: float = 110) -> str:
    """tiling: hexagonal lattice, translation-invariant honeycomb."""
    parts = []
    dx, dy = r * 3, r * math.sqrt(3) / 2
    for row in range(-1, int(h / dy) + 2):
        off = 0 if row % 2 == 0 else r * 1.5
        for col in range(-1, int(w / dx) + 2):
            cx, cy = col * dx + off, row * dy
            pts = " ".join(f"{cx + r * math.cos(math.pi / 3 * k):.0f},"
                           f"{cy + r * math.sin(math.pi / 3 * k):.0f}" for k in range(6))
            parts.append(f'<polygon points="{pts}" fill="none" stroke="#1a1a1a" '
                         'stroke-width="5"/>')
    return svg_doc(w, h, "".join(parts))


def skel_braid(strands: int = 3, cycles: float = 2.0, w: int = 1536, h: int = 1024) -> str:
    """braiding: three sinusoidal strands weaving over/under — at each crossing
    the under strand is nicked white."""
    margin, cy, amp, n = 120, h / 2, 190, 600
    xs = [margin + (w - 2 * margin) * i / n for i in range(n + 1)]

    def y(s, i):
        u = i / n
        return cy + amp * math.sin(2 * math.pi * cycles * u + 2 * math.pi * s / strands)

    parts = []
    for s in range(strands):
        d = " ".join(f"{'M' if i == 0 else 'L'} {xs[i]:.0f} {y(s, i):.0f}"
                     for i in range(n + 1))
        parts.append(f'<path d="{d}" fill="none" stroke="#1a1a1a" stroke-width="10" '
                     'stroke-linecap="round"/>')
    # find crossings per pair, alternate over/under, nick the under strand
    for a in range(strands):
        for b in range(a + 1, strands):
            k = 0
            for i in range(n):
                d1, d2 = y(a, i) - y(b, i), y(a, i + 1) - y(b, i + 1)
                if d1 == 0 or d1 * d2 > 0:
                    continue
                over = a if k % 2 == 0 else b   # alternate: a genuine weave
                k += 1
                i0, i1 = max(0, i - 14), min(n, i + 15)
                seg = " ".join(f"{'M' if j == i0 else 'L'} {xs[j]:.0f} "
                               f"{y(over, j):.0f}" for j in range(i0, i1 + 1))
                parts.append(f'<path d="{seg}" fill="none" stroke="#ffffff" '
                             'stroke-width="30" stroke-linecap="round"/>')
                parts.append(f'<path d="{seg}" fill="none" stroke="#1a1a1a" '
                             'stroke-width="10" stroke-linecap="round"/>')
    return svg_doc(w, h, "".join(parts))


def skel_mobius(w: int = 1536, h: int = 1024) -> str:
    """mobius: 3D parametrization of the band projected to 2D — boundary edge
    (one continuous curve, twice around) + depth-sorted rungs with occlusion."""
    cx, cy, R, half = w / 2, h / 2 + 30, 360, 95
    tilt_y, tilt_z = 0.42, 0.9

    def pt(u, v):
        x = (R + v * math.cos(u / 2)) * math.cos(u)
        yy = (R + v * math.cos(u / 2)) * math.sin(u)
        z = v * math.sin(u / 2)
        return (cx + x, cy + yy * tilt_y - z * tilt_z, yy)   # yy = depth

    # rungs, painter-sorted back to front, each with a white casing
    rungs = []
    n_r = 36
    for k in range(n_r):
        u = 2 * math.pi * k / n_r
        p1, p2 = pt(u, -half), pt(u, half)
        rungs.append(((p1[2] + p2[2]) / 2, p1, p2))
    parts = []
    for _, p1, p2 in sorted(rungs, key=lambda t: -t[0]):     # far first
        parts.append(f'<line x1="{p1[0]:.0f}" y1="{p1[1]:.0f}" x2="{p2[0]:.0f}" '
                     f'y2="{p2[1]:.0f}" stroke="#ffffff" stroke-width="18"/>')
        parts.append(f'<line x1="{p1[0]:.0f}" y1="{p1[1]:.0f}" x2="{p2[0]:.0f}" '
                     f'y2="{p2[1]:.0f}" stroke="#1a1a1a" stroke-width="6"/>')
    # boundary edge: v = +half, u through 4*pi closes the single edge
    n_b = 400
    d = " ".join(f"{'M' if i == 0 else 'L'} {pt(4 * math.pi * i / n_b, half)[0]:.0f} "
                 f"{pt(4 * math.pi * i / n_b, half)[1]:.0f}" for i in range(n_b + 1))
    parts.append(f'<path d="{d}" fill="none" stroke="#1a1a1a" stroke-width="5"/>')
    return svg_doc(w, h, "".join(parts))


def skel_isomorphism(w: int = 1536, h: int = 1024) -> str:
    """isomorphism: the same 5-vertex graph drawn twice (pentagon vs star layout),
    dashed lines pairing corresponding vertices."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]   # C5: same edges both sides
    cy, r = h / 2, 260
    left = [(390 + r * math.cos(2 * math.pi * k / 5 - math.pi / 2),
             cy + r * math.sin(2 * math.pi * k / 5 - math.pi / 2)) for k in range(5)]
    # right: same C5 but drawn as a pentagram (vertices reordered 0,2,4,1,3)
    order = [0, 2, 4, 1, 3]
    right = [None] * 5
    for slot, k in enumerate(order):
        right[k] = (w - 390 + r * math.cos(2 * math.pi * slot / 5 - math.pi / 2),
                    cy + r * math.sin(2 * math.pi * slot / 5 - math.pi / 2))
    parts = []
    for pts in (left, right):
        for a, b in edges:
            parts.append(f'<line x1="{pts[a][0]:.0f}" y1="{pts[a][1]:.0f}" '
                         f'x2="{pts[b][0]:.0f}" y2="{pts[b][1]:.0f}" '
                         'stroke="#1a1a1a" stroke-width="6"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="15" fill="#1a1a1a"/>')
    for k in range(5):  # the correspondence phi
        parts.append(f'<line x1="{left[k][0]:.0f}" y1="{left[k][1]:.0f}" '
                     f'x2="{right[k][0]:.0f}" y2="{right[k][1]:.0f}" '
                     'stroke="#8a8a8a" stroke-width="2.5" stroke-dasharray="8 10"/>')
    return svg_doc(w, h, "".join(parts))


def skel_interference(w: int = 1536, h: int = 1024) -> str:
    """interference: two overlapping sets of concentric circles — bright and dark
    bands (hyperbolas) appear where the wavefronts agree / disagree."""
    parts = []
    for cx in (w / 2 - 230, w / 2 + 230):
        for k in range(1, 14):
            parts.append(f'<circle cx="{cx:.0f}" cy="{h / 2:.0f}" r="{k * 72}" '
                         'fill="none" stroke="#1a1a1a" stroke-width="5"/>')
    return svg_doc(w, h, "".join(parts))


def skel_diagonal(w: int = 1024, h: int = 1024) -> str:
    """diagonalization: a 6x6 bit grid, the anti-diagonal flipped into a new
    row that differs from every row — the escape below the table."""
    M = [[1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 1],
         [0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0]]
    g, x0, y0, r = 130, 120, 110, 24
    parts = []
    for i in range(6):
        for j in range(6):
            x, y = x0 + g * j, y0 + g * i
            fill = "#1a1a1a" if M[i][j] else "#ffffff"
            parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
                         'stroke="#1a1a1a" stroke-width="5"/>')
    parts.append(f'<line x1="{x0 - 50}" y1="{y0 - 50}" x2="{x0 + 5 * g + 50}" '
                 f'y2="{y0 + 5 * g + 50}" stroke="#b3402a" stroke-width="4"/>')
    for i in range(6):  # red ring on the diagonal + arrow to escape row
        x = x0 + g * i
        parts.append(f'<circle cx="{x}" cy="{y0 + g * i}" r="{r + 10}" '
                     'fill="none" stroke="#b3402a" stroke-width="5"/>')
        parts.append(f'<line x1="{x}" y1="{y0 + g * i + r + 14}" x2="{x}" '
                     f'y2="{y0 + 6 * g + 40 - r - 8}" stroke="#b3402a" '
                     'stroke-width="3" stroke-dasharray="7 7"/>')
    ye = y0 + 6 * g + 40
    for i in range(6):  # escape row = flipped diagonal
        x = x0 + g * i
        fill = "#ffffff" if M[i][i] else "#1a1a1a"
        parts.append(f'<circle cx="{x}" cy="{ye}" r="{r}" fill="{fill}" '
                     'stroke="#b3402a" stroke-width="6"/>')
    return svg_doc(w, h, "".join(parts))


def skel_descent(w: int = 1024, h: int = 1024) -> str:
    """infinite descent: strictly shrinking squares stepping down to a hard
    floor at 1 — the next step is impossible (red cross)."""
    sizes = [(80, 130, 190), (270, 250, 135), (410, 340, 95),
             (530, 410, 66), (620, 460, 46), (685, 495, 32)]
    parts = []
    marker = ('<defs><marker id="d" viewBox="0 0 10 10" refX="8" refY="5" '
              'markerWidth="7" markerHeight="7" orient="auto">'
              '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a"/></marker></defs>')
    parts.append(marker)
    centers = []
    for x, y, s in sizes:
        parts.append(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" '
                     'fill="#1a1a1a"/>')
        centers.append((x + s / 2, y + s / 2))
    for (x1, y1), (x2, y2) in zip(centers, centers[1:]):
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     'stroke="#1a1a1a" stroke-width="4" marker-end="url(#d)"/>')
    floor_y = 720
    parts.append(f'<line x1="60" y1="{floor_y}" x2="{w - 60}" y2="{floor_y}" '
                 'stroke="#1a1a1a" stroke-width="9"/>')
    x1, y1 = centers[-1]
    parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x1 + 60:.0f}" '
                 f'y2="{floor_y + 60}" stroke="#1a1a1a" stroke-width="4" '
                 'marker-end="url(#d)"/>')
    xc, yc = x1 + 42, floor_y
    parts.append(f'<line x1="{xc - 26:.0f}" y1="{yc - 26}" x2="{xc + 26:.0f}" '
                 f'y2="{yc + 26}" stroke="#b3402a" stroke-width="10"/>')
    parts.append(f'<line x1="{xc - 26:.0f}" y1="{yc + 26}" x2="{xc + 26:.0f}" '
                 f'y2="{yc - 26}" stroke="#b3402a" stroke-width="10"/>')
    return svg_doc(w, h, "".join(parts))


def skel_coprime(w: int = 1024, h: int = 1024) -> str:
    """coprime orbit: {8/3} star polygon (one closed stroke through all 8) +
    a small gcd=2 contrast ring that splits into two squares."""
    cx, cy, R = 400, 420, 310
    pts = [(cx + R * math.cos(2 * math.pi * k / 8 - math.pi / 2),
            cy + R * math.sin(2 * math.pi * k / 8 - math.pi / 2)) for k in range(8)]
    parts = []
    order = [0, 3, 6, 1, 4, 7, 2, 5, 0]
    d = " ".join(f"{'M' if i == 0 else 'L'} {pts[k][0]:.0f} {pts[k][1]:.0f}"
                 for i, k in enumerate(order))
    parts.append(f'<path d="{d}" fill="none" stroke="#1a1a1a" stroke-width="7"/>')
    for x, y in pts:
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="14" fill="#1a1a1a"/>')
    # contrast: step 2 splits into gcd=2 squares
    cx2, cy2, R2 = 830, 830, 130
    pts2 = [(cx2 + R2 * math.cos(2 * math.pi * k / 8 - math.pi / 2),
             cy2 + R2 * math.sin(2 * math.pi * k / 8 - math.pi / 2)) for k in range(8)]
    for start in (0, 1):
        seq = [(start + 2 * i) % 8 for i in range(4)] + [start]
        d2 = " ".join(f"{'M' if i == 0 else 'L'} {pts2[k][0]:.0f} {pts2[k][1]:.0f}"
                      for i, k in enumerate(seq))
        parts.append(f'<path d="{d2}" fill="none" stroke="#8a8a8a" '
                     'stroke-width="3" stroke-dasharray="9 7"/>')
    for x, y in pts2:
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="7" fill="#8a8a8a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_pigeonhole(w: int = 1536, h: int = 1024) -> str:
    """pigeonhole: 5 items, 4 boxes — one box necessarily receives two."""
    marker = ('<defs><marker id="p" viewBox="0 0 10 10" refX="8" refY="5" '
              'markerWidth="7" markerHeight="7" orient="auto">'
              '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a"/></marker></defs>')
    parts = [marker]
    xs_top = [180 + k * 290 for k in range(5)]
    for x in xs_top:
        parts.append(f'<circle cx="{x}" cy="200" r="26" fill="#1a1a1a"/>')
    boxes = [250 + k * 290 for k in range(4)]   # box centers
    targets = [0, 1, 2, 2, 3]                    # item -> box (box 2 doubles)
    for i, x in enumerate(xs_top):
        bx = boxes[targets[i]]
        off = -40 if (targets[i] == 2 and i == 2) else (40 if targets[i] == 2 else 0)
        parts.append(f'<line x1="{x}" y1="236" x2="{bx + off}" y2="640" '
                     'stroke="#1a1a1a" stroke-width="5" marker-end="url(#p)"/>')
    for k, bx in enumerate(boxes):
        hot = k == 2
        stroke = "#b3402a" if hot else "#1a1a1a"
        parts.append(f'<rect x="{bx - 110}" y="660" width="220" height="220" '
                     f'fill="none" stroke="{stroke}" stroke-width="{10 if hot else 6}"/>')
        if hot:
            parts.append(f'<circle cx="{bx - 40}" cy="770" r="24" fill="#1a1a1a"/>')
            parts.append(f'<circle cx="{bx + 40}" cy="770" r="24" fill="#1a1a1a"/>')
        else:
            parts.append(f'<circle cx="{bx}" cy="770" r="24" fill="#1a1a1a"/>')
    return svg_doc(w, h, "".join(parts))


def skel_duality(w: int = 1536, h: int = 1024) -> str:
    """projective duality: complete quadrangle (4 points, 6 lines) vs complete
    quadrilateral (4 lines, 6 points) with a swap arrow between."""
    parts = []
    # left: 4 fat points, 6 thin lines
    P = [(220, 240), (560, 220), (250, 640), (590, 610)]
    for i in range(4):
        for j in range(i + 1, 4):
            (x1, y1), (x2, y2) = P[i], P[j]
            dx, dy = x2 - x1, y2 - y1
            parts.append(f'<line x1="{x1 - 0.25 * dx:.0f}" y1="{y1 - 0.25 * dy:.0f}" '
                         f'x2="{x2 + 0.25 * dx:.0f}" y2="{y2 + 0.25 * dy:.0f}" '
                         'stroke="#1a1a1a" stroke-width="2.5"/>')
    for x, y in P:
        parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="#1a1a1a"/>')
    # middle: sigma double arrow
    parts.append('<path d="M 740 430 L 830 430" stroke="#b3402a" stroke-width="6"/>')
    parts.append('<path d="M 830 430 l -22 -14 v 28 z" fill="#b3402a"/>')
    parts.append('<path d="M 760 470 L 850 470" stroke="#b3402a" stroke-width="6"/>')
    parts.append('<path d="M 760 470 l 22 -14 v 28 z" fill="#b3402a"/>')
    # right: 4 fat lines, 6 small hollow intersection points
    L = [((920, 150), (1450, 690)), ((965, 760), (1430, 200)),
         ((920, 330), (1470, 430)), ((965, 720), (1470, 320))]
    for (x1, y1), (x2, y2) in L:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     'stroke="#1a1a1a" stroke-width="9"/>')

    def inter(l1, l2):
        (x1, y1), (x2, y2) = l1
        (x3, y3), (x4, y4) = l2
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-9:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return px, py
    for i in range(4):
        for j in range(i + 1, 4):
            p = inter(L[i], L[j])
            if p and 900 < p[0] < 1500 and 100 < p[1] < 920:
                parts.append(f'<circle cx="{p[0]:.0f}" cy="{p[1]:.0f}" r="12" '
                             'fill="#ffffff" stroke="#1a1a1a" stroke-width="5"/>')
    return svg_doc(w, h, "".join(parts))


def skel_cobweb(w: int = 1024, h: int = 1024) -> str:
    """contraction fixed point: cobweb plot of x -> x/2 + 80 spiralling into
    the fixed point on the diagonal."""
    s = 3.0  # scale: math coords [0,300] -> px

    def X(v):
        return 60 + v * s

    def Y(v):
        return h - 80 - v * s
    parts = [f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(300)}" y2="{Y(300)}" '
             'stroke="#8a8a8a" stroke-width="3"/>',
             f'<line x1="{X(0)}" y1="{Y(80)}" x2="{X(300)}" y2="{Y(230)}" '
             'stroke="#1a1a1a" stroke-width="6"/>']
    pts = [(20, 20), (20, 90), (90, 90), (90, 125), (125, 125), (125, 142.5),
           (142.5, 142.5), (142.5, 151.25), (151.25, 151.25), (151.25, 155.6),
           (155.6, 155.6)]
    d = " ".join(f"{'M' if i == 0 else 'L'} {X(px):.1f} {Y(py):.1f}"
                 for i, (px, py) in enumerate(pts))
    parts.append(f'<path d="{d}" fill="none" stroke="#b3402a" stroke-width="4"/>')
    parts.append(f'<circle cx="{X(160):.0f}" cy="{Y(160):.0f}" r="12" fill="#b3402a"/>')
    parts.append(f'<circle cx="{X(160):.0f}" cy="{Y(160):.0f}" r="24" fill="none" '
                 'stroke="#b3402a" stroke-width="4"/>')
    return svg_doc(w, h, "".join(parts))


def skel_penrose(w: int = 1024, h: int = 1024) -> str:
    """aperiodic tiling: Penrose P3 'sun' patch — 5 fat rhombs meeting at the
    center, their 5 outer notches filled by thin rhombs. Geometrically exact:
    all tiles edge-to-edge, no gap, no overlap; 5-fold symmetric."""
    cx, cy, s = w / 2, h / 2, 210

    def u(deg):
        return (s * math.cos(math.radians(deg)), -s * math.sin(math.radians(deg)))

    def vadd(*ps):
        return (sum(p[0] for p in ps), sum(p[1] for p in ps))

    def poly(points, fill):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return (f'<polygon points="{d}" fill="{fill}" stroke="#333333" '
                'stroke-width="3"/>')
    parts = []
    C = (cx, cy)
    fat, thin = "#e8c468", "#3b4a8c"
    far = []          # outer tip of each fat rhomb
    shared = []       # shared inner-ring vertex between rhomb k and k+1
    for k in range(5):
        th = 90 + 72 * k
        e1 = u(th - 36)
        e2 = u(th + 36)
        p1 = vadd(C, e1)
        pf = vadd(C, e1, e2)
        p3 = vadd(C, e2)          # == p1 of rhomb k+1
        far.append(pf)
        shared.append(p3)
        parts.append(poly([C, p1, pf, p3], fat))
    for k in range(5):            # thin rhomb fills notch at shared[k]
        v = shared[k]
        a = far[k]
        b = far[(k + 1) % 5]
        d = vadd(a, b, (-v[0], -v[1]))   # a + b - v
        parts.append(poly([v, a, d, b], thin))
    return svg_doc(w, h, "".join(parts))


# ---------------------------------------------------------------------------
# motif registry
#
# All motif metadata (names, equations, formal math, theorems, invariants,
# story form devices) lives in data/motif_library.json — the verified library.
# This file only maps each motif to its skeleton renderers and scene prompts.
# ---------------------------------------------------------------------------

LIBRARY = {m["id"]: m for m in
           json.loads((ROOT / "data" / "motif_library.json").read_text())}

SKELETONS = {
    "recursion": {"tree": lambda: skel_tree()},
    "nesting": {"concentric": lambda: skel_concentric()},
    "cycle": {"ring": lambda: skel_ring()},
    "self_reference": {"frames": lambda: skel_frames()},
    "symmetry": {"mirror": lambda: skel_mirror()},
    "alternation": {"chain": lambda: skel_alternation()},
    "spiral": {"logspiral": lambda: skel_spiral()},
    "convergence": {"funnel": lambda: skel_converge()},
    "bottleneck": {"dumbbell": lambda: skel_bottleneck()},
    "symmetry_breaking": {"field": lambda: skel_broken()},
    "strange_loop": {"ladder": lambda: skel_strange_loop()},
    "figure_ground": {"tessellation": lambda: skel_figure_ground()},
    "isomorphism": {"twographs": lambda: skel_isomorphism()},
    "mobius": {"band": lambda: skel_mobius()},
    "braiding": {"weave": lambda: skel_braid()},
    "interference": {"ripples": lambda: skel_interference()},
    "tiling": {"honeycomb": lambda: skel_tiling()},
    "hub": {"star": lambda: skel_hub()},
    "diagonalization": {"grid": lambda: skel_diagonal()},
    "infinite_descent": {"staircase": lambda: skel_descent()},
    "coprime_orbit": {"star": lambda: skel_coprime()},
    "pigeonhole": {"boxes": lambda: skel_pigeonhole()},
    "projective_duality": {"quadrangle": lambda: skel_duality()},
    "contraction_fixed_point": {"cobweb": lambda: skel_cobweb()},
    "aperiodic_tiling": {"penrose": lambda: skel_penrose()},
}

REF = "Use this diagram ONLY as a compositional skeleton. "

# scenes generated with the skeleton as reference image
SCENES = {
    "recursion": [
        ("river_delta", "tree", "1536x1024",
         REF + "Paint a photorealistic AERIAL photograph of a river delta in "
         "golden wetlands: one main river enters where the trunk is, and every "
         "fork of the diagram becomes a fork of the river channels, splitting "
         "into narrower and narrower waterways along the drawn branches. Water "
         "glinting in low sun, marsh grass, mudflats. " + NO_DIAGRAM),
        ("lightning", "tree", "1536x1024",
         REF + "A dramatic long-exposure photograph of a single lightning bolt "
         "at night: the main discharge channel where the trunk is, every fork "
         "of the diagram becomes a fork of the lightning, branching into "
         "thinner glowing filaments along the drawn branches. Dark storm sky, "
         "city horizon barely visible. " + NO_DIAGRAM),
    ],
    "nesting": [
        ("walled_city", "concentric", "1024x1024",
         REF + "A photorealistic aerial photograph at dawn of an ancient "
         "walled city: each circle of the diagram is a ring of city wall, one "
         "inside another, with rooftops, trees and lanes filling the space "
         "between the walls, and a small palace at the exact center. Warm "
         "morning light, long shadows. " + NO_DIAGRAM),
    ],
    "cycle": [
        ("ouroboros", "ring", "1024x1024",
         REF + "A classical ink-and-gold painting of an ouroboros: a serpent "
         "whose body follows exactly the circle of the diagram, its mouth "
         "closing on its own tail where the arrows meet, scales rendered in "
         "fine ink lines with muted gold accents on aged paper. " + NO_DIAGRAM),
        ("canal_town", "ring", "1024x1024",
         REF + "A photorealistic aerial photograph of a small old town built "
         "around a perfectly circular canal that follows exactly the circle "
         "of the diagram; boats moving one way around it, little bridges, "
         "tiled roofs, morning mist. " + NO_DIAGRAM),
    ],
    "self_reference": [
        ("droste_study", "frames", "1536x1024",
         REF + "A warm photograph of a quiet wood-panelled study: the "
         "outermost rectangle is the room's back wall, and on that wall hangs "
         "a framed painting — the next rectangle — showing the SAME study "
         "with the same desk and lamp; inside it, the next frame shows the "
         "study again, smaller, and again, receding with each drawn rectangle "
         "(Droste effect). Soft lamplight. " + NO_DIAGRAM),
    ],
    "symmetry": [
        ("butterfly", "mirror", "1536x1024",
         REF + "An exquisite macro photograph of a butterfly with wings fully "
         "spread, photographed exactly head-on: the two drawn polyline halves "
         "become the wing edges and vein patterns, every spot and streak on "
         "the left wing mirrored exactly on the right, the dashed axis is the "
         "slim body — the only unpaired part. Dark velvet background. "
         + NO_DIAGRAM),
        ("temple_facade", "mirror", "1536x1024",
         REF + "A fine woodblock print of a grand temple gate seen exactly "
         "front-on: the two drawn polyline halves become the left and right "
         "rooflines, brackets, pillars and stone lions, perfectly mirrored "
         "about the central doorway on the dashed axis. Warm paper tone, "
         "crisp carved lines. " + NO_DIAGRAM),
    ],
    "alternation": [
        ("lantern_row", "chain", "1536x1024",
         REF + "A photograph at dusk of a long row of paper lanterns strung "
         "between poles along a village street: lanterns hang at the drawn "
         "node positions, strictly alternating — one glowing warm red, the "
         "next plain white paper, red, white — never two alike in a row, the "
         "string dipping and rising along the drawn zigzag. " + NO_DIAGRAM),
    ],
    "spiral": [
        ("nautilus", "logspiral", "1024x1024",
         REF + "A luxurious macro photograph of a nautilus shell cut in "
         "cross-section: the mother-of-pearl septa chambers follow exactly "
         "the drawn spiral, each chamber a fixed ratio larger than the last "
         "as the curve winds outward from the center. Soft studio light on "
         "iridescent nacre, dark background. " + NO_DIAGRAM),
        ("typhoon", "logspiral", "1024x1024",
         REF + "A satellite photograph of a great typhoon over the ocean seen "
         "straight from above: the rain bands coil along exactly the drawn "
         "spiral, each band farther from the calm round eye than the last, "
         "white cloud arms against deep blue sea. " + NO_DIAGRAM),
    ],
    "convergence": [
        ("watershed", "funnel", "1536x1024",
         REF + "A photorealistic aerial photograph of a mountain watershed: "
         "small streams rise separately at the drawn source dots, join in "
         "pairs exactly where the drawn paths merge, growing wider at each "
         "confluence, until one great river pours into a single lake at the "
         "drawn sink. Forested valleys, late afternoon light. " + NO_DIAGRAM),
        ("homing_birds", "funnel", "1536x1024",
         REF + "A long-exposure photograph at dusk: streaks of birds flying "
         "home from many distant points, their light trails following the "
         "drawn merging paths, flocks joining flocks, all converging onto one "
         "single great tree silhouetted at the drawn endpoint. Deep orange "
         "sky. " + NO_DIAGRAM),
    ],
    "bottleneck": [
        ("gorge_bridge", "dumbbell", "1536x1024",
         REF + "A photorealistic aerial photograph of two mountain villages "
         "separated by a deep gorge: each village has its own dense web of "
         "footpaths matching the drawn cluster edges, and the ONLY connection "
         "between the two sides is one narrow stone bridge at the drawn "
         "center node — every drawn line into the center becomes a path "
         "funnelling toward the bridge. Morning haze in the gorge. "
         + NO_DIAGRAM),
    ],
    "symmetry_breaking": [
        ("orchard_anomaly", "field", "1536x1024",
         REF + "A photorealistic aerial photograph of a vast orchard planted "
         "in a perfect grid: identical green trees at every drawn square — "
         "except at the drawn black circle, where a single ancient tree "
         "blazes autumn-red, the one exception in a perfectly regular field. "
         "Even soft light. " + NO_DIAGRAM),
    ],
    "strange_loop": [
        ("impossible_stairs", "ladder", "1024x1024",
         REF + "A photorealistic architectural photograph of a courtyard "
         "staircase built as an impossible figure: each drawn platform is a "
         "landing, every flight between them climbs clearly UPWARD step by "
         "step, yet the drawn return curve closes the last flight back onto "
         "the first landing — an endless ascending loop (Penrose stairs). "
         "Stone steps, dramatic shadows, small figures walking upward "
         "forever. " + NO_DIAGRAM),
    ],
    "figure_ground": [
        ("day_night_birds", "tessellation", "1536x1024",
         REF + "A masterful woodcut print where the drawn interlocking "
         "triangles become interlocking birds: black birds flying left and "
         "white birds flying right, fitted together so perfectly that the "
         "space between the black birds IS the white birds and the space "
         "between the white birds IS the black birds — no unused background "
         "anywhere, figure and ground swapping roles across the sheet. "
         + NO_DIAGRAM),
    ],
    "isomorphism": [
        ("constellations", "twographs", "1536x1024",
         REF + "A deep night-sky photograph with two constellations, one on "
         "the left and one on the right, their stars at the drawn dot "
         "positions and their classical connecting lines following the drawn "
         "solid edges: the two look completely different, yet star for star "
         "they connect in exactly the same pattern, and faint pale traces "
         "along the drawn dashed lines pair each star with its counterpart. "
         "Milky Way behind. " + NO_DIAGRAM),
    ],
    "mobius": [
        ("silk_band", "band", "1536x1024",
         REF + "An elegant studio photograph of a wide silk ribbon given a "
         "single half-twist and joined into a loop, following exactly the "
         "drawn band: the rungs become the weave of the fabric, and at the "
         "twist the shimmering outside face turns seamlessly into the matte "
         "inside face — one continuous single-sided surface. Deep red silk "
         "on dark background, soft light. " + NO_DIAGRAM),
    ],
    "braiding": [
        ("hair_braid", "weave", "1536x1024",
         REF + "A close-up photograph in warm side light of a long braid of "
         "dark hair laid on rough linen: three thick strands weave exactly "
         "along the drawn curves, each crossing OVER then UNDER in turn, and "
         "the order of the strands at the tail end is visibly different from "
         "the order at the root. " + NO_DIAGRAM),
        ("interchange", "weave", "1536x1024",
         REF + "A photorealistic aerial photograph of a great highway "
         "interchange at dusk: three elevated roadways sweep along exactly "
         "the drawn curves, passing over and under one another at each drawn "
         "crossing, headlight trails tracing each road, so the left-to-right "
         "order of the roads at the far side differs from the near side. "
         + NO_DIAGRAM),
    ],
    "interference": [
        ("two_ripples", "ripples", "1536x1024",
         REF + "A photorealistic overhead photograph of a still dark pond a "
         "moment after two stones struck it at the two drawn centers: two "
         "families of concentric ripples spread and pass through each other, "
         "and where the drawn circles intersect, a woven lattice of bright "
         "and calm bands appears — a larger pattern neither stone made "
         "alone. Soft dawn light. " + NO_DIAGRAM),
    ],
    "tiling": [
        ("honeycomb", "honeycomb", "1536x1024",
         REF + "A luminous macro photograph of a natural honeycomb backlit "
         "with golden light: perfect hexagonal wax cells at exactly the drawn "
         "lattice, identical cells repeating without gap or overlap to every "
         "edge of the frame, a few cells capped, one bee at work. "
         + NO_DIAGRAM),
    ],
    "hub": [
        ("firework", "star", "1024x1024",
         REF + "A long-exposure photograph of a single great firework at the "
         "moment of its burst: glowing trails shoot from one blinding center "
         "outward along exactly the drawn spokes, each trail ending in a "
         "bright head at the drawn outer dots, no trail touching another. "
         "Night sky over water with reflections. " + NO_DIAGRAM),
    ],
    "diagonalization": [
        ("jacquard_loom", "grid", "1024x1024",
         REF + "A photorealistic straight-on photograph of a small jacquard "
         "sampler on a loom, woven as a square grid of large plain squares, "
         "dark or cream, one square per drawn cell, EXACTLY matching the drawn "
         "6x6 pattern. A single RED thread is woven along the main diagonal, "
         "boxing the six diagonal squares. BELOW the square cloth, separated "
         "by a finger of bare warp threads, hangs ONE extra woven strip of six "
         "squares bound in red selvedge: it is the diagonal row with every "
         "colour flipped (cream where the diagonal square was dark, dark where "
         "it was cream), so this strip matches no row of the cloth above. "
         "Warm workshop light. " + NO_DIAGRAM),
    ],
    "infinite_descent": [
        ("nesting_dolls", "staircase", "1536x1024",
         REF + "A photorealistic overhead photograph of a row of Russian "
         "nesting dolls opened and lined up largest to smallest along exactly "
         "the drawn descending steps, each strictly smaller than the one "
         "before. At the drawn hard floor the smallest doll is opened to reveal "
         "it is empty — yet still demanding a smaller doll that cannot exist; a "
         "faint red cross marks the impossible next step. Soft museum light. "
         + NO_DIAGRAM),
    ],
    "coprime_orbit": [
        ("string_art_star", "star", "1024x1024",
         REF + "A photorealistic close-up of string art on a dark wooden board: "
         "brass nails driven in a circle at exactly the drawn outer points, and "
         "a single continuous bright thread that leaps over two nails to the "
         "third each time, tracing the drawn eight-point star in one unbroken "
         "path that visits every nail before closing. Raking light, real "
         "thread shadows. " + NO_DIAGRAM),
    ],
    "pigeonhole": [
        ("swallow_nests", "boxes", "1536x1024",
         REF + "A photorealistic dusk photograph under a wooden eave: exactly "
         "four swallow nests at the drawn boxes and five returning swallows at "
         "the drawn upper points, each flying down its drawn path into a nest. "
         "Four birds fill three nests one apiece; the highlighted nest is "
         "forced to hold two swallows together — more birds than nests, so one "
         "nest must double. Golden backlight, soft silhouettes. " + NO_DIAGRAM),
    ],
    "projective_duality": [
        ("spiderweb_dewfall", "quadrangle", "1536x1024",
         REF + "A photorealistic diptych of the SAME spider web. LEFT: dry, the "
         "silk threads are thick and bright as the main subject (the drawn fat "
         "lines) and the knots where they cross are faint (the drawn small "
         "points). RIGHT: the same web heavy with dawn dew, now the dewdrops at "
         "the crossings swell into a bright array of beads (points become the "
         "subject) while the threads recede to thin guides — points and lines "
         "have swapped roles, yet every 'which thread meets which' relation is "
         "preserved. Cool morning light. " + NO_DIAGRAM),
    ],
    "contraction_fixed_point": [
        ("you_are_here", "cobweb", "1024x1024",
         REF + "A photorealistic close-up of a polished bronze city map inlaid "
         "in a plaza floor, with a single brass stud reading nothing but "
         "marking one point. The stud sits exactly where the map depicts its "
         "own location — the shrunken city pressed onto the real city pins one "
         "unmoving point. Let the drawn spiral of the cobweb guide the eye "
         "inward to that fixed stud. Warm afternoon light, worn metal. "
         + NO_DIAGRAM),
    ],
    "aperiodic_tiling": [
        ("penrose_floor", "penrose", "1024x1024",
         REF + "A photorealistic overhead photograph of a stone-tiled plaza "
         "floor laid in a Penrose pattern of two rhombus tiles (fat and thin) "
         "arranged exactly as the drawn 'sun' patch: five-fold symmetric about "
         "the center, tiles meeting edge to edge with no gaps, the pattern "
         "clearly never repeating as it radiates outward. A single person "
         "stands at the center. Overcast even light, real stone texture. "
         + NO_DIAGRAM),
    ],
}

# scenes with no skeleton (pure text-to-image)
T2I_SCENES = {
    "recursion": [
        ("winter_tree", "1024x1024",
         "A striking photograph of a single leafless tree against a pale sky "
         "in winter, seen so its branching is fully visible. The trunk splits "
         "into a few boughs; each bough splits the same way into smaller "
         "boughs; each of those into finer twigs — the identical branching "
         "pattern repeating at every scale, clearly self-similar and fractal. "
         "Elegant, high-contrast. No text, no words, no letters."),
    ],
    "nesting": [
        ("pond_ripples", "1024x1024",
         "A photograph-like, tranquil overhead view of a still dark pond at "
         "dawn where a single raindrop has just struck the center. Perfectly "
         "concentric circular ripples spread outward — many evenly spaced "
         "rings, each one a larger copy of the ring inside it, nested from a "
         "tiny bright point at the center out to the edges. Soft morning "
         "light, calm and meditative, photorealistic water surface. No text, "
         "no words, no letters."),
        ("moon_gates", "1024x1024",
         "A photograph looking straight down the axis of a classical Chinese "
         "garden: a sequence of round moon gates, one behind another, each "
         "framing the next smaller gate within it, receding layer inside "
         "layer into soft darkness — you may enter ever deeper but the "
         "deepest court is never seen. Grey brick, moss, late light. No "
         "text, no words, no letters."),
    ],
    "self_reference": [
        ("mountain_temple", "1024x1024",
         "A serene traditional Chinese ink-wash and light-color landscape "
         "painting. A misty mountain; on it a small temple; inside, through "
         "the open door, an old monk sits telling a story to a young monk. "
         "On the wall behind them hangs a painting — and that painting shows "
         "the SAME mountain, the SAME temple, the SAME two monks, smaller; "
         "and inside THAT scene, another painting of the same scene, smaller "
         "again — a picture within a picture, receding infinitely toward the "
         "center (mise en abyme, Droste effect). Exactly three clearly "
         "visible nested levels before it fades into mist. Muted ink tones "
         "with soft gold light. No text, no words, no letters."),
    ],
    "alternation": [
        ("zebra", "1536x1024",
         "A high-contrast photograph filling the whole frame with the flank "
         "of a zebra: bold black and white stripes in strict alternation, "
         "black white black white with never two alike together, gently "
         "curving with the body. No text, no words, no letters."),
    ],
    "symmetry_breaking": [
        ("red_plum", "1024x1024",
         "A fine photograph of a tree covered in thousands of identical "
         "white plum blossoms — and exactly ONE blossom, near the heart of "
         "the frame, is deep red. Everything else repeats the same rule; the "
         "single red bloom is the only exception and the eye goes straight "
         "to it. Soft grey sky. No text, no words, no letters."),
    ],
    "strange_loop": [
        ("drawing_hands", "1536x1024",
         "A warm photograph staged on a wooden desk: two real human hands, "
         "each holding a pencil, each drawing the OTHER hand on the same "
         "sheet of paper — the drawn hands merging seamlessly into the real "
         "hands so it is impossible to say which hand is the drawer and "
         "which is the drawing. Lamplight, graphite shading half-becoming "
         "photograph. No text, no words, no letters."),
    ],
    "figure_ground": [
        ("papercut", "1024x1024",
         "A photograph of two sheets of paper side by side on a dark table: "
         "on the left, an intricate red paper-cut of blossoms and birds; on "
         "the right, the white sheet it was cut FROM — the holes in one are "
         "exactly the figures of the other, each sheet a complete picture "
         "made of the other's absence. Soft raking light. No text, no words, "
         "no letters."),
    ],
    "isomorphism": [
        ("tree_and_roots", "1024x1024",
         "A traditional ink-wash painting split by the ground line: above, a "
         "great tree's crown branching into the sky; below, its root system "
         "branching into the earth — and the two branchings correspond limb "
         "for limb, fork for fork, the same structure spoken twice in two "
         "different worlds, one in air, one in soil. Muted earth tones. No "
         "text, no words, no letters."),
    ],
    "mobius": [
        ("ant_band", "1024x1024",
         "A crisp studio photograph of a wide paper band given a half-twist "
         "and glued into a loop, with a line of small red ants marching "
         "single file along its surface: the column passes the twist and "
         "continues seamlessly onto what seems the other side — because "
         "there is only one side. Neutral grey background, sharp detail. No "
         "text, no words, no letters."),
    ],
    "interference": [
        ("moire_curtains", "1024x1024",
         "A photograph of a bright window hung with two layers of fine mesh "
         "curtain, slightly offset: where the two regular weaves overlap, "
         "broad soft dark and light moiré bands ripple across the fabric — "
         "a large slow pattern that neither curtain contains alone. Backlit, "
         "airy. No text, no words, no letters."),
    ],
    "tiling": [
        ("salt_pans", "1536x1024",
         "A photorealistic aerial photograph of coastal salt evaporation "
         "pans: identical rectangular pools repeating row after row in a "
         "perfect grid to every edge of the frame, thin white ridges between "
         "them, subtle pastel differences in the brine. Shift the whole "
         "picture by one pool and it would land on itself. No text, no "
         "words, no letters."),
    ],
    "hub": [
        ("starfish", "1024x1024",
         "A crisp overhead macro photograph of a single orange starfish "
         "centered on dark wet sand at low tide: five arms radiating from "
         "one central disc, arms never touching one another, fine ripple "
         "texture around it. No text, no words, no letters."),
    ],
    "braiding": [],
    "convergence": [],
    "bottleneck": [
        ("hourglass", "1024x1024",
         "A dramatic macro photograph of an hourglass mid-flow against a "
         "dark background: rich dunes of sand above and below, and between "
         "them the single narrow throat through which every falling grain "
         "must pass, one thin bright thread of sand connecting the two "
         "worlds. No text, no words, no letters."),
    ],
    "cycle": [],
    "symmetry": [],
    "spiral": [],
}


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def gpt_edit(ref: Path, prompt: str, size: str, out: Path) -> bool:
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/images/edits",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-F", f"model={MODEL}", "-F", f"image[]=@{ref}",
           "-F", f"prompt={prompt}", "-F", f"size={size}", "-F", "quality=high"]
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  retry {out.name} (attempt {attempt + 1})")
            continue
        if "data" in resp:
            out.write_bytes(base64.b64decode(resp["data"][0]["b64_json"]))
            return True
        print(f"  FAIL {out.name}: {json.dumps(resp)[:200]}")
        return False
    return False


def gpt_t2i(prompt: str, size: str, out: Path) -> bool:
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           f"{os.environ['OPENAI_BASE_URL']}/images/generations",
           "-H", f"Authorization: Bearer {os.environ['OPENAI_API_KEY']}",
           "-H", "Content-Type: application/json",
           "-d", json.dumps({"model": MODEL, "prompt": prompt,
                             "size": size, "quality": "high", "n": 1})]
    for attempt in range(3):
        raw = subprocess.run(cmd, capture_output=True, text=True).stdout
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  retry {out.name} (attempt {attempt + 1})")
            continue
        if "data" in resp:
            item = resp["data"][0]
            if item.get("b64_json"):
                out.write_bytes(base64.b64decode(item["b64_json"]))
            else:
                subprocess.run(["curl", "-s", "-o", str(out), item["url"]], check=True)
            return True
        print(f"  FAIL {out.name}: {json.dumps(resp)[:200]}")
        return False
    return False


def build_motif(mid: str) -> None:
    lib = LIBRARY[mid]
    d = OUT / mid
    d.mkdir(parents=True, exist_ok=True)

    (d / "schema.json").write_text(json.dumps({
        "motif_id": mid,
        "name": lib["name_en"],
        "name_zh": lib["name_zh"],
        "equation": lib["equation_katex"],
        "invariant": lib["invariant_zh"],
        "math_home": lib["math_home"],
        "story_form_device": lib["story_form_device"],
        "geb_echo": lib["geb_echo"],
        "nuisance_variables": ["node count", "depth", "angles", "layout",
                               "scale", "domain", "medium"],
    }, indent=2, ensure_ascii=False))
    (d / "math.md").write_text(
        f"# {lib['name_zh']} — mathematics voice\n\n"
        f"$$" + lib["equation_katex"] + "$$\n\n"
        f"**定义** {lib['math_formal']}\n\n"
        f"**归属** {lib['math_home']}\n\n"
        f"**定理** {lib['math_theorem']}\n")

    # skeletons
    for name, render in SKELETONS.get(mid, {}).items():
        svg = d / f"skeleton_{name}.svg"
        png = d / f"skeleton_{name}.png"
        svg.write_text(render())
        subprocess.run(["rsvg-convert", str(svg), "-o", str(png)], check=True)

    # scenes from skeleton refs
    for scene, skel, size, prompt in SCENES.get(mid, []):
        out = d / f"scene_{scene}.png"
        if out.exists():
            print(f"  skip {mid}/{out.name}")
            continue
        if gpt_edit(d / f"skeleton_{skel}.png", prompt, size, out):
            print(f"  ok {mid}/{out.name}")

    # scenes with no skeleton (pure text-to-image)
    for scene, size, prompt in T2I_SCENES.get(mid, []):
        out = d / f"scene_{scene}.png"
        if out.exists():
            print(f"  skip {mid}/{out.name}")
            continue
        if gpt_t2i(prompt, size, out):
            print(f"  ok {mid}/{out.name}")


def main():
    ids = sys.argv[1:] or list(LIBRARY)
    for mid in ids:
        build_motif(mid)


if __name__ == "__main__":
    main()
