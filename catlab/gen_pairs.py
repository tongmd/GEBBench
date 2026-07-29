#!/usr/bin/env python3
"""Functorial-abstraction pilot: matched skeleton pairs (x, g(x)).  v2.

Design contract (tightened after the adversarial visual audit of v1):
  * every generator takes (seed, transformed, variant)
  * STRUCTURAL parameters (node count, depth, k, tile count) come from
    fixed per-seed tables -> all six seeds are structurally distinct where
    the motif allows, and NEVER collide by rng accident
  * CONTINUOUS nuisance (jitter, rotation phase, ratios, offsets) comes from
    a per-(edge,seed,variant) rng -> no byte-identical images
  * the nuisance twin of a figure is fn(seed, transformed, variant=1):
    SAME structural parameters, different continuous nuisance only; this is
    asserted via the facts dict
  * the audit's geometry bugs are fixed: the broken-symmetry displacement is
    clamped inside the panel, recursion child boxes keep a visible gap,
    dictionary dashes are curved so pairings are explicit

Output:
  catlab/pairs/<edge>_s<seed>_{base,img,ref,nuis}.{svg,png}
  catlab/pairs/comp_s<seed>.png
  catlab/pairs/manifest.json
"""
from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pairs"
INK = "#1a1a1a"
ACC = "#b3402a"

W, H = 1024, 1024
N_SEEDS = 6


def R(edge: str, seed: int, variant: int) -> random.Random:
    return random.Random(f"{edge}|{seed}|{variant}")


def svg_doc(w: int, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
            f'<rect width="100%" height="100%" fill="#ffffff"/>{body}</svg>')


def arrow_defs(color: str = INK, name: str = "a") -> str:
    return (f'<defs><marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker></defs>')


# --------------------------------------------------------------------------
# edge 1: symmetry -> symmetry_breaking
# --------------------------------------------------------------------------
SYM_N = [6, 7, 8, 6, 7, 8]           # structural: nodes per side


def e_symmetry(seed: int, broken: bool, variant: int = 0):
    rng = R("sym", seed, variant)
    n = SYM_N[seed]
    pts, y = [], 170
    for _ in range(n):
        pts.append((rng.randint(90, 390), y + rng.randint(-25, 25)))
        y += (H - 340) / (n - 1)
    parts = [f'<line x1="{W/2}" y1="70" x2="{W/2}" y2="{H-70}" stroke="{INK}" '
             'stroke-width="2" stroke-dasharray="10 12"/>']
    kbreak = 1 + rng.randrange(n - 2)
    n_asym = 0
    for sgn in (-1, 1):
        coords = []
        for i, (x, yy) in enumerate(pts):
            px, py = W / 2 + sgn * x, yy
            if broken and sgn == 1 and i == kbreak:
                # displace toward the panel interior; clamp inside canvas
                dx = 150 if px + 150 <= W - 70 else -150
                py2 = min(H - 70, py + 60)
                px, py = px + dx, py2
                n_asym += 1
            coords.append((px, py))
        d = " ".join(f"{'M' if i == 0 else 'L'} {x:.0f} {y:.0f}"
                     for i, (x, y) in enumerate(coords))
        parts.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="6" '
                     'stroke-linejoin="round"/>')
        for i, (x, y) in enumerate(coords):
            if broken and sgn == 1 and i == kbreak:
                parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="16" '
                             f'fill="{ACC}"/>')
            else:
                parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="12" '
                             f'fill="{INK}"/>')
    facts = {"n_nodes_per_side": n, "n_asymmetric_nodes": n_asym}
    assert n_asym == (1 if broken else 0)
    return svg_doc(W, H, "".join(parts)), facts


# --------------------------------------------------------------------------
# edge 2: cycle -> strange_loop
# --------------------------------------------------------------------------
CYC_N = [6, 7, 8, 9, 7, 8]           # structural: nodes


def e_cycle(seed: int, looped: bool, variant: int = 0):
    rng = R("cyc", seed, variant)
    n = CYC_N[seed]
    phase = rng.uniform(0, 2 * math.pi)
    parts = [arrow_defs(), arrow_defs(ACC, "b")]
    if not looped:
        cx, cy = W / 2 + rng.randint(-30, 30), H / 2 + rng.randint(-30, 30)
        r = min(W, H) / 2 - 150 - rng.randint(0, 60)
        pos = [(cx + r * math.cos(2 * math.pi * k / n + phase),
                cy + r * math.sin(2 * math.pi * k / n + phase))
               for k in range(n)]
        tiers = [0] * n
    else:
        margin = 170 + rng.randint(-30, 30)
        top, bot = 150 + rng.randint(0, 40), H - 150 - rng.randint(0, 40)
        wig = [rng.randint(-45, 45) for _ in range(n)]
        pos = [(margin + (W - 2 * margin) * k / (n - 1) + wig[k],
                bot - (bot - top) * k / (n - 1)) for k in range(n)]
        tiers = list(range(n))
        for k in range(n):
            yy = bot - (bot - top) * k / (n - 1)
            parts.append(f'<line x1="90" y1="{yy:.0f}" x2="{W-90}" y2="{yy:.0f}" '
                         f'stroke="#bbbbbb" stroke-width="1.5" '
                         'stroke-dasharray="6 10"/>')
    for k in range(n):
        a, b = pos[k], pos[(k + 1) % n]
        closing = (k == n - 1)
        col = ACC if (looped and closing) else INK
        mk = "b" if (looped and closing) else "a"
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        L = math.hypot(vx, vy) or 1
        ax, ay = ax + vx / L * 34, ay + vy / L * 34
        bx, by = bx - vx / L * 34, by - vy / L * 34
        if looped and closing:
            parts.append(f'<path d="M {ax:.0f} {ay:.0f} Q {W-50} {H/2:.0f} '
                         f'{bx:.0f} {by:.0f}" fill="none" stroke="{col}" '
                         f'stroke-width="5" marker-end="url(#{mk})"/>')
        elif not looped:
            r = math.hypot(a[0] - (pos[0][0] + pos[n // 2][0]) / 2,
                           a[1] - (pos[0][1] + pos[n // 2][1]) / 2)
            parts.append(f'<path d="M {ax:.0f} {ay:.0f} A {r:.0f} {r:.0f} 0 0 1 '
                         f'{bx:.0f} {by:.0f}" fill="none" stroke="{col}" '
                         f'stroke-width="5" marker-end="url(#{mk})"/>')
        else:
            parts.append(f'<line x1="{ax:.0f}" y1="{ay:.0f}" x2="{bx:.0f}" '
                         f'y2="{by:.0f}" stroke="{col}" stroke-width="5" '
                         f'marker-end="url(#{mk})"/>')
    for x, y in pos:
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="28" fill="#ffffff" '
                     f'stroke="{INK}" stroke-width="5"/>')
    facts = {"n_nodes": n, "n_edges": n, "n_levels": len(set(tiers)),
             "consistent_level_assignment": not looped}
    return svg_doc(W, H, "".join(parts)), facts


# --------------------------------------------------------------------------
# edge 3: nesting -> recursion
# --------------------------------------------------------------------------
NEST_D = [3, 4, 3, 4, 3, 4]          # structural: depth


def e_nesting(seed: int, recursive: bool, variant: int = 0):
    rng = R("nest", seed, variant)
    depth = NEST_D[seed]
    q = rng.uniform(0.58, 0.70)
    corners = rng.choice([("tl", "br"), ("tr", "bl")])
    parts = []

    def place(x, y, w, h, corner, nw, nh):
        pad_x, pad_y = w * 0.05, h * 0.05
        if corner == "tl":
            return x + pad_x, y + pad_y
        if corner == "tr":
            return x + w - nw - pad_x, y + pad_y
        if corner == "bl":
            return x + pad_x, y + h - nh - pad_y
        return x + w - nw - pad_x, y + h - nh - pad_y

    def box(x, y, w, h, level):
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                     f'height="{h:.1f}" fill="none" stroke="{INK}" '
                     f'stroke-width="{max(2, 6 - level)}"/>')
        if level >= depth:
            return
        if recursive:
            nw, nh = w * 0.42, h * 0.42     # 0.42 guarantees a visible gap
            for c in corners:
                px, py = place(x, y, w, h, c, nw, nh)
                box(px, py, nw, nh, level + 1)
        else:
            nw, nh = w * q, h * q
            jx = (w - nw) * rng.uniform(0.3, 0.7)
            jy = (h - nh) * rng.uniform(0.3, 0.7)
            box(x + jx, y + jy, nw, nh, level + 1)

    m = 70 + rng.randint(0, 40)
    box(m, m, W - 2 * m, H - 2 * m, 0)
    facts = {"depth": depth, "branching_factor": 2 if recursive else 1,
             "self_similar_rule": recursive}
    return svg_doc(W, H, "".join(parts)), facts


# --------------------------------------------------------------------------
# edge 4: tiling -> aperiodic_tiling
# --------------------------------------------------------------------------
TILE_N = [12, 13, 14, 15, 20, 21]   # structural: tiles per row
# (Fibonacci-word prefixes of length 16-19 have period 8; these six lengths
#  are verified period-free by the assertion below.)


def fib_word(n: int) -> str:
    s = "L"
    while len(s) < n:
        s = "".join("LS" if c == "L" else "L" for c in s)
    return s[:n]


def has_period(seq: str) -> int | None:
    for p in range(1, len(seq) // 2 + 1):
        if all(seq[i] == seq[i + p] for i in range(len(seq) - p)):
            return p
    return None


def e_tiling(seed: int, aperiodic: bool, variant: int = 0):
    rng = R("tile", seed, variant)
    n_tiles = TILE_N[seed]
    if aperiodic:
        seq = fib_word(n_tiles)
    else:
        unit = rng.choice(["LS", "SL"])          # nuisance: phase of the unit
        seq = (unit * n_tiles)[:n_tiles]
    wide, narrow = 118.0, 73.0
    widths = [wide if c == "L" else narrow for c in seq]
    scale = (W - 120) / sum(widths)
    widths = [w * scale for w in widths]
    rows = rng.choice([3, 4])
    parts = []
    band_h = rng.choice([150, 170])
    y0 = (H - rows * band_h) / 2
    for r in range(rows):
        x = 60 + rng.randint(0, 26)
        y = y0 + r * band_h
        for i, wdt in enumerate(widths):
            if x + wdt > W - 40:
                break
            fill = "#ffffff" if seq[i] == "L" else "#e8e2d8"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{wdt:.1f}" '
                         f'height="{band_h-14:.1f}" fill="{fill}" stroke="{INK}" '
                         'stroke-width="4"/>')
            parts.append(f'<circle cx="{x+wdt/2:.1f}" cy="{y+(band_h-14)/2:.1f}" '
                         f'r="9" fill="{INK}"/>')
            x += wdt
    p = has_period(seq)
    facts = {"sequence": seq, "n_tiles": n_tiles, "translation_period": p,
             "tile_alphabet": sorted(set(seq))}
    if aperiodic:
        assert p is None
    else:
        assert p is not None
    return svg_doc(W, H, "".join(parts)), facts


# --------------------------------------------------------------------------
# edge 5: isomorphism -> projective_duality
# --------------------------------------------------------------------------
ISO_K = [4, 5, 6, 4, 5, 6]           # structural: elements per class


def e_isomorphism(seed: int, dual: bool, variant: int = 0):
    rng = R("iso", seed, variant)
    k = ISO_K[seed]
    phase = rng.uniform(0, 2 * math.pi)
    parts = [arrow_defs(ACC, "b")]
    if not dual:
        rr = 200 + rng.randint(0, 40)
        for s, (cx, tag) in enumerate([(255, "circle"), (W - 255, "square")]):
            cy = H / 2 + rng.randint(-25, 25)
            pos = [(cx + rr * math.cos(2 * math.pi * i / k + phase) * 0.8,
                    cy + rr * math.sin(2 * math.pi * i / k + phase) * 0.8)
                   for i in range(k)]
            for i in range(k):
                for j in range(i + 1, k):
                    parts.append(f'<line x1="{pos[i][0]:.0f}" y1="{pos[i][1]:.0f}" '
                                 f'x2="{pos[j][0]:.0f}" y2="{pos[j][1]:.0f}" '
                                 f'stroke="{INK}" stroke-width="2.5"/>')
            for x, y in pos:
                if tag == "circle":
                    parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="22" '
                                 f'fill="#ffffff" stroke="{INK}" stroke-width="5"/>')
                else:
                    parts.append(f'<rect x="{x-20:.0f}" y="{y-20:.0f}" width="40" '
                                 f'height="40" fill="#ffffff" stroke="{INK}" '
                                 'stroke-width="5"/>')
            if s == 0:
                left = pos
            else:
                right = pos
        for i in range(k):
            # curved dictionary dash so each pairing is visually explicit
            mx = (left[i][0] + right[i][0]) / 2
            my = (left[i][1] + right[i][1]) / 2 - 90 - 26 * i
            parts.append(f'<path d="M {left[i][0]:.0f} {left[i][1]:.0f} '
                         f'Q {mx:.0f} {my:.0f} {right[i][0]:.0f} '
                         f'{right[i][1]:.0f}" fill="none" '
                         f'stroke="{ACC}" stroke-width="2" '
                         'stroke-dasharray="9 9"/>')
        n_systems, involutive = 2, False
    else:
        cx, cy = W / 2, H / 2 + rng.randint(0, 40)
        r = 280 + rng.randint(0, 40)
        pos = [(cx + r * math.cos(2 * math.pi * i / k + phase),
                cy + r * math.sin(2 * math.pi * i / k + phase))
               for i in range(k)]
        mid = []
        for i in range(k):
            j = (i + 1) % k
            parts.append(f'<line x1="{pos[i][0]:.0f}" y1="{pos[i][1]:.0f}" '
                         f'x2="{pos[j][0]:.0f}" y2="{pos[j][1]:.0f}" '
                         f'stroke="{INK}" stroke-width="3"/>')
            mid.append(((pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2))
        for x, y in pos:
            parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="22" fill="#ffffff" '
                         f'stroke="{INK}" stroke-width="5"/>')
        for x, y in mid:
            parts.append(f'<rect x="{x-19:.0f}" y="{y-19:.0f}" width="38" '
                         f'height="38" fill="#ffffff" stroke="{INK}" '
                         f'stroke-width="5" transform="rotate(45 {x:.0f} '
                         f'{y:.0f})"/>')
        for i in range(k):
            x1, y1 = pos[i]
            x2, y2 = mid[i]
            ux, uy = (x1 + x2) / 2, (y1 + y2) / 2
            nx, ny = -(y2 - y1), (x2 - x1)
            L = math.hypot(nx, ny) or 1
            ox, oy = ux + nx / L * 30, uy + ny / L * 30
            ix, iy = ux - nx / L * 30, uy - ny / L * 30
            parts.append(f'<path d="M {x1:.0f} {y1:.0f} Q {ox:.0f} {oy:.0f} '
                         f'{x2:.0f} {y2:.0f}" fill="none" stroke="{ACC}" '
                         'stroke-width="2.5" stroke-dasharray="8 8" '
                         'marker-end="url(#b)"/>')
            parts.append(f'<path d="M {x2:.0f} {y2:.0f} Q {ix:.0f} {iy:.0f} '
                         f'{x1:.0f} {y1:.0f}" fill="none" stroke="{ACC}" '
                         'stroke-width="2.5" stroke-dasharray="8 8" '
                         'marker-end="url(#b)"/>')
        n_systems, involutive = 1, True
    facts = {"n_elements_per_class": k, "n_systems": n_systems,
             "swap_is_involution": involutive}
    return svg_doc(W, H, "".join(parts)), facts


# --------------------------------------------------------------------------
# edge 6: recursion -> contraction_fixed_point
# --------------------------------------------------------------------------
REC_D = [5, 6, 5, 6, 5, 6]           # structural: depth


def e_recursion(seed: int, contract: bool, variant: int = 0):
    rng = R("rec", seed, variant)
    depth = REC_D[seed]
    q = rng.uniform(0.60, 0.68)
    ang = rng.uniform(26, 34)
    parts = []
    endpoints = []

    def branch(x, y, heading, ln, level):
        x2 = x + ln * math.cos(math.radians(heading))
        y2 = y + ln * math.sin(math.radians(heading))
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{INK}" stroke-width="{max(2, 8 - 1.2 * level):.1f}" '
                     'stroke-linecap="round"/>')
        if level >= depth:
            endpoints.append((x2, y2))
            return
        if contract:
            branch(x2, y2, heading + ang, ln * q, level + 1)
        else:
            branch(x2, y2, heading - ang, ln * q, level + 1)
            branch(x2, y2, heading + ang, ln * q, level + 1)

    if contract:
        branch(220 + rng.randint(-40, 40), H - 200, -58 + rng.uniform(-6, 6),
               330, 0)
    else:
        branch(W / 2 + rng.randint(-40, 40), H - 120, -90, 300, 0)
    if contract:
        fx, fy = endpoints[-1]
        parts.append(f'<circle cx="{fx:.0f}" cy="{fy:.0f}" r="15" fill="{ACC}"/>')
        parts.append(f'<circle cx="{fx:.0f}" cy="{fy:.0f}" r="34" fill="none" '
                     f'stroke="{ACC}" stroke-width="3" stroke-dasharray="6 7"/>')
    facts = {"depth": depth, "branching_factor": 1 if contract else 2,
             "shrink_ratio": round(q, 3), "n_endpoints": len(endpoints),
             "has_marked_fixed_point": contract}
    return svg_doc(W, H, "".join(parts)), facts


# --------------------------------------------------------------------------
EDGES = {
    "sym_to_symbreak": {
        "src": "symmetry", "dst": "symmetry_breaking", "fn": e_symmetry,
        "change": "exactly one of the mirrored correspondences is destroyed",
    },
    "cycle_to_strangeloop": {
        "src": "cycle", "dst": "strange_loop", "fn": e_cycle,
        "change": "the flat ring is re-laid across levels, so following the "
                  "arrows climbs a hierarchy and still returns to the start",
    },
    "nesting_to_recursion": {
        "src": "nesting", "dst": "recursion", "fn": e_nesting,
        "change": "containment becomes generation: each part is replaced by "
                  "smaller copies of the whole under one rule",
    },
    "tiling_to_aperiodic": {
        "src": "tiling", "dst": "aperiodic_tiling", "fn": e_tiling,
        "change": "the same two tiles obey the same local rule but the "
                  "translation period is destroyed",
    },
    "iso_to_duality": {
        "src": "isomorphism", "dst": "projective_duality", "fn": e_isomorphism,
        "change": "a dictionary between two separate systems becomes an "
                  "involutive role swap inside one system",
    },
    "recursion_to_contraction": {
        "src": "recursion", "dst": "contraction_fixed_point", "fn": e_recursion,
        "change": "branching stops: one rule is applied to its own output and "
                  "the orbit selects a unique fixed point",
    },
}

STRUCT_KEYS = {          # facts that must AGREE between nuisance twins
    "sym_to_symbreak": ["n_nodes_per_side", "n_asymmetric_nodes"],
    "cycle_to_strangeloop": ["n_nodes", "n_edges", "n_levels"],
    "nesting_to_recursion": ["depth", "branching_factor"],
    "tiling_to_aperiodic": ["n_tiles", "tile_alphabet"],
    "iso_to_duality": ["n_elements_per_class", "n_systems"],
    "recursion_to_contraction": ["depth", "branching_factor"],
}


def render(svg: str, stem: Path, width: int = 760):
    stem.with_suffix(".svg").write_text(svg)
    subprocess.run(["rsvg-convert", str(stem.with_suffix(".svg")),
                    "-w", str(width), "-o", str(stem.with_suffix(".png"))],
                   check=True)


def side_by_side(svg_a: str, svg_b: str) -> str:
    def inner(s):
        marker = 'fill="#ffffff"/>'
        return s.split(marker, 1)[1].rsplit("</svg>", 1)[0]
    gap, lab = 130, 66
    tw, th = W * 2 + gap, H + lab
    body = ['<rect width="100%" height="100%" fill="#ffffff"/>']
    body.append(f'<g transform="translate(0,{lab})">'
                f'<rect x="2" y="2" width="{W-4}" height="{H-4}" fill="none" '
                f'stroke="#cccccc" stroke-width="3"/>{inner(svg_a)}</g>')
    body.append(f'<g transform="translate({W+gap},{lab})">'
                f'<rect x="2" y="2" width="{W-4}" height="{H-4}" fill="none" '
                f'stroke="#cccccc" stroke-width="3"/>{inner(svg_b)}</g>')
    cy = lab + H / 2
    body.append(f'<line x1="{W+18}" y1="{cy}" x2="{W+gap-18}" y2="{cy}" '
                f'stroke="{ACC}" stroke-width="9" marker-end="url(#big)"/>')
    body.append('<defs><marker id="big" viewBox="0 0 10 10" refX="8" refY="5" '
                'markerWidth="6" markerHeight="6" orient="auto">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ACC}"/></marker></defs>')
    body.append(f'<text x="{W/2}" y="46" font-family="Helvetica,Arial" '
                f'font-size="40" fill="{INK}" text-anchor="middle">BEFORE</text>')
    body.append(f'<text x="{W+gap+W/2}" y="46" font-family="Helvetica,Arial" '
                f'font-size="40" fill="{INK}" text-anchor="middle">AFTER</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{tw}" height="{th}" '
            f'viewBox="0 0 {tw} {th}">' + "".join(body) + "</svg>")


def stack_two(svg_a: str, svg_b: str) -> str:
    def guts(s):
        head, rest = s.split('viewBox="0 0 ', 1)
        dims, body = rest.split('">', 1)
        w, h = [float(v) for v in dims.split()[0:2]]
        return w, h, body.rsplit("</svg>", 1)[0]
    wa, ha, ba = guts(svg_a)
    wb, hb, bb = guts(svg_b)
    lab = 60
    tw, th = max(wa, wb), ha + hb + lab * 2
    out = ['<rect width="100%" height="100%" fill="#ffffff"/>']
    out.append(f'<text x="30" y="44" font-family="Helvetica,Arial" '
               f'font-size="42" fill="{ACC}">STEP 1</text>')
    out.append(f'<g transform="translate(0,{lab})">{ba}</g>')
    out.append(f'<text x="30" y="{ha+lab*2-16:.0f}" font-family="Helvetica,Arial" '
               f'font-size="42" fill="{ACC}">STEP 2</text>')
    out.append(f'<g transform="translate(0,{ha+lab*2:.0f})">{bb}</g>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{tw:.0f}" '
            f'height="{th:.0f}" viewBox="0 0 {tw:.0f} {th:.0f}">'
            + "".join(out) + "</svg>")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    refs = {}
    seen_hashes = {}
    import hashlib
    for ekey, spec in EDGES.items():
        for s in range(N_SEEDS):
            svg_a, facts_a = spec["fn"](s, False, 0)
            svg_b, facts_b = spec["fn"](s, True, 0)
            svg_a2, facts_a2 = spec["fn"](s, False, 1)   # nuisance twin
            # twin must agree on every structural fact
            for kf in STRUCT_KEYS[ekey]:
                assert facts_a[kf] == facts_a2[kf], (ekey, s, kf)
            # no byte-identical twin, no cross-seed duplicates
            for name, svg in (("base", svg_a), ("img", svg_b), ("twin", svg_a2)):
                hsh = hashlib.md5(svg.encode()).hexdigest()
                assert (hsh not in seen_hashes), \
                    (ekey, s, name, "duplicate of", seen_hashes.get(hsh))
                seen_hashes[hsh] = (ekey, s, name)
            assert svg_a != svg_a2, (ekey, s, "twin identical")

            stem_a = OUT / f"{ekey}_s{s}_base"
            stem_b = OUT / f"{ekey}_s{s}_img"
            stem_r = OUT / f"{ekey}_s{s}_ref"
            stem_n = OUT / f"{ekey}_s{s}_nuis"
            render(svg_a, stem_a)
            render(svg_b, stem_b)
            ref = side_by_side(svg_a, svg_b)
            refs[(ekey, s)] = ref
            render(ref, stem_r, 1200)
            render(side_by_side(svg_a, svg_a2), stem_n, 1200)
            manifest.append({
                "edge": ekey, "seed": s,
                "src_motif": spec["src"], "dst_motif": spec["dst"],
                "change": spec["change"],
                "base_png": str(stem_a.with_suffix(".png").relative_to(ROOT.parent)),
                "img_png": str(stem_b.with_suffix(".png").relative_to(ROOT.parent)),
                "ref_png": str(stem_r.with_suffix(".png").relative_to(ROOT.parent)),
                "nuis_png": str(stem_n.with_suffix(".png").relative_to(ROOT.parent)),
                "facts_base": facts_a, "facts_img": facts_b,
                "facts_twin": facts_a2,
            })

    for s in range(N_SEEDS):
        render(stack_two(refs[("nesting_to_recursion", s)],
                         refs[("recursion_to_contraction", s)]),
               OUT / f"comp_s{s}", 1100)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} matched pairs (+twins, +{N_SEEDS} composites) -> {OUT}")
    for e in EDGES:
        ex = [m for m in manifest if m["edge"] == e][0]
        print(f"  {e:28s} {ex['src_motif']:>22s} -> {ex['dst_motif']:<24s} "
              f"{ex['facts_base']} => {ex['facts_img']}")


if __name__ == "__main__":
    main()
