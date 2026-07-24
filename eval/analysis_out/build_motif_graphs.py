#!/usr/bin/env python3
"""Independent formal distance, step 1: typed relational graphs per motif.

Each of the 25 motifs is reduced to a small TYPED directed graph derived
mechanically from the RELEASED artifacts only:
  - skeleton_spec  (the SVG drawing-program prose in data/motif_library.json)
  - equation_katex (the motif's defining formula)
Neither the confusable families nor any model output is consulted.

Coding rubric (documented in RUBRIC below, provenance per motif in
PROVENANCE): every node and edge must be traceable to a quoted phrase of the
skeleton_spec or a symbol of the equation.

Output: eval/analysis_out/motif_graphs.json
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

ROOT = Path("/Users/stone/GEB-benchmark")
OUT = ROOT / "eval" / "analysis_out"

RUBRIC = {
    "R1_nodes": ("Every class of repeated drawn primitive (circle/dot/square/"
                 "tile/strand endpoint/box) becomes a node. Repetition counts "
                 "are nuisance: chains and rings are reduced to <=6 elements "
                 "preserving any 2-periodic pattern; cliques to K3; stars to "
                 "center+5; grids to the minimal deviant neighborhood. Every "
                 "reduction is listed in the motif's provenance."),
    "R2_node_types": ("u = generic unit; s0/s1 = one of exactly two visually "
                      "distinguished states/species (fill parity, tile "
                      "species, point-vs-line role, strand start/end bank); "
                      "d = the unique distinguished element (larger radius, "
                      "deviant color, unpaired center, cut vertex, sink, "
                      "fixed point, floor, overloaded box, beat marker). "
                      "Each type must be traceable to an explicit visual "
                      "distinction in skeleton_spec."),
    "R3_edge_types": ("link = a drawn connection between units (segment, "
                      "arc, spoke, clique edge); contain = an arrow "
                      "explicitly stated to point to a deeper/inner element; "
                      "self = a drawn or equation-level self-loop (unique "
                      "self-edge, f(x*)=x*, psi<->phi(code(psi))); back = a "
                      "visually distinguished return edge that closes a "
                      "closed walk (drawn separately from the uniform "
                      "edges, e.g. a Bezier return curve); map = a "
                      "correspondence edge between two distinctly drawn "
                      "groups (dashed pairing line, rung, assignment arrow "
                      "into a box, diagonal-to-escape-row arrow); twist = an "
                      "edge explicitly drawn as a crossing / exchange with "
                      "over-under gaps (Mobius crossover, braid swap)."),
    "R4_direction": ("An edge is directed iff the spec draws an arrowhead "
                     "(marker) or assigns an explicit level/order index "
                     "(layer depth d, ladder height, strictly decreasing "
                     "labels, iteration order); otherwise both directions "
                     "are inserted."),
    "R5_costs_nodes": ("GED node substitution: 0 same type; 0.25 s0<->s1 "
                       "(the two state labels are exchangeable); 0.5 "
                       "u<->s*; 1.0 d<->anything else. Insert/delete 1.0."),
    "R6_costs_edges": ("GED edge substitution: 0 same type; 0.5 "
                       "link<->contain (containment is a specialized link); "
                       "1.0 any other type pair. Insert/delete 1.0."),
    "R7_normalization": ("GED(g1,g2) normalized by (|V1|+|V2|+|E1|+|E2|), "
                         "the cost of deleting g1 and inserting g2."),
}


def _bi(g, a, b, t):
    g.add_edge(a, b, t=t)
    g.add_edge(b, a, t=t)


def build_graphs():
    G, P = {}, {}

    # ---------------------------------------------------------------- 1
    g = nx.DiGraph()
    g.add_node("a", t="u")
    g.add_edge("a", "a", t="self")
    G["self_reference"] = g
    P["self_reference"] = [
        "node a(u): '画布中心画一枚实心圆节点'",
        "edge self a->a: '构成唯一自环；全图仅此一个节点、一条边'; eq: psi <-> phi(code(psi))",
    ]

    # ---------------------------------------------------------------- 2
    g = nx.DiGraph()
    g.add_node("r", t="u"); g.add_node("i1", t="u"); g.add_node("i2", t="u")
    for k in range(1, 5):
        g.add_node(f"l{k}", t="d")
    for a, b in [("r", "i1"), ("r", "i2"), ("i1", "l1"), ("i1", "l2"),
                 ("i2", "l3"), ("i2", "l4")]:
        g.add_edge(a, b, t="link")
    G["recursion"] = g
    P["recursion"] = [
        "7 nodes: '深度 3 的满二叉树（7 个圆、6 条线段）'",
        "leaves typed d: '最底层 4 个叶圆涂实表示基例，其余空心'",
        "6 link edges directed root->leaf by layer index (R4): '第 d 层节点 y=40+90d'; eq: Node(T,T)",
    ]

    # ---------------------------------------------------------------- 3
    g = nx.DiGraph()
    for k in range(4):
        g.add_node(f"c{k}", t="u")
    g.add_node("c4", t="d")
    for k in range(4):
        g.add_edge(f"c{k}", f"c{k+1}", t="contain")
    G["nesting"] = g
    P["nesting"] = [
        "5 nodes: '画 5 个嵌套圆'",
        "innermost d: '最内圆改用约 260° 的圆弧留一缺口，表示最深一层未收口'",
        "4 contain edges, directed inward: '4 条带箭头 line…箭头一律指向更深层、无任何向外箭头'; eq: strict superset chain",
    ]

    # ---------------------------------------------------------------- 4
    g = nx.DiGraph()
    for k in range(6):
        g.add_node(k, t="u")
    for k in range(6):
        g.add_edge(k, (k + 1) % 6, t="link")
    G["cycle"] = g
    P["cycle"] = [
        "6 nodes: '取 6 个等距点…每点画半径 6 的实心圆'",
        "6 uniform directed link edges closing the ring: '从点 i 到点 (i+1) mod 6 画一条略向外凸的弧…marker-end 三角箭头'; no distinguished back edge ('除这 6 条弧外无任何其他边'); eq: v1->...->vn->v1",
    ]

    # ---------------------------------------------------------------- 5
    g = nx.DiGraph()
    for k in range(1, 4):
        g.add_node(f"p{k}", t="u"); g.add_node(f"q{k}", t="u")
    g.add_node("m", t="d")
    for k in range(1, 4):
        _bi(g, f"p{k}", f"q{k}", "map")
    G["symmetry"] = g
    P["symmetry"] = [
        "7 nodes (3 mirror pairs + center): '画 7 个半径 8 的圆'",
        "center d: '中心点 i=4 填第四色且无弧'",
        "3 bi map edges: '镜像对 (1,7)(2,6)(3,5)…每对以上方半圆弧相连' (pairing arcs, no arrowheads -> bidirectional); eq: (w^R)_i = w_{n+1-i}",
    ]

    # ---------------------------------------------------------------- 6
    g = nx.DiGraph()
    for k in range(6):
        g.add_node(k, t=f"s{k % 2}")
    for k in range(5):
        _bi(g, k, k + 1, "link")
    G["alternation"] = g
    P["alternation"] = [
        "chain of 10 reduced to 6 preserving 2-periodicity (R1): '沿水平线等距放 10 个圆'",
        "alternating s0/s1: 'i 为偶数画黑色实心圆，i 为奇数画白色空心圆…黑白严格相间'; eq: c(i)=i mod 2",
        "5 bi link edges: '相邻圆心间画黑色线段相连' (no arrowheads)",
    ]

    # ---------------------------------------------------------------- 7
    g = nx.DiGraph()
    for k in range(6):
        g.add_node(k, t="u")
    for k in range(5):
        g.add_edge(k, k + 1, t="link")
    G["spiral"] = g
    P["spiral"] = [
        "chain of 13 reduced to 6 (R1): '画 k=0..12 共 13 个实心圆点'",
        "5 directed link edges, ordered by index k (R4): '相邻点依次以直线段相连成向外展开的螺线折线'",
        "open chain, no closing edge: '首末两点方位角相同而半径比恰为 1.25^12' (scale growth is metric, not topological; recorded as reduction); eq: r = a e^{b theta}",
    ]

    # ---------------------------------------------------------------- 8
    g = nx.DiGraph()
    for k in range(1, 5):
        g.add_node(f"a{k}", t="u")
    g.add_node("m1", t="u"); g.add_node("m2", t="u"); g.add_node("z", t="d")
    for a, b in [("a1", "m1"), ("a2", "m1"), ("a3", "m2"), ("a4", "m2"),
                 ("m1", "z"), ("m2", "z")]:
        g.add_edge(a, b, t="link")
    G["convergence"] = g
    P["convergence"] = [
        "4+2+1 nodes: 'x=40 处竖排 4 个源点…2 个中间点…1 个汇点'",
        "sink d: '汇点（r=13）' vs r=7 sources; '汇点无出边'",
        "6 directed link edges toward sink: '6 条带三角箭头的直线边…4→2→1 的漏斗形内向树'; eq: exists! P: v ~> r",
    ]

    # ---------------------------------------------------------------- 9
    g = nx.DiGraph()
    for s in ("L", "R"):
        for k in range(3):
            g.add_node((s, k), t="u")
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            _bi(g, (s, a), (s, b), "link")
    g.add_node("cut", t="d")
    for s in ("L", "R"):
        for k in (0, 1):
            _bi(g, (s, k), "cut", "link")
    G["bottleneck"] = g
    P["bottleneck"] = [
        "two K5 cliques reduced to K3 (R1): '左簇 5 个圆点…两两连线（K5 共 10 条），右簇同构'",
        "cut vertex d: '在 (200,100) 画 r=12 的加大圆点'",
        "2 spokes per side, no other cross edges: '左、右簇各取…2 个点各连一条线段至该大点，两簇之间除经此点外无任何连线'; eq: min separator = max disjoint paths",
    ]

    # ---------------------------------------------------------------- 10
    g = nx.DiGraph()
    g.add_node("c", t="d")
    for k in range(1, 5):
        g.add_node(f"n{k}", t="s1")
    g.add_node("f1", t="s0"); g.add_node("f2", t="s0")
    for k in range(1, 5):
        _bi(g, f"n{k}", "c", "link")
    _bi(g, "f1", "n1", "link"); _bi(g, "f2", "n3", "link")
    G["symmetry_breaking"] = g
    P["symmetry_breaking"] = [
        "7x7 lattice reduced to deviant neighborhood (R1): '7×7 圆点阵、间距 40px' (lattice adjacency implicit in the grid)",
        "center d: '唯中心点…改为半径 8 的红色圆且不画箭头'",
        "tangential ring s1 (8 neighbors reduced to 4): '其 8 个邻点的箭头改为切向…形成绕中心的漩涡零点'",
        "uniform field s0: '每点…自圆心向正右方引长 14 的箭头线段（同色同径同向）'; eq: sum of indices = chi(M)",
    ]

    # ---------------------------------------------------------------- 11
    g = nx.DiGraph()
    for k in range(4):
        g.add_node(f"l{k}", t="u")
    for k in range(3):
        g.add_edge(f"l{k}", f"l{k+1}", t="link")
    g.add_edge("l3", "l0", t="back")
    G["strange_loop"] = g
    P["strange_loop"] = [
        "4 level nodes: '画 4 个圆…排成竖直阶梯；每圆右侧画一段水平虚线标示层高'",
        "3 directed link edges up the ladder: '相邻圆之间画向上直线箭头'",
        "1 distinguished back edge (typed back, not link): '再画一条带箭头的三次贝塞尔回边…从最高圆右侧弯回最低圆'; eq: e* = (v_k, v_0)",
    ]

    # ---------------------------------------------------------------- 12
    g = nx.DiGraph()
    pat = [1, 0, 1, 0]
    for i, b in enumerate(pat):
        g.add_node(("t", i), t=f"s{b}")
        g.add_node(("b", i), t=f"s{1-b}")
        _bi(g, ("t", i), ("b", i), "map")
    G["figure_ground"] = g
    P["figure_ground"] = [
        "two rows of 8 reduced to 4 columns (R1): '画两行各 8 个半径 12 的圆'",
        "complementary states: '下行…填色与上行逐位取反（黑变白、白变黑）'",
        "4 bi map edges: '每列上下两圆圆心之间画一条 stroke-dasharray 的竖直虚线对齐' (dashed pairing); eq: A in Sigma_1^0, complement not",
    ]

    # ---------------------------------------------------------------- 13
    g = nx.DiGraph()
    for s in ("L", "R"):
        for k in range(3):
            g.add_node((s, k), t="u")
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            _bi(g, (s, a), (s, b), "link")
    for k in range(3):
        _bi(g, ("L", k), ("R", k), "map")
    G["isomorphism"] = g
    P["isomorphism"] = [
        "same 5-vertex graph drawn twice, reduced to two K3 copies (R1): '左侧按正五边形布局画 G…右侧把同一边集按打乱的固定坐标再画一遍'",
        "both sides SAME type u (same roles, unlike projective_duality): '外观迥异、边集相同'",
        "3 bi map edges: '用 5 条灰色虚线把两侧同号顶点 i–i 一一相连'; eq: phi(a.b)=phi(a) o phi(b), bijective",
    ]

    # ---------------------------------------------------------------- 14
    g = nx.DiGraph()
    for k in range(3):
        g.add_node(("i", k), t="s0")
        g.add_node(("o", k), t="s1")
    for k in range(2):
        g.add_edge(("i", k), ("i", k + 1), t="link")
        g.add_edge(("o", k), ("o", k + 1), t="link")
    g.add_edge(("i", 2), ("o", 0), t="twist")
    g.add_edge(("o", 2), ("i", 0), t="twist")
    for k in range(3):
        _bi(g, ("i", k), ("o", k), "map")
    G["mobius"] = g
    P["mobius"] = [
        "two rings of 12 reduced to 3+3 (R1): '取内外两个同心圆各 12 个等角圆点'",
        "inner s0 / outer s1 banks; directed rim links: '对 k=0..10 连线段 内k→内k+1、外k→外k+1'",
        "2 twist edges: '最后一档改为交叉互换：内11→外0 与 外11→内0，两线交点处…留 6px 空隙表示下穿'",
        "3 bi map rungs: '再画 12 条内k–外k 短横档示带宽'; eq: (0,t) ~ (1, 1-t)",
    ]

    # ---------------------------------------------------------------- 15
    g = nx.DiGraph()
    for k in range(1, 4):
        g.add_node(("a", k), t="s0")
        g.add_node(("b", k), t="s1")
    perm = {1: 2, 2: 3, 3: 1}
    for k in range(1, 4):
        g.add_edge(("a", k), ("b", perm[k]), t="twist")
    G["braiding"] = g
    P["braiding"] = [
        "3 strands = 3 left endpoints (s0) + 3 right endpoints (s1): '画三条自左向右的折线（初始 y=50/100/150）'",
        "3 directed twist edges by the composite permutation (a 3-cycle): '被交换的两线…以斜直线段互换 y 坐标，居下（被越过）的线段…留 6px 缺口示上下方向，右端按复合置换（此例为三轮换…）标注各股编号'; eq: sigma_i sigma_{i+1} sigma_i = ...",
    ]

    # ---------------------------------------------------------------- 16
    g = nx.DiGraph()
    g.add_node("a1", t="s0"); g.add_node("a2", t="s0")
    g.add_node("b1", t="s1"); g.add_node("b2", t="s1")
    g.add_node("z1", t="d"); g.add_node("z2", t="d")
    _bi(g, "a1", "z1", "map"); _bi(g, "b1", "z1", "map")
    _bi(g, "a2", "z2", "map"); _bi(g, "b2", "z2", "map")
    G["interference"] = g
    P["interference"] = [
        "two line species s0/s1 (families reduced to 2 lines each, R1): 'A 组 x=20k（周期 20px），B 组 x=24k（周期 24px）两组半透明竖直线段'",
        "beat markers d (reduced to 2): '叠加后每 lcm(20,24)=120px 出现一次两线重合的深色拍带，并在 x=0,120,… 处各画一个小圆点标记同相位置'",
        "map edges: each beat marker is the coincidence of one A line and one B line ('两线重合'); eq: sin a + sin b = 2 sin((a+b)/2) cos((a-b)/2)",
    ]

    # ---------------------------------------------------------------- 17
    g = nx.DiGraph()
    for k in range(10):
        g.add_node(k, t="u")
    ring1 = [0, 1, 2, 3, 4, 5]
    for i in range(6):
        _bi(g, ring1[i], ring1[(i + 1) % 6], "link")
    ring2 = [5, 6, 7, 8, 9, 4]
    for i in range(5):
        _bi(g, ring2[i], ring2[i + 1], "link")
    G["tiling"] = g
    P["tiling"] = [
        "honeycomb of 8x7 hexagons reduced to two edge-sharing cells (R1): '画平顶正六边形蜂窝…i=0..7、j=0..6'",
        "each cell a 6-ring of bi link edges, shared edge deduplicated: '每个中心把六个顶点…依次连成 6 条线段，并在每个顶点画…圆点（重复顶点去重）'",
        "single uniform species (contrast aperiodic_tiling); eq: T + lambda = T for the full lattice",
    ]

    # ---------------------------------------------------------------- 18
    g = nx.DiGraph()
    g.add_node("c", t="d")
    for k in range(5):
        g.add_node(k, t="u")
        _bi(g, "c", k, "link")
    G["hub"] = g
    P["hub"] = [
        "center d: '画半径 14 的实心大圆作中心'",
        "8 spokes reduced to 5 (R1): '8 个半径 6 的小圆置于半径 120 的圆周上'",
        "bi link spokes only, no rim: '从中心到每个小圆心各画一条直线段，小圆之间不画任何连线'; eq: G ~ K_{1,n}",
    ]

    # ---------------------------------------------------------------- 19
    g = nx.DiGraph()
    bits = [1, 0, 1]
    for i, b in enumerate(bits):
        g.add_node(("dg", i), t=f"s{b}")
        g.add_node(("e", i), t=f"s{1-b}")
        g.add_edge(("dg", i), ("e", i), t="map")
    G["diagonalization"] = g
    P["diagonalization"] = [
        "6 diagonal cells reduced to 3 preserving bit alternation (R1): '对角格 (i,i) 各套红圈,其位序为 1 0 1 0 1 0'",
        "escape row with flipped states: '第 7 行(表外新行):第 i 列点取对角格 (i,i) 的相反填充'",
        "3 directed map arrows: '从每个对角格…画红色竖直箭头引至正下方新行格…示新行第 i 格 = 对角第 i 格翻面'; eq: d(n) = 1 - f(n)(n)",
    ]

    # ---------------------------------------------------------------- 20
    g = nx.DiGraph()
    for k in range(1, 5):
        g.add_node(f"q{k}", t="u")
    g.add_node("floor", t="d")
    for k in range(1, 4):
        g.add_edge(f"q{k}", f"q{k+1}", t="link")
    g.add_edge("q4", "floor", t="link")
    G["infinite_descent"] = g
    P["infinite_descent"] = [
        "6 squares reduced to 4 (R1): '六个实心正方形沿对角线向右下排列…编号严格递减'",
        "directed links toward smaller (R4): '相邻方块中心间画直线箭头（方向指向更小者）'",
        "floor d: '画一条 4px 加粗横线（地板）…其与地板交点处叠画一个红色大叉。地板下方区域填浅灰色并标 ∅' (blocked terminal); eq: => forall n not P(n)",
    ]

    # ---------------------------------------------------------------- 21
    g = nx.DiGraph()
    for k in range(6):
        g.add_node(k, t="u")
    for k in range(6):
        g.add_edge(k, (k + 1) % 6, t="link")
    for k in range(4):
        g.add_node(("c", k), t="s1")
    _bi(g, ("c", 0), ("c", 1), "link")
    _bi(g, ("c", 2), ("c", 3), "link")
    G["coprime_orbit"] = g
    P["coprime_orbit"] = [
        "main {8/3} star is, as a graph, one closed orbit through all points; 8 reduced to 6 (R1): '按 0→3→6→1→4→7→2→5→0 顺次连直线弦…成一笔闭合的 {8/3} 八角星'",
        "contrast inset: disjoint sub-orbits in a second style (s1, dashed), two components reduced to two pairs: '小对照圆:同样 8 点按步长 2 连弦,得两个互不相交的正方形,用虚线,并标 gcd=2'; eq: <k> = Z_n iff gcd(k,n)=1",
    ]

    # ---------------------------------------------------------------- 22
    g = nx.DiGraph()
    for k in range(3):
        g.add_node(("i", k), t="s0")
    g.add_node(("b", 0), t="s1")
    g.add_node(("b", 1), t="d")
    g.add_edge(("i", 0), ("b", 0), t="map")
    g.add_edge(("i", 1), ("b", 1), t="map")
    g.add_edge(("i", 2), ("b", 1), t="map")
    G["pigeonhole"] = g
    P["pigeonhole"] = [
        "5 items reduced to 3, 4 boxes to 2, preserving |A|>|B| (R1): '上排 5 个实心圆点…下排 4 个 70×70 方框'; eq: |A| > |B|",
        "assignment arrows typed map: '画 5 条箭头把每个点分入一框'",
        "overloaded box d: '其中恰有一框接到两条箭头,该框边线加粗描红且框内画两个点'",
    ]

    # ---------------------------------------------------------------- 23
    g = nx.DiGraph()
    for k in range(3):
        g.add_node(("p", k), t="s0")
        g.add_node(("l", k), t="s1")
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        _bi(g, ("p", a), ("p", b), "link")
        _bi(g, ("l", a), ("l", b), "link")
    for k in range(3):
        _bi(g, ("p", k), ("l", k), "map")
    G["projective_duality"] = g
    P["projective_duality"] = [
        "complete quadrilateral pair reduced to K3 copies (R1): '左幅完全四点形:4 个大实心圆点…两两连出 6 条细线；右幅完全四线形:4 条粗线段…两两交点恰 6 个'",
        "sides carry SWAPPED roles s0 vs s1 (unlike isomorphism where both are u): '视觉互换轻重:左幅点大线细,右幅线粗点小；左 4 点 6 线 ↔ 右 4 线 6 点,计数逐项对调'",
        "bi map edges between the two panels: '两幅之间画双向箭头,上注 σ、下注 点↔线'; eq: sigma^2 = id",
    ]

    # ---------------------------------------------------------------- 24
    g = nx.DiGraph()
    for k in range(1, 5):
        g.add_node(f"x{k}", t="u")
    g.add_node("fix", t="d")
    for k in range(1, 4):
        g.add_edge(f"x{k}", f"x{k+1}", t="link")
    g.add_edge("x4", "fix", t="link")
    g.add_edge("fix", "fix", t="self")
    G["contraction_fixed_point"] = g
    P["contraction_fixed_point"] = [
        "cobweb staircase reduced to 4 iterates (R1): '阶梯折线顶点依次…竖直段长 70,35,17.5,8.75 逐次严格减半' (iteration order directs the chain, R4)",
        "fixed point d: '终点 (160,160) 画实心红点加红圈'",
        "self-loop at the fixed point, from the drawn annotation and equation: '标注 f(x^*)=x^*'; eq: exists! x*: f(x*)=x*",
    ]

    # ---------------------------------------------------------------- 25
    g = nx.DiGraph()
    g.add_node("C", t="d")
    for k in range(1, 4):
        g.add_node(("f", k), t="s0")
        _bi(g, "C", ("f", k), "link")
    for k in range(1, 3):
        g.add_node(("t", k), t="s1")
        _bi(g, ("f", k), ("t", k), "link")
        _bi(g, ("f", k + 1), ("t", k), "link")
    G["aperiodic_tiling"] = g
    P["aperiodic_tiling"] = [
        "sun patch reduced: center + 5 fat + 5 thin -> center + 3 fat + 2 thin (R1): '5 个胖菱尖端共聚 C,恰满 5×72°=360°'",
        "two tile species s0 (fat) / s1 (thin): '胖菱填 #E8C468(浅金),瘦菱填 #3B4A8C(靛蓝)'",
        "thin tiles wedged between adjacent fats: '相邻两胖菱在共享外顶点…余 144° 缺口,各嵌 1 个瘦菱形'",
        "center d, non-uniform neighborhood (contrast tiling): '整体呈五瓣放射状,绝无方格/六角网格感'; eq: N_fat/N_thin -> phi",
    ]

    return G, P


def main():
    lib = json.loads((ROOT / "data" / "motif_library.json").read_text())
    order = [m["id"] for m in lib]
    G, P = build_graphs()
    assert set(G) == set(order), set(order) ^ set(G)

    out = {"rubric": RUBRIC, "motifs": {}}
    for mid in order:
        g = G[mid]
        out["motifs"][mid] = {
            "nodes": [[str(n), d["t"]] for n, d in g.nodes(data=True)],
            "edges": [[str(a), str(b), d["t"]]
                      for a, b, d in g.edges(data=True)],
            "n_nodes": g.number_of_nodes(),
            "n_edges": g.number_of_edges(),
            "provenance": P[mid],
        }
    (OUT / "motif_graphs.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    for mid in order:
        g = G[mid]
        types = sorted({d["t"] for _, _, d in g.edges(data=True)})
        print(f"{mid:26s} n={g.number_of_nodes():2d} "
              f"e={g.number_of_edges():2d} edge_types={types}")
    print("wrote", OUT / "motif_graphs.json")


if __name__ == "__main__":
    main()
