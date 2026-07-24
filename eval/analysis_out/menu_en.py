#!/usr/bin/env python3
"""English menu for the story2motif-EN (bilingual transport) replication.

Faithful English translations of data/motif_library.json `invariant_zh`
lines, done clause-by-clause so the EN option menu carries the SAME
information as the shipped Chinese menu (including the long exclusion
clauses of the round-2 motifs).  Motif ids are unchanged.
"""

INVARIANT_EN = {
    "self_reference": (
        "The object contains a sentence describing itself, and the description "
        "checks out verbatim as true — it points back to itself in one step, "
        "with no climb through levels and no reference to a second system."),
    "recursion": (
        "The whole is generated from smaller copies of itself by the same rule; "
        "however many levels you enter, that many you exit, terminating at a "
        "base case and then returning level by level along the original path."),
    "nesting": (
        "Each time a new level is entered, the previous level is never returned "
        "to; depth only ever increases."),
    "cycle": (
        "Following same-level causal arrows through n>=3 mutually distinct "
        "events returns you to the starting event; one lap restores everything "
        "verbatim (no escalation, no growth, no flip), all points equal in "
        "status, no hierarchy."),
    "symmetry": (
        "The sequence of events read backwards is item-for-item identical to "
        "reading it forwards; folded at the temporal midpoint, it coincides "
        "point by point."),
    "alternation": (
        "Exactly two states strictly take turns: adjacent items always differ, "
        "items one apart are always the same, the period is constantly two, "
        "without a single exception."),
    "spiral": (
        "Each time it comes back around to the same bearing it grows one notch "
        "by the same fixed ratio, never returning to the original scale."),
    "convergence": (
        "Multiple threads, mutually unaware, each advance on their own and "
        "finally all merge into the same endpoint, and there it ends."),
    "bottleneck": (
        "Each of the two shores is internally multiply connected, but every "
        "path between the shores must pass through one and the same midway "
        "point, continuing separately beyond it; deleting that point splits "
        "the graph into exactly two still-connected halves, rather than "
        "shattering it into isolated pieces."),
    "symmetry_breaking": (
        "One rule covers the whole field, with exactly one precisely locatable "
        "violation; the violation is unique and everywhere else is without "
        "exception."),
    "strange_loop": (
        "Climbing step by step up the hierarchy of 'who narrates whom', the "
        "final step nevertheless lands back at the starting point; no "
        "consistent level can be assigned."),
    "figure_ground": (
        "Figure and ground complementarily fill the whole canvas and the "
        "ground itself forms a figure: the second half takes the first half's "
        "ground as its figure and the first half's figure as its ground, while "
        "the narrated events and their order remain unchanged (not reversed "
        "order, and not a translation between two vocabularies)."),
    "isomorphism": (
        "Between two things made of entirely different material there exists a "
        "one-to-one dictionary preserving all operations and relations, "
        "translatable term by term in the same order, without a single "
        "exception."),
    "mobius": (
        "Traveling once around the band returns you to the starting place with "
        "orientation flipped (left and right exchanged); only after two laps "
        "is everything restored point by point."),
    "braiding": (
        "Three or more strands cross adjacently several times; the final "
        "arrangement of the ends equals exactly the composition of all the "
        "crossings in order, and the order cannot be freely exchanged."),
    "interference": (
        "Two patterns with different periods, each monotonously repeating on "
        "its own, are superposed and a new period larger than either original "
        "emerges."),
    "tiling": (
        "Shift the entire pattern by one cell and it coincides exactly with "
        "the original, seamless and without overlap, without a single "
        "exception."),
    "hub": (
        "Between any two endpoints stands only the center point; remove the "
        "center and the whole graph shatters into isolated points."),
    "diagonalization": (
        "A table enumerated row by row; the new row built by flipping each "
        "entry along the diagonal is guaranteed to differ from row n at "
        "position n, hence differs from every row — forever outside the "
        "table. Unlike self_reference (an object pointing back to itself in "
        "one step): the diagonal element does not talk about itself; the "
        "attack targets the table's entire enumeration, not itself. Crucially "
        "unlike symmetry_breaking (exactly one isolated violation in an "
        "otherwise regular field): here it is not one anomalous cell but n "
        "systematic differences along a moving diagonal — row n differs at "
        "position n, the position shifting right by one each row, exactly one "
        "per row, and this construction guarantees 'not equal to any row'; "
        "symmetry_breaking is a single-point flaw that neither escapes nor "
        "refutes any enumeration."),
    "infinite_descent": (
        "Every case forces a strictly smaller case of the same kind, and the "
        "natural numbers admit no bottomless descending chain, so the first "
        "case does not exist at all — the proof's blade swings backwards: not "
        "advancing to an end, but cancelling the beginning. Exactly opposite "
        "to recursion: recursion exits as many levels as it enters and stands "
        "on the existence of a base case; descent stands on the absence of one "
        "and condemns every witness to nonexistence. Unlike zeno_halving: "
        "Zeno's steps shrink by a fixed ratio, total length finite, the "
        "journey completes; descent's decrease is arbitrary but each step is "
        "an integer, the chain is forbidden by well-ordering, the conclusion "
        "is 'impossible'. Unlike contraction_fixed_point: contraction iterates "
        "forever and converges to a genuinely existing fixed point; the "
        "descending chain converges to nothing, must hit floor 1 in finitely "
        "many steps, so the whole chain together with its starting point is "
        "judged nonexistent. Opposite in direction to ordinal_ascent: ordinal "
        "ascent is unbounded upward with every level real; descent is bounded "
        "below with the whole chain illusory. Unlike invariant_barrier: no "
        "conserved quantity is invoked; the only killing move is that strict "
        "decrease cannot go on forever."),
    "coprime_orbit": (
        "On a ring of n points, jump with a fixed stride k, connecting as you "
        "go: if the stride is coprime to the ring length, a single stroke "
        "visits every point before closing; if not, the ring splits into gcd "
        "parts. Unlike cycle (adjacent arrows closing head-to-tail in one "
        "lap): the large stride weaves a star shape, traversal is conditional "
        "on coprimality, and there is a failure mode — not coprime means "
        "splitting. Unlike interference (two independent periods superposed, "
        "an lcm-sized larger period emerging): here there is only one orbit "
        "and one number-theoretic condition. Unlike hub (relaying through a "
        "center): there is no central point at all; traversal is guaranteed "
        "purely by gcd(k,n)=1."),
    "pigeonhole": (
        "More items than cages forces two into the same cage: the collision is "
        "proved certain, yet no one can point to where it happens. Unlike "
        "convergence (all threads, knowingly or not, ending at the same KNOWN "
        "point): here counting only guarantees that at some UNSPECIFIED place "
        "at least two meet; no witness is constructed. Unlike bottleneck "
        "(every path forced through the same midway point): there is no "
        "forced path and no relay hub, only a single count."),
    "projective_duality": (
        "Within a single system, two classes of roles are swapped wholesale "
        "and every relational proposition remains true; swapping twice "
        "restores the original verbatim (an involution, sigma^2=id). "
        "Exclusions: unlike isomorphism (a relation-preserving dictionary "
        "between two heterogeneous materials) — duality is a role swap inside "
        "one and the same material and is necessarily an involution; unlike "
        "figure_ground (figure and ground complementarily filling one canvas) "
        "— duality produces two pictures each complete in itself, not relying "
        "on negative space; unlike transformed_canon (the same theme "
        "reappearing in a lagged voice under a fixed transformation) — "
        "duality is a wholesale sequential restatement that swaps role "
        "vocabulary while truth values stay unchanged, not a voice-level "
        "pitch/time transformation; unlike chirality (mirror image and "
        "original cannot be superposed, the point being 'it does not fit') — "
        "the two dual statements both hold within the same system and the "
        "swap restores everything exactly; unlike mobius (the flip requires "
        "traveling one full lap along a path) — duality has no path, it is an "
        "immediate wholesale role swap; unlike symmetry (item-by-item "
        "reversed palindrome) — duality does not reverse order, it only "
        "swaps roles."),
    "contraction_fixed_point": (
        "One and the same rule is applied again and again to its own output; "
        "each step shrinks the distance to the endpoint by a fixed ratio q<1, "
        "finally resting at the one point that the rule sends back to itself "
        "unchanged — the endpoint is not announced in advance but selected "
        "from inside the rule by the fixed-point property f(x*)=x*. Unlike "
        "spiral (fixed-ratio growth, never returning to the original scale): "
        "the direction is reversed and there is a real endpoint; unlike "
        "convergence (many mutually unaware threads ending at one place): "
        "here there is from start to finish a single self-iterating chain; "
        "unlike recursion (as many exits as entries, returning level by level "
        "to the caller): the iteration only collapses forward and never "
        "returns; unlike zeno_halving (infinitely shrinking segments laid end "
        "to end, the point being that the infinite sum is exactly a finite "
        "whole): here the point is that the endpoint is fixed by the map — "
        "what is checked is f(x*)=x*, not a sum of distances; unlike "
        "negative_feedback (alternating over- and under-shoot around a target "
        "announced at the outset): the target must not be pre-announced, no "
        "sign alternation is required, one-sided collapse suffices, and the "
        "endpoint certifies itself by its own fixedness."),
    "aperiodic_tiling": (
        "Everywhere obeying the same local matching rules, seamlessly covering "
        "the whole plane, yet no translation whatsoever makes the whole "
        "picture coincide with itself; at the same time any local patch "
        "reappears within a bounded gap everywhere — 'seen before, never "
        "repeated'. Exclusion clauses: unlike tiling (shift one cell and the "
        "whole coincides; here it is provable that no translational period "
        "exists at all); unlike symmetry_breaking (a periodic background with "
        "exactly one locatable violation; here there is no violation anywhere, "
        "yet no period exists at all); unlike spiral (phi here is the "
        "frequency ratio of the two tile kinds, not level-by-level fixed-ratio "
        "growth; the figure has no self-similar scaling center); unlike "
        "transformed_canon (the same theme reappearing verbatim under a fixed "
        "transformation; here no block ever recurs exactly at a fixed "
        "interval); unlike interference (two periods superposed still yield "
        "an lcm period; here the limiting structure's frequency is irrational, "
        "so there is never a period)."),
}
