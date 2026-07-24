#!/usr/bin/env python3
"""Experiment ② design table — Do VLMs Compare Spontaneously?

Within-item design over the 25 motifs.  For every motif we fix:

  TARGET  the single scene used in C1, and as the LEFT panel of every pair
  FAR     an existing scene of the SAME motif with maximally different
          surface (medium / domain / palette); 7 motifs had only one scene,
          for those a new far scene is generated (see FAR_GEN)
  FOIL    (foil_motif, scene) — a surface-similar scene of a DIFFERENT motif
          drawn from the target's confusable family (eval/build_items.py
          FAMILIES); chosen to be the most surface-similar family scene
  NEAR    generated: a nuisance-resample of the TARGET prompt (same subject
          type, medium, palette, composition; different instance)

Conditions
  C1 single      TARGET alone
  C2 near pair   TARGET | NEAR
  C3 far pair    TARGET | FAR
  C4 foil pair   TARGET | FOIL
  C5 instructed  same stimulus as C3, prompt adds an explicit compare step

Scoring is always the TARGET's motif_id (统一指令、单图计分: the prompt asks
about the LEFT panel in all pair conditions with identical wording C2-C4).
"""

# motif -> (target_scene, far_scene, (foil_motif, foil_scene))
# far_scene == None  =>  generated, stimuli/far_src_<motif>.png (FAR_GEN below)
DESIGN = {
    "recursion":             ("river_delta",       "lightning",      ("convergence", "watershed")),
    "nesting":               ("pond_ripples",      "moon_gates",     ("spiral", "typhoon")),
    "cycle":                 ("canal_town",        "ouroboros",      ("interference", "two_ripples")),
    "self_reference":        ("droste_study",      "mountain_temple", ("strange_loop", "drawing_hands")),
    "symmetry":              ("butterfly",         "temple_facade",  ("mobius", "silk_band")),
    "alternation":           ("zebra",             "lantern_row",    ("interference", "moire_curtains")),
    "spiral":                ("nautilus",          "typhoon",        ("mobius", "silk_band")),
    "convergence":           ("watershed",         "homing_birds",   ("recursion", "river_delta")),
    "bottleneck":            ("gorge_bridge",      "hourglass",      ("convergence", "watershed")),
    "symmetry_breaking":     ("orchard_anomaly",   "red_plum",       ("tiling", "salt_pans")),
    "strange_loop":          ("impossible_stairs", "drawing_hands",  ("self_reference", "droste_study")),
    "figure_ground":         ("day_night_birds",   "papercut",       ("symmetry", "temple_facade")),
    "isomorphism":           ("constellations",    "tree_and_roots", ("projective_duality", "spiderweb_dewfall")),
    "mobius":                ("silk_band",         "ant_band",       ("spiral", "nautilus")),
    "braiding":              ("hair_braid",        "interchange",    ("interference", "moire_curtains")),
    "interference":          ("two_ripples",       "moire_curtains", ("cycle", "canal_town")),
    "tiling":                ("honeycomb",         "salt_pans",      ("aperiodic_tiling", "penrose_floor")),
    "hub":                   ("firework",          "starfish",       ("convergence", "homing_birds")),
    "diagonalization":       ("jacquard_loom",     None,             ("symmetry_breaking", "orchard_anomaly")),
    "infinite_descent":      ("nesting_dolls",     None,             ("contraction_fixed_point", "you_are_here")),
    "coprime_orbit":         ("string_art_star",   None,             ("cycle", "ouroboros")),
    "pigeonhole":            ("swallow_nests",     None,             ("convergence", "homing_birds")),
    "projective_duality":    ("spiderweb_dewfall", None,             ("isomorphism", "constellations")),
    "contraction_fixed_point": ("you_are_here",    None,             ("spiral", "nautilus")),
    "aperiodic_tiling":      ("penrose_floor",     None,             ("tiling", "salt_pans")),
}

NO_DIAGRAM = ("It must look like a real photograph or a piece of art, NOT a "
              "diagram: no visible diagram lines, no text, no letters.")
REF = "Use this diagram ONLY as a compositional skeleton. "

# far scenes to generate for single-scene motifs: motif -> (name, skeleton, size, prompt)
# medium/domain chosen to be maximally different from the target scene
FAR_GEN = {
    "diagonalization": ("window_grid", "grid", "1024x1024",
        REF + "A night long-exposure photograph of a modern apartment block "
        "facade, straight-on: a grid of identical windows, each either warmly "
        "LIT or DARK, matching EXACTLY the drawn 6x6 dot pattern (dark dot = "
        "lit window). Along the main diagonal, one window per row glows "
        "reddish, picking out the diagonal. At street level BELOW the grid, "
        "separated by a concrete band, runs one extra row of six shop windows "
        "trimmed in red neon: it is the diagonal sequence with every state "
        "flipped (dark where the diagonal window was lit, lit where it was "
        "dark), so this row matches no floor above. " + NO_DIAGRAM),
    "infinite_descent": ("stone_cairn", "staircase", "1024x1024",
        REF + "A moody dusk long-exposure photograph of a stacked stone cairn "
        "in a mountain stream: flat stones piled following the drawn steps, "
        "each stone strictly and clearly smaller than the one beneath, down "
        "to a pebble at the top. Beside the pebble a tiny gap where the next, "
        "yet smaller stone should sit — but no smaller stone exists; a faint "
        "red twine cross marks that impossible next placement. Silky water, "
        "blue hour light. " + NO_DIAGRAM),
    "coprime_orbit": ("lantern_dance", "star", "1024x1024",
        REF + "A night long-exposure photograph of a plaza folk dance: eight "
        "dancers stand in a circle at exactly the drawn outer points, and a "
        "single lantern has been passed leaping over two dancers to the third "
        "each time, its light trail tracing the drawn eight-point star in one "
        "unbroken glowing stroke that visits every dancer before closing. "
        "Dark plaza, warm lantern trail. " + NO_DIAGRAM),
    "pigeonhole": ("mail_wall", "boxes", "1536x1024",
        REF + "A photorealistic morning photograph inside an old post office: "
        "a wooden wall of exactly four open pigeonhole compartments at the "
        "drawn boxes, and five letters that have just been sorted along the "
        "drawn paths. Three compartments hold one letter each; the "
        "highlighted compartment is forced to hold two letters together — "
        "more letters than compartments, so one must double. Warm interior "
        "light, worn wood. " + NO_DIAGRAM),
    "projective_duality": ("city_day_night", "quadrangle", "1536x1024",
        REF + "A photorealistic aerial diptych of the SAME city crossroads "
        "district. LEFT, by day: the roads read as bold dark lines (the drawn "
        "fat lines) and the intersections are barely-visible small squares "
        "(the drawn small points). RIGHT, the same district by night: now "
        "each intersection blooms into a bright ball of streetlight (points "
        "become the subject) while the roads recede to thin car-light threads "
        "— points and lines have swapped roles, yet every 'which road meets "
        "which crossing' relation is preserved. " + NO_DIAGRAM),
    "contraction_fixed_point": ("marble_bowl", "cobweb", "1024x1024",
        REF + "A dark studio long-exposure photograph of a glass marble "
        "rolling inside a large polished ceramic bowl: its light trace swings "
        "back and forth, each swing a fixed fraction shorter than the last, "
        "following the drawn inward path, until the trace tightens into one "
        "motionless bright point at the bowl's lowest point where the marble "
        "finally rests. Black background, single rim light. " + NO_DIAGRAM),
    "aperiodic_tiling": ("girih_wall", "penrose", "1024x1024",
        REF + "A photograph of a medieval Persian shrine wall covered in "
        "glazed girih ceramic tilework: two rhombus tile shapes (wide and "
        "narrow) in lapis blue and gold arranged exactly as the drawn 'sun' "
        "patch, five-fold symmetric about the center, tiles meeting edge to "
        "edge with no gaps, the ornament clearly never repeating as it "
        "spreads. Slanting afternoon light on ceramic relief. " + NO_DIAGRAM),
}

# appended to the target's original prompt to get a NEAR (nuisance-resample) render
NEAR_VARIATION = (
    " IMPORTANT: render a DIFFERENT instance of the same kind of scene — keep "
    "the same subject type, medium, palette and composition style, but vary "
    "the small incidental details, viewpoint and lighting slightly so it is "
    "clearly not the identical picture.")

CONDITIONS = ["single", "near", "far", "foil", "instructed"]
