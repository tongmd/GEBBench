#!/usr/bin/env python3
"""Generate additional academic figures following benchmark conventions."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "eval"
PAPER = ROOT / "paper"

# Load data
LIB = json.loads((ROOT / "data" / "motif_library.json").read_text())
MOTIFS = [m["id"] for m in LIB]
LIB_MAP = {m["id"]: m for m in LIB}
AGG = json.loads((EVAL / "results" / "_aggregate.json").read_text())

# Warm palette
BG = "#F7F4EF"
SURFACE = "#FBF9F5"
TERRACOTTA = "#C4612F"
MUTED = "#5C635D"
INK = "#1F2421"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": BG,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#E7E1D7",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#E7E1D7",
    "grid.alpha": 0.4,
})


def per_motif_heatmap():
    """Figure: Model × Motif accuracy heatmap (all 25 motifs, all tasks pooled)."""
    models = ["api_gpt-5.5", "api_gpt-5.2", "api_claude-opus-4-8",
              "qwen2.5-vl-32b", "qwen2.5-coder-32b",
              "qwen2.5-14b", "qwen2.5-7b", "qwen2-vl-7b"]
    model_labels = ["GPT-5.5", "GPT-5.2", "Claude Opus 4.8",
                    "Qwen2.5-VL-32B", "Qwen2.5-Coder-32B",
                    "Qwen2.5-14B", "Qwen2.5-7B", "Qwen2-VL-7B"]

    # Build motif accuracy matrix: for each model, compute per-motif accuracy
    matrix = np.zeros((len(models), len(MOTIFS)))

    for i, mid in enumerate(models):
        if mid not in AGG["models"]:
            matrix[:, i] = np.nan
            continue
        result_path = EVAL / "results" / f"{mid}.json"
        if not result_path.exists():
            matrix[:, i] = np.nan
            continue

        data = json.loads(result_path.read_text())
        records = data.get("records", [])

        # Group by motif
        motif_correct = defaultdict(list)
        for rec in records:
            answer = rec.get("answer")
            correct = rec.get("correct", False)
            if answer in MOTIFS:
                motif_correct[answer].append(correct)

        for j, motif_id in enumerate(MOTIFS):
            if motif_id in motif_correct and len(motif_correct[motif_id]) > 0:
                matrix[i, j] = np.mean(motif_correct[motif_id])
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # Diverging colormap: red (0) -> white (0.5) -> blue (1)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#D73027", "#FFFFFF", "#4575B4"]
    cmap = LinearSegmentedColormap.from_list("diverging", colors)

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(MOTIFS)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.3 or val > 0.7 else INK
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                       fontsize=6, color=text_color, weight="medium")

    ax.set_xticks(range(len(MOTIFS)))
    ax.set_xticklabels([LIB_MAP[m]["name_en"].split("—")[0].strip() for m in MOTIFS],
                       rotation=90, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels)

    ax.set_xlabel("Motif (25 total)", weight="medium")
    ax.set_ylabel("Model", weight="medium")
    ax.set_title("Per-Motif Accuracy Across Models (pooled over all tasks)",
                weight="medium", pad=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Accuracy", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(PAPER / "figs" / "per_motif_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER / "figs" / "per_motif_heatmap.png", dpi=150, bbox_inches="tight")
    print("✓ per_motif_heatmap.pdf/png")


def task_breakdown_bars():
    """Figure: Per-task accuracy breakdown for all models."""
    models = ["api_gpt-5.5", "api_gpt-5.2", "api_claude-opus-4-8",
              "qwen2.5-vl-32b", "qwen2.5-coder-32b",
              "qwen2.5-14b", "qwen2.5-7b", "qwen2-vl-7b"]
    model_labels = ["GPT-5.5", "GPT-5.2", "Claude Opus 4.8",
                    "Qwen2.5-VL-32B", "Qwen2.5-Coder-32B",
                    "Qwen2.5-14B", "Qwen2.5-7B", "Qwen2-VL-7B"]
    tasks = ["scene2motif", "skeleton2motif", "story2motif", "theorem2motif", "crossvoice"]
    task_labels = ["Scene", "Skeleton", "Story", "Theorem", "Cross-voice"]

    data = np.zeros((len(models), len(tasks)))
    for i, mid in enumerate(models):
        if mid not in AGG["models"]:
            data[i, :] = np.nan
            continue
        for j, task in enumerate(tasks):
            if task in AGG["models"][mid]["tasks"]:
                data[i, j] = AGG["models"][mid]["tasks"][task]["acc"] * 100
            else:
                data[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(models))
    width = 0.15
    colors = ["#8B4513", "#C4612F", "#D4A373", "#6B8E23", "#556B2F"]

    for j, task_label in enumerate(task_labels):
        offset = (j - len(tasks)/2 + 0.5) * width
        bars = ax.bar(x + offset, data[:, j], width, label=task_label,
                     color=colors[j], edgecolor="white", linewidth=0.5)

    # Random chance lines
    ax.axhline(4.0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.6, label="Chance (4-way)")
    ax.axhline(25.0, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.6, label="Chance (25-way)")

    ax.set_xlabel("Model", weight="medium")
    ax.set_ylabel("Accuracy (%)", weight="medium")
    ax.set_title("Per-Task Accuracy Breakdown", weight="medium", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=True, fancybox=False, shadow=False, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PAPER / "figs" / "task_breakdown.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER / "figs" / "task_breakdown.png", dpi=150, bbox_inches="tight")
    print("✓ task_breakdown.pdf/png")


def hardest_motifs_bars():
    """Figure: Top 10 hardest motifs (lowest average accuracy across all models)."""
    # Aggregate per-motif accuracy
    motif_accs = defaultdict(list)

    for mid in AGG["models"]:
        result_path = EVAL / "results" / f"{mid}.json"
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text())
        records = data.get("records", [])

        motif_correct = defaultdict(list)
        for rec in records:
            answer = rec.get("answer")
            correct = rec.get("correct", False)
            if answer in MOTIFS:
                motif_correct[answer].append(correct)

        for motif_id in MOTIFS:
            if motif_id in motif_correct and len(motif_correct[motif_id]) > 0:
                motif_accs[motif_id].append(np.mean(motif_correct[motif_id]))

    # Average across models
    avg_accs = {m: np.mean(vals) for m, vals in motif_accs.items() if len(vals) > 0}
    sorted_motifs = sorted(avg_accs.items(), key=lambda x: x[1])[:10]

    fig, ax = plt.subplots(figsize=(8, 5))

    motif_ids = [m[0] for m in sorted_motifs]
    accs = [m[1] * 100 for m in sorted_motifs]
    labels = [LIB_MAP[m]["name_en"].split("—")[0].strip() for m in motif_ids]

    bars = ax.barh(range(len(motif_ids)), accs, color=TERRACOTTA, edgecolor="white", linewidth=0.5)

    # Annotate
    for i, (motif_id, acc) in enumerate(zip(motif_ids, accs)):
        ax.text(acc + 1, i, f"{acc:.1f}%", va="center", fontsize=8, color=MUTED)

    ax.set_yticks(range(len(motif_ids)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average Accuracy (%) Across All Models", weight="medium")
    ax.set_title("Ten Hardest Motifs", weight="medium", pad=12)
    ax.set_xlim(0, 100)
    ax.axvline(4.0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.5, label="Random (4%)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(PAPER / "figs" / "hardest_motifs.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER / "figs" / "hardest_motifs.png", dpi=150, bbox_inches="tight")
    print("✓ hardest_motifs.pdf/png")


def motif_taxonomy_tree():
    """Figure: Motif taxonomy as a conceptual tree/hierarchy."""
    # Define categories (hand-curated conceptual grouping)
    categories = {
        "Self-Referential Structures": ["self_reference", "strange_loop", "recursion", "infinite_descent"],
        "Fixed Points & Iterations": ["contraction_fixed_point", "convergence", "cycle", "coprime_orbit"],
        "Spatial & Topological": ["mobius", "spiral", "nesting", "figure_ground"],
        "Symmetry & Breaking": ["symmetry", "symmetry_breaking", "alternation", "duality"],
        "Combinatorial Patterns": ["tiling", "aperiodic_tiling", "braiding", "interference"],
        "Graph & Network": ["bottleneck", "hub", "isomorphism", "projective_duality"],
        "Counting & Constraints": ["pigeonhole", "diagonalization"],
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    # Tree layout: categories as level 1, motifs as level 2
    y_start = 0.95
    y_step = 0.13

    for i, (cat, motif_ids) in enumerate(categories.items()):
        y_cat = y_start - i * y_step

        # Category box
        rect = mpatches.FancyBboxPatch((0.05, y_cat - 0.04), 0.25, 0.06,
                                       boxstyle="round,pad=0.005",
                                       edgecolor=TERRACOTTA, facecolor=SURFACE, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(0.175, y_cat - 0.01, cat, ha="center", va="center",
               fontsize=9, weight="bold", color=INK)

        # Motif labels
        x_motif = 0.35
        x_step = 0.13
        for j, motif_id in enumerate(motif_ids):
            if motif_id not in LIB_MAP:
                continue
            x_pos = x_motif + (j % 5) * x_step
            y_pos = y_cat - 0.01 - (j // 5) * 0.03

            name = LIB_MAP[motif_id]["name_en"].split("—")[0].strip()
            ax.text(x_pos, y_pos, name, ha="left", va="center",
                   fontsize=7, color=MUTED)

            # Line from category to motif
            ax.plot([0.30, x_pos - 0.01], [y_cat - 0.01, y_pos],
                   color="#E7E1D7", linewidth=0.5, alpha=0.6)

    ax.text(0.5, 0.98, "GEB-Bench Motif Taxonomy", ha="center", va="top",
           fontsize=12, weight="bold", color=INK)
    ax.text(0.5, 0.02, "25 motifs grouped by conceptual affinity", ha="center", va="bottom",
           fontsize=8, color=MUTED, style="italic")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PAPER / "figs" / "motif_taxonomy.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER / "figs" / "motif_taxonomy.png", dpi=150, bbox_inches="tight")
    print("✓ motif_taxonomy.pdf/png")


if __name__ == "__main__":
    (PAPER / "figs").mkdir(exist_ok=True, parents=True)

    print("Generating additional academic figures...")
    per_motif_heatmap()
    task_breakdown_bars()
    hardest_motifs_bars()
    motif_taxonomy_tree()
    print("\nAll additional figures generated.")
