"""Shared ICLR figure style for all GEB-Bench paper figures.

Import and call `use()` at the top of every figure script, after the
matplotlib backend is set. Aesthetics only: no script may change what data
it plots. The palette fixes one color per model across every figure so a
reader can track a model between plots.
"""

MODEL_COLORS = {
    # frontier proprietary
    "GPT-5.5":          "#B3402A",
    "GPT-5.2":          "#E09044",
    "Claude Opus 4.8":  "#7A4B94",
    "Gemini 3.1 Pro":   "#3E7DA6",
    # open-weights
    "Kimi K2.5":        "#2E8B74",
    "Qwen2.5-VL-72B":   "#4C8C5C",
    "DeepSeek-V3.2":    "#556677",
    # smaller open
    "Qwen2.5-VL-32B":   "#7FA85A",
    "Qwen2-VL-7B":      "#3B4A8C",
    "Qwen2.5-14B":      "#A8874A",
    "Qwen2.5-7B":       "#C4A97E",
    "DeepSeek-Coder-V2-Lite": "#999999",
}

MODEL_MARKERS = {
    "GPT-5.5": "o", "GPT-5.2": "s", "Claude Opus 4.8": "D",
    "Gemini 3.1 Pro": "^", "Kimi K2.5": "v", "Qwen2.5-VL-72B": "P",
    "DeepSeek-V3.2": "X", "Qwen2.5-VL-32B": "<", "Qwen2-VL-7B": ">",
    "Qwen2.5-14B": "h", "Qwen2.5-7B": "d", "DeepSeek-Coder-V2-Lite": "*",
}

ACCENT = "#C4612F"      # the paper's terracotta accent
NEUTRAL = "#4A4A4A"
GRID = "#DDDDDD"

# text widths in inches for the ICLR column (5.5in text block)
FULL = 5.5
HALF = 2.7

RC = {
    # Times-compatible serif to match the paper body
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # sizes assume the PDF is included near its natural width (no rescaling)
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    # geometry
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    # grid: horizontal only, drawn behind
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    # legend
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.0,
    "legend.handletextpad": 0.5,
    # export
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def use():
    import matplotlib
    matplotlib.rcParams.update(RC)
