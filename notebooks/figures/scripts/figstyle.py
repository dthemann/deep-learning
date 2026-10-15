"""Shared matplotlib style for the companion-guide figures.

Usage (from any figure script):

    from figstyle import *          # palette, apply_style(), save()
    apply_style()
    fig, ax = plt.subplots()
    ...
    save(fig, "01", "dot_product")  # -> notebooks/figures/01/dot_product.png

Design rules baked in: one colorblind-safe categorical order (assigned in fixed
order, never cycled), a single-hue sequential map for magnitudes, a blue-red
diverging map for signed quantities, thin marks, recessive grid, no dual axes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_ROOT = Path(__file__).resolve().parent.parent          # notebooks/figures

# Categorical palette (fixed order; use SERIES[0] for the first series, etc.)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

# Chrome and ink
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_MID = "#f0efec"

# Status colours (reserved for good / warning / serious / critical, never for a series)
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

# Sequential (one hue, light -> dark) and diverging (blue <-> neutral <-> red) colour maps
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("course_seq", SEQ_STEPS)
DIV_CMAP = LinearSegmentedColormap.from_list("course_div", ["#0d366b", "#2a78d6", NEUTRAL_MID, "#e34948", "#7a1f1f"])
# A second sequential ramp (orange) for when two magnitudes must be shown side by side
SEQ2_CMAP = LinearSegmentedColormap.from_list("course_seq2", ["#fde5d9", "#f7b89b", "#eb6834", "#9e3a12"])


def apply_style():
    """Apply the shared look to matplotlib."""
    plt.rcParams.update({
        "figure.figsize": (7.5, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "text.color": INK,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "image.cmap": "course_seq",
    })
    try:
        matplotlib.colormaps.register(SEQ_CMAP, name="course_seq")
        matplotlib.colormaps.register(DIV_CMAP, name="course_div")
        matplotlib.colormaps.register(SEQ2_CMAP, name="course_seq2")
    except ValueError:
        pass  # already registered


def save(fig, nb, name, tight=True):
    """Save a figure as notebooks/figures/<nb>/<name>.png and close it. Returns the relative path."""
    out_dir = FIG_ROOT / nb
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    rel = f"figures/{nb}/{name}.png"
    print("saved", rel)
    return rel


def bar(ax, labels, values, color=BLUE, width=0.6, label_fmt=None, **kw):
    """Thin bars with a small gap and optional direct value labels."""
    bars = ax.bar(labels, values, width=width, color=color, edgecolor=SURFACE, linewidth=1.5, **kw)
    if label_fmt:
        for b, v in zip(bars, values):
            ax.annotate(label_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK_2, xytext=(0, 2), textcoords="offset points")
    return bars
