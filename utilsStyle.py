"""
utilsStyle.py  —  Palette, style rcParams et helpers bas niveau
===============================================================
Importé automatiquement par utilsPoint.py et utilsField.py.
Ne pas modifier sauf pour changer les couleurs ou la typographie.

PALETTE
-------
    PALETTE["blue"]        #3A6FA8   bleu acier   (primaire)
    PALETTE["red"]         #C94040   rouge chaud  (secondaire)
    PALETTE["blue_light"]  #8BB4D8   bleu clair
    PALETTE["red_light"]   #E08A8A   rouge clair
    PALETTE["grey"]        #6B6B6B   gris neutre
    PALETTE["green"]       #3A8A5A   vert forêt
    PALETTE["orange"]      #D4742A   orange ambre
    PALETTE["purple"]      #6A4FA8   violet sourd
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# ============================================================
#  PALETTE
# ============================================================
"""
PALETTE = {
    "orange":     "#D4742A",
    "green":      "#3A8A5A",
    "purple":     "#6A4FA8",
    "color1":     "#1f9e89",
    "color2":     "#5f7fa2",
    "blue":       "#107BED",
    "red":        "#E71F1F",
    }
"""
PALETTE = {
    "orange": "#F28E2B",  # remplace le rose
    "purple": "#8E44AD",  # remplace le rouge
    "grey":     "#5f7fa2",
    "green":     "#1f9e89",
    "blue":       "#0057E7",
    "pink":     "#CB80B1",
}

COLORS = [
    PALETTE["green"],
    PALETTE["orange"],
    PALETTE["orange"],
    PALETTE["blue"],
    PALETTE["purple"],
    PALETTE["grey"],
]


COLORS = list(PALETTE.values())

INDIVIDUAL_TITLE_SIZE  = 16
INDIVIDUAL_LABEL_SIZE  = 16
INDIVIDUAL_TICK_SIZE   = 14
INDIVIDUAL_LEGEND_SIZE = 12

# ============================================================
#  STYLE GLOBAL
# ============================================================

def apply_style():
    plt.rcParams.update({
        "mathtext.fontset":  "cm",
        "figure.figsize":    (9.5, 6.0),
        "figure.dpi":        130,
        "savefig.dpi":       400,
        "savefig.bbox":      "tight",
        "font.size":         INDIVIDUAL_TICK_SIZE,
        "axes.labelsize":    INDIVIDUAL_LABEL_SIZE,
        "axes.titlesize":    INDIVIDUAL_TITLE_SIZE,
        "legend.fontsize":   INDIVIDUAL_LEGEND_SIZE,
        "xtick.labelsize":   INDIVIDUAL_TICK_SIZE,
        "ytick.labelsize":   INDIVIDUAL_TICK_SIZE,
        "axes.linewidth":    1.2,
        "lines.linewidth":   2.2,
        "lines.markersize":  4.8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "xtick.major.size":  6,
        "ytick.major.size":  6,
        "xtick.minor.size":  3,
        "ytick.minor.size":  3,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "legend.frameon":    True,
        "legend.framealpha": 1.0,
        "legend.edgecolor":  "black",
        "axes.prop_cycle":   cycler(color=COLORS),
    })

apply_style()


# ============================================================
#  HELPERS BAS NIVEAU  (réutilisés dans Point et Field)
# ============================================================

def style_ax(ax, xlabel="", ylabel="", title="", grid=True, minorticks=True):
    """Labels, grille et ticks secondaires sur un Axes existant."""
    if xlabel:    ax.set_xlabel(xlabel)
    if ylabel:    ax.set_ylabel(ylabel)
    if title:     ax.set_title(title)
    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.grid(True, which="minor", linestyle=":",  linewidth=0.5, alpha=0.35)
    if minorticks:
        ax.minorticks_on()

'''
def add_hline(ax, y, label=None, color="black", linestyle="--", linewidth=1.5):
    """Ligne horizontale de référence."""
    ax.axhline(y, linestyle=linestyle, linewidth=linewidth, color=color,
               label=label if label else f"ref = {y:g}", zorder=1)

'''
def add_hline(ax, y, label=None, color="black", linestyle="--", linewidth=1.5):
    ax.axhline(
        y,
        linestyle=linestyle,
        linewidth=linewidth,
        color=color,
        label="_nolegend_" if label is None else label,
        zorder=1,
    )

def style_legend(
    ax,
    title=None,
    loc="best",
    many_threshold=3,
    max_columns=3,
    **kwargs,
):
    """Place dense legends above the axes, otherwise use the requested location."""
    handles, labels = ax.get_legend_handles_labels()
    visible_labels = [
        label for label in labels
        if label and not str(label).startswith("_")
    ]
    if not visible_labels:
        return None

    if len(visible_labels) >= many_threshold:
        ncol = min(max_columns, len(visible_labels))
        return ax.legend(
            title=title,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            borderaxespad=0.0,
            **kwargs,
        )

    return ax.legend(title=title, loc=loc, **kwargs)

def finalize(fig, save_path=None, show=True):
    """tight_layout, sauvegarde PDF+PNG optionnelle, affichage."""
    fig.tight_layout()
    if save_path:
        base = str(save_path).replace(".pdf", "").replace(".png", "")
        fig.savefig(base + ".pdf")
        fig.savefig(base + ".png")
        print(f"  ✓ Saved : {base}.pdf / .png")
    if show and "agg" not in plt.get_backend().lower():
        plt.show()
