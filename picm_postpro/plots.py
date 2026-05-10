from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

DEFAULT_FORMATS = ("png", "pdf")

# ---------------------------------------------------------------------------
# Load utilsStyle from the PostPro root (parent of this package directory).
# Importing it triggers apply_style() at module level, setting all rcParams.
# ---------------------------------------------------------------------------
_postpro_root = Path(__file__).resolve().parents[1]
if str(_postpro_root) not in sys.path:
    sys.path.insert(0, str(_postpro_root))

try:
    import utilsStyle as _us
    PALETTE = _us.PALETTE
    style_ax = _us.style_ax
    style_legend = _us.style_legend
except ImportError:  # fallback if running outside the PostPro tree
    PALETTE = {
        "blue":   "#0057E7",
        "orange": "#F28E2B",
        "green":  "#1f9e89",
        "purple": "#8E44AD",
        "grey":   "#5f7fa2",
        "pink":   "#CB80B1",
    }

    def style_ax(ax, xlabel="", ylabel="", title="", grid=True, minorticks=True):
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if grid:
            ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.55)
        if minorticks:
            ax.minorticks_on()

    def style_legend(ax, title=None, loc="best", many_threshold=3, max_columns=3, **kwargs):
        handles, labels = ax.get_legend_handles_labels()
        visible = [l for l in labels if l and not str(l).startswith("_")]
        if not visible:
            return None
        if len(visible) >= many_threshold:
            ncol = min(max_columns, len(visible))
            return ax.legend(title=title, loc="lower center",
                             bbox_to_anchor=(0.5, 1.02), ncol=ncol,
                             borderaxespad=0.0, **kwargs)
        return ax.legend(title=title, loc=loc, **kwargs)


def parse_formats(value: Optional[Union[str, Iterable[str]]]) -> Tuple[str, ...]:
    if value is None:
        return DEFAULT_FORMATS
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    formats = tuple(dict.fromkeys(item.strip().lstrip(".") for item in items if item.strip()))
    if not formats:
        raise ValueError("empty image format list")
    return formats


def save_figure(fig, output_stem: Path, *, formats: Iterable[str] = DEFAULT_FORMATS) -> None:
    """Save fig to each format; dpi and bbox come from rcParams (utilsStyle)."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for image_format in formats:
        fig.savefig(output_stem.with_suffix(f".{image_format}"))
