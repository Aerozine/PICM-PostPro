#!/usr/bin/env python3
"""Render lamb-vorticity and rankine-normVelocity VTI fields as clean PNG images.

Output: PI-final-report/7-FLAPIC/comparison/lamb-{pic,flip,apic}-field.png
                                              rankine-{pic,flip,apic}-field.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from picm_postpro.core import read_vti_field
from picm_postpro.paths import DATA_DIR, POSTPRO_ROOT

OUT_DIR = Path("/home/lpd/cours/master/pi/PI-final-report/7-FLAPIC/comparison")


def last_vti(raw_dir: Path, prefix: str) -> Path:
    files = sorted(raw_dir.glob(f"{prefix}_*.vti"))
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.vti in {raw_dir}")
    return files[-1]


def render_field(
    arr: np.ndarray,
    out_path: Path,
    cmap: str,
    vmin=None,
    vmax=None,
) -> None:
    """Write a scalar field as a clean RGB PNG with no axes, border, or whitespace."""
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())

    # Normalise to [0, 1] and apply colourmap  →  RGBA float array
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    rgba = cm(norm(arr))            # shape (ny, nx, 4), float64 in [0, 1]

    # Composite over white background and convert to uint8 RGB
    alpha = rgba[..., 3:4]
    rgb_f = rgba[..., :3] * alpha + (1 - alpha)   # white background
    rgb_u8 = (rgb_f * 255).clip(0, 255).astype(np.uint8)

    # Flip to image convention (row 0 = top)
    rgb_u8 = rgb_u8[::-1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save via PIL to guarantee RGB PNG with no extra chrome
    from PIL import Image
    Image.fromarray(rgb_u8, mode="RGB").save(str(out_path))
    print(f"[render] {out_path.name}")


# ---------------------------------------------------------------------------
# Lamb-Oseen vorticity  (RdBu_r, symmetric)
# ---------------------------------------------------------------------------

LAMB_RUNS = {
    "pic":  DATA_DIR / "lamb" / "runs" / "lamb_pic_ppc3_t4"  / "raw",
    "flip": DATA_DIR / "lamb" / "runs" / "lamb_flip_ppc3_coefpic0_t4" / "raw",
    "apic": DATA_DIR / "lamb" / "runs" / "lamb_apic_ppc3_t4" / "raw",
}

# Compute a global symmetric colour range from all three runs so the colour
# scale is consistent across the three panels.
vort_arrays = {}
for method, raw_dir in LAMB_RUNS.items():
    if not raw_dir.exists():
        print(f"[skip] {raw_dir} not found")
        continue
    try:
        vti = last_vti(raw_dir, "vorticity")
        vort_arrays[method] = read_vti_field(vti, "vorticity")
    except Exception as e:
        print(f"[warn] {method}: {e}")

if vort_arrays:
    abs_max = max(float(np.abs(arr).max()) for arr in vort_arrays.values())
    for method, arr in vort_arrays.items():
        render_field(
            arr,
            OUT_DIR / f"lamb-{method}-field.png",
            cmap="RdBu_r",
            vmin=-abs_max,
            vmax=abs_max,
        )

# ---------------------------------------------------------------------------
# Rankine vortex — norm velocity  (plasma, 0 → max)
# ---------------------------------------------------------------------------

RANKINE_RUNS = {
    "pic":  DATA_DIR / "rankine" / "runs" / "rankine_pic_ppc2_t4"              / "raw",
    "flip": DATA_DIR / "rankine" / "runs" / "rankine_flip_ppc2_coefpic0_t4"   / "raw",
    "apic": DATA_DIR / "rankine" / "runs" / "rankine_apic_ppc2_t4"             / "raw",
}

vel_arrays = {}
for method, raw_dir in RANKINE_RUNS.items():
    if not raw_dir.exists():
        print(f"[skip] {raw_dir} not found")
        continue
    try:
        vti = last_vti(raw_dir, "normVelocity")
        vel_arrays[method] = read_vti_field(vti, "normVelocity")
    except Exception as e:
        print(f"[warn] {method}: {e}")

if vel_arrays:
    v_max = max(float(arr.max()) for arr in vel_arrays.values())
    for method, arr in vel_arrays.items():
        render_field(
            arr,
            OUT_DIR / f"rankine-{method}-field.png",
            cmap="plasma",
            vmin=0.0,
            vmax=v_max,
        )

print("[done]")
