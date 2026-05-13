#!/usr/bin/env python3
from typing import Dict, List, Optional
"""Unified plotter: reads all CSVs from data/ and writes all images to img/."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from picm_postpro.paths import DATA_DIR, IMG_DIR
from picm_postpro.plots import PALETTE, style_ax, style_legend, save_figure, parse_formats
from picm_postpro.core import read_csv

# ---------------------------------------------------------------------------
# Method coloring helpers
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "pic": PALETTE["blue"],
    "flip": PALETTE["orange"],
    "apic": PALETTE["green"],
    "mixed": PALETTE["purple"],
}
MIXED_COEF_COLORS = {
    0.01: PALETTE["purple"],
    0.05: PALETTE["pink"],
    0.1: PALETTE["grey"],
}


def method_label(method: str, flip_coef=None) -> str:
    if method == "pic":
        return "PIC"
    if method == "apic":
        return "APIC"
    coef = None if flip_coef in ("", None) else float(flip_coef)
    if coef is None or abs(coef) < 1e-12:
        return "FLIP"
    return f"FLIP α={coef:g}"


def method_color(method: str, flip_coef=None) -> str:
    if method == "pic":
        return METHOD_COLORS["pic"]
    if method == "apic":
        return METHOD_COLORS["apic"]
    coef = None if flip_coef in ("", None) else float(flip_coef)
    if coef is None or abs(coef) < 1e-12:
        return METHOD_COLORS["flip"]
    return MIXED_COEF_COLORS.get(coef, PALETTE["grey"])


def _try_float(val) -> Optional[float]:
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _load_or_warn(path: Path, name: str) -> Optional[List[dict]]:
    if not path.exists():
        print(f"[plot] skip {name}: {path} not found")
        return None
    rows = read_csv(path)
    if not rows:
        print(f"[plot] skip {name}: {path} is empty")
        return None
    return rows


# ---------------------------------------------------------------------------
# Plot 1: Kinetic energy time series
# ---------------------------------------------------------------------------

def plot_energy(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    csv_path = data_dir / "report" / "kinetic_energy.csv"
    rows = _load_or_warn(csv_path, "energyL2")
    if rows is None:
        return

    # Group by run key (method + flip_coef combo)
    # We need summary to get method/flip_coef per run
    summary_csv = data_dir / "report" / "summary.csv"
    summary = read_csv(summary_csv)
    run_meta: Dict[str, dict] = {r["run"]: r for r in summary}

    # Group rows by run
    from collections import defaultdict
    by_run: Dict[str, list] = defaultdict(list)
    for row in rows:
        run = row.get("run", "")
        t = _try_float(row.get("time"))
        ke = _try_float(row.get("kinetic_energy"))
        if t is not None and ke is not None:
            by_run[run].append((t, ke))

    if not by_run:
        print("[plot] energyL2: no data to plot")
        return

    fig, ax = plt.subplots()
    for run, data in sorted(by_run.items()):
        data.sort(key=lambda x: x[0])
        ts, kes = zip(*data)
        meta = run_meta.get(run, {})
        method = meta.get("method", "pic")
        coef = meta.get("flip_coef", None)
        color = method_color(method, coef)
        label = method_label(method, coef)
        ax.plot(ts, kes, color=color, label=label)

    style_ax(ax, xlabel="Time [s]", ylabel="Kinetic energy [J/m]", title="Kinetic Energy")
    style_legend(ax)
    out = img_dir / "energy" / "energyL2"
    save_figure(fig, out, formats=formats)
    plt.close(fig)
    print(f"[plot] wrote {out}.*")


# ---------------------------------------------------------------------------
# Plot 2: Vorticity L2 time series
# ---------------------------------------------------------------------------

def plot_vorticity(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    csv_path = data_dir / "report" / "vorticity.csv"
    rows = _load_or_warn(csv_path, "vorticityL2")
    if rows is None:
        return

    summary_csv = data_dir / "report" / "summary.csv"
    summary = read_csv(summary_csv)
    run_meta: Dict[str, dict] = {r["run"]: r for r in summary}

    from collections import defaultdict
    by_run: Dict[str, list] = defaultdict(list)
    for row in rows:
        run = row.get("run", "")
        t = _try_float(row.get("time"))
        vl2 = _try_float(row.get("vorticity_l2"))
        if t is not None and vl2 is not None:
            by_run[run].append((t, vl2))

    if not by_run:
        print("[plot] vorticityL2: no data to plot")
        return

    fig, ax = plt.subplots()
    for run, data in sorted(by_run.items()):
        data.sort(key=lambda x: x[0])
        ts, vl2s = zip(*data)
        meta = run_meta.get(run, {})
        method = meta.get("method", "pic")
        coef = meta.get("flip_coef", None)
        color = method_color(method, coef)
        label = method_label(method, coef)
        ax.plot(ts, vl2s, color=color, label=label)

    style_ax(ax, xlabel="Time [s]", ylabel="Vorticity L2", title="Vorticity")
    style_legend(ax)
    out = img_dir / "vorticity" / "vorticityL2"
    save_figure(fig, out, formats=formats)
    plt.close(fig)
    print(f"[plot] wrote {out}.*")


# ---------------------------------------------------------------------------
# Plot 3: Velocity L2 (free fall air study)
# ---------------------------------------------------------------------------

def plot_velocity(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    csv_path = data_dir / "free_fall" / "velocity.csv"
    rows = _load_or_warn(csv_path, "velocityL2")
    if rows is None:
        return

    from collections import defaultdict
    by_run: Dict[str, list] = defaultdict(list)
    for row in rows:
        run = row.get("run", "")
        t = _try_float(row.get("time"))
        v_med = _try_float(row.get("v_median"))
        v_theory = _try_float(row.get("v_theory"))
        if t is not None and v_med is not None:
            by_run[run].append({
                "t": t, "v_median": v_med, "v_theory": v_theory,
                "method": row.get("method", "pic"),
                "flip_coef": row.get("flip_coef", None),
            })

    if not by_run:
        print("[plot] velocityL2: no data to plot")
        return

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    theory_plotted = False
    for run, data in sorted(by_run.items()):
        data.sort(key=lambda x: x["t"])
        ts = [d["t"] for d in data]
        v_obs = [d["v_median"] for d in data]
        v_th = [d["v_theory"] for d in data if d["v_theory"] is not None]
        method = data[0]["method"]
        coef = data[0]["flip_coef"]
        color = method_color(method, coef)
        label = method_label(method, coef)

        ax.plot(ts, v_obs, color=color, label=label)

        # Residual on right axis
        if len(v_th) == len(ts):
            residuals = [obs - th for obs, th in zip(v_obs, v_th)]
            ax2.plot(ts, residuals, color=color, linestyle=":", alpha=0.6)

        # Theory line (once)
        if not theory_plotted and v_th:
            ts_full = [d["t"] for d in data if d["v_theory"] is not None]
            ax.plot(ts_full, v_th, "k--", linewidth=1.5, label=r"$v_0 + g\,t$")
            theory_plotted = True

    style_ax(ax, xlabel="Time [s]", ylabel="Particle velocity [m/s]",
             title="Free Fall: Velocity Validation")
    ax2.set_ylabel("Residual [m/s]")
    style_legend(ax)
    out = img_dir / "velocity" / "velocityL2"
    save_figure(fig, out, formats=formats)
    plt.close(fig)
    print(f"[plot] wrote {out}.*")


# ---------------------------------------------------------------------------
# Plot 4: Volume and particle count (water free fall)
# ---------------------------------------------------------------------------

def plot_volume_count(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    csv_path = data_dir / "free_fall" / "particle_count.csv"
    rows = _load_or_warn(csv_path, "volumeCount")
    if rows is None:
        return

    from collections import defaultdict
    by_run: Dict[str, list] = defaultdict(list)
    for row in rows:
        run = row.get("run", "")
        t = _try_float(row.get("time"))
        vol = _try_float(row.get("volume_m2"))
        n = _try_float(row.get("n_particles"))
        if t is not None and vol is not None and n is not None:
            by_run[run].append({
                "t": t, "volume_m2": vol, "n_particles": n,
                "method": row.get("method", "pic"),
                "ppc": row.get("ppc", ""),
                "flip_coef": row.get("flip_coef", None),
            })

    if not by_run:
        print("[plot] volumeCount: no data to plot")
        return

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    for run, data in sorted(by_run.items()):
        data.sort(key=lambda x: x["t"])
        ts = [d["t"] for d in data]
        vols = [d["volume_m2"] for d in data]
        ns = [d["n_particles"] for d in data]
        method = data[0]["method"]
        coef = data[0]["flip_coef"]
        ppc = data[0]["ppc"]
        color = method_color(method, coef)
        label = f"{method_label(method, coef)} ppc={ppc}"

        ax.plot(ts, vols, color=color, label=label)
        ax2.plot(ts, ns, color=color, linestyle="--", alpha=0.7)

    style_ax(ax, xlabel="Time [s]", ylabel="Fluid volume [m²]",
             title="Free Fall: Volume and Particle Count")
    ax2.set_ylabel("Particle count")
    style_legend(ax)
    out = img_dir / "volume" / "volumeCount"
    save_figure(fig, out, formats=formats)
    plt.close(fig)
    print(f"[plot] wrote {out}.*")


# ---------------------------------------------------------------------------
# Plot 5 & 6: PPC impact on final energy / vorticity
# ---------------------------------------------------------------------------

def _plot_ppc(
    data_dir: Path,
    img_dir: Path,
    formats: tuple,
    metric: str,
    col: str,
    ylabel: str,
    stem: str,
    title: str,
) -> None:
    csv_path = data_dir / "report" / "summary.csv"
    rows = _load_or_warn(csv_path, stem)
    if rows is None:
        return

    from collections import defaultdict
    series_data: Dict[tuple, dict] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        val = _try_float(row.get(col))
        ppc = _try_float(row.get("ppc"))
        if val is None or ppc is None:
            continue
        method = row.get("method", "pic")
        coef = row.get("flip_coef", None)
        key = (method, str(coef) if coef else "")
        if key not in series_data:
            series_data[key] = {}
        p = int(ppc)
        if p not in series_data[key]:
            series_data[key][p] = []
        series_data[key][p].append(val)

    if not series_data:
        print(f"[plot] {stem}: no data")
        return

    fig, ax = plt.subplots()
    for (method, coef_str), ppc_dict in sorted(series_data.items()):
        coef = _try_float(coef_str)
        color = method_color(method, coef)
        label = method_label(method, coef)
        ppcs = sorted(ppc_dict.keys())
        means = [np.mean(ppc_dict[p]) for p in ppcs]
        stds = [np.std(ppc_dict[p]) for p in ppcs]
        ax.errorbar(ppcs, means, yerr=stds, color=color, label=label,
                    marker="o", capsize=3)

    ax.set_xlim(0, 11)
    style_ax(ax, xlabel="Particles per cell", ylabel=ylabel, title=title)
    style_legend(ax)
    out = img_dir / "ppc" / stem
    save_figure(fig, out, formats=formats)
    plt.close(fig)
    print(f"[plot] wrote {out}.*")


def plot_ppc_energy(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    _plot_ppc(
        data_dir, img_dir, formats,
        metric="energy", col="final_kinetic_energy",
        ylabel="Final kinetic energy [J/m]",
        stem="ppc_energyL2",
        title="PPC Impact: Kinetic Energy",
    )


def plot_ppc_vorticity(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    _plot_ppc(
        data_dir, img_dir, formats,
        metric="vorticity", col="final_vorticity_l2",
        ylabel="Final vorticity L2",
        stem="ppc_vorticityL2",
        title="PPC Impact: Vorticity",
    )


# ---------------------------------------------------------------------------
# Plot 7: Iterative solver comparison
# ---------------------------------------------------------------------------

def plot_iterative(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    csv_path = data_dir / "iterative" / "iterations.csv"
    rows = _load_or_warn(csv_path, "iterative")
    if rows is None:
        return

    from collections import defaultdict
    by_solver: Dict[str, list] = defaultdict(list)
    hit_max_by_solver: Dict[str, list] = defaultdict(list)

    for row in rows:
        solver = row.get("solver", "unknown")
        step = _try_float(row.get("step"))
        iters = _try_float(row.get("iterations"))
        if step is None or iters is None:
            continue
        hit_max = str(row.get("hit_max", "False")).lower() in ("true", "1", "yes")
        by_solver[solver].append((int(step), iters))
        if hit_max:
            hit_max_by_solver[solver].append((int(step), iters))

    if not by_solver:
        print("[plot] iterative: no data to plot")
        return

    # Check if log scale is needed
    all_iters = [v for data in by_solver.values() for _, v in data]
    use_log = (max(all_iters) / max(min(all_iters), 1)) > 50 if all_iters else False

    # Assign colors from palette cycling
    palette_vals = list(PALETTE.values())
    fig, ax = plt.subplots()
    for i, (solver, data) in enumerate(sorted(by_solver.items())):
        data.sort(key=lambda x: x[0])
        steps, iters = zip(*data)
        color = palette_vals[i % len(palette_vals)]
        ax.plot(steps, iters, color=color, label=solver)

        # Mark hit_max steps
        hm = hit_max_by_solver.get(solver, [])
        if hm:
            hm_steps, hm_iters = zip(*hm)
            ax.plot(hm_steps, hm_iters, "o", color=color, markersize=6)

    if use_log:
        ax.set_yscale("log")

    style_ax(ax, xlabel="Time step", ylabel="Iterations", title="Pressure Solver: Iteration Count")
    style_legend(ax)
    out = img_dir / "iterative" / "iterative"
    save_figure(fig, out, formats=formats)
    plt.close(fig)
    print(f"[plot] wrote {out}.*")


# ---------------------------------------------------------------------------
# Plot 8: Scaling (strong and weak)
# ---------------------------------------------------------------------------

def _plot_scaling(
    data_dir: Path,
    img_dir: Path,
    formats: tuple,
    study: str,
) -> None:
    csv_path = data_dir / "scaling" / "scaling.csv"
    rows = _load_or_warn(csv_path, f"scaling/{study}")
    if rows is None:
        return

    filtered = [r for r in rows if r.get("study") == study]
    if not filtered:
        print(f"[plot] scaling/{study}: no rows for study='{study}'")
        return

    from collections import defaultdict
    by_binding: Dict[str, list] = defaultdict(list)
    for row in filtered:
        binding = row.get("binding", "")
        threads = _try_float(row.get("threads"))
        wall = _try_float(row.get("wall_time_s"))
        if threads is None or wall is None:
            continue
        by_binding[binding].append((int(threads), wall))

    if not by_binding:
        print(f"[plot] scaling/{study}: no valid data")
        return

    palette_vals = list(PALETTE.values())
    fig, ax = plt.subplots()
    for i, (binding, data) in enumerate(sorted(by_binding.items())):
        data.sort(key=lambda x: x[0])
        threads, walls = zip(*data)
        color = palette_vals[i % len(palette_vals)]
        ax.plot(threads, walls, color=color, label=binding, marker="o")

    style_ax(ax, xlabel="Threads", ylabel="Wall time [s]",
             title=f"{study.capitalize()} Scaling")
    style_legend(ax)
    out = img_dir / "scaling" / study
    save_figure(fig, out, formats=formats)
    plt.close(fig)
    print(f"[plot] wrote {out}.*")


def plot_scaling_strong(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    _plot_scaling(data_dir, img_dir, formats, "strong")


def plot_scaling_weak(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    _plot_scaling(data_dir, img_dir, formats, "weak")


# ---------------------------------------------------------------------------
# Plot 9: Velocity time series per method (PIC / FLIP / APIC / mixed-FLIP)
# ---------------------------------------------------------------------------

def plot_velocity_methods(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    """Velocity L2 vs time for every method variant — Strouhal-style comparison."""
    ke_csv = data_dir / "report" / "kinetic_energy.csv"
    sum_csv = data_dir / "report" / "summary.csv"
    ke_rows = _load_or_warn(ke_csv, "methods/velocityMethods")
    if ke_rows is None:
        return

    from collections import defaultdict
    run_meta: Dict[str, dict] = {}
    for r in read_csv(sum_csv):
        run_meta[r["run"]] = r

    # Aggregate time → mean velocity_l2 per (method, flip_coef) key
    series: Dict[tuple, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in ke_rows:
        run = row.get("run", "")
        t = _try_float(row.get("time"))
        v = _try_float(row.get("velocity_l2"))
        if t is None or v is None:
            continue
        meta = run_meta.get(run, {})
        method = meta.get("method", row.get("method", "pic"))
        coef = meta.get("flip_coef", row.get("flip_coef", None))
        key = (method, coef if coef not in ("", None) else None)
        series[key][t].append(v)

    if not series:
        print("[plot] methods/velocityMethods: no data")
        return

    # Sort keys: pic < flip-pure < flip-mixed... < apic
    order = {"pic": 0, "apic": 1, "flip": 2}
    def _sort_key(key):
        m, c = key
        base = order.get(m, 9)
        coef_val = float(c) if c not in (None, "", "None") else 0.0
        # pure FLIP (coef=0) before mixed; apic last
        if m == "apic":
            return (3, 0.0)
        if m == "pic":
            return (0, 0.0)
        return (1, coef_val)

    fig, ax = plt.subplots()
    for key in sorted(series.keys(), key=_sort_key):
        method, coef = key
        coef_val = float(coef) if coef not in (None, "", "None") else None
        color = method_color(method, coef_val)
        label = method_label(method, coef_val)
        bucket = series[key]
        ts = sorted(bucket.keys())
        vs = [sum(bucket[t]) / len(bucket[t]) for t in ts]
        ax.plot(ts, vs, color=color, label=label, lw=1.5)

    style_ax(ax,
             xlabel=r"Time $t$ [s]",
             ylabel=r"$\|\mathbf{v}\|_{L^2}$ [m/s]",
             title="Method comparison: velocity")
    style_legend(ax)
    out = img_dir / "methods" / "velocityMethods"
    save_figure(fig, out, formats=formats)
    plt.close(fig)
    print(f"[plot] wrote {out}.*")


# ---------------------------------------------------------------------------
# Plot 10: Wake-point velocity vs time (von-Kármán, method comparison)
# ---------------------------------------------------------------------------

def plot_vk_point(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    """Two plots from wake_point.csv:
      - wakePoint_methods : PIC / FLIP (coef=0) / APIC
      - wakePoint_flip    : FLIP α=0.01 / 0.05 / 0.1
    """
    csv_path = data_dir / "vk_point" / "wake_point.csv"
    rows = _load_or_warn(csv_path, "vk_point/wakePoint")
    if rows is None:
        return

    from collections import defaultdict
    series: Dict[tuple, list] = defaultdict(list)
    for row in rows:
        t = _try_float(row.get("time"))
        v = _try_float(row.get("normVelocity"))
        if t is None or v is None:
            continue
        method = row.get("method", "pic")
        coef = row.get("flip_coef", None)
        coef_val = _try_float(coef)
        key = (method, coef_val)
        series[key].append((t, v))

    if not series:
        print("[plot] vk_point/wakePoint: no data")
        return

    def _plot_subset(keys, stem, title):
        keys = [k for k in keys if k in series]
        if not keys:
            print(f"[plot] vk_point/{stem}: no matching data")
            return
        fig, ax = plt.subplots()
        for method, coef_val in keys:
            color = method_color(method, coef_val)
            label = method_label(method, coef_val)
            data = sorted(series[(method, coef_val)], key=lambda x: x[0])
            ts, vs = zip(*data)
            ax.plot(ts, vs, color=color, label=label, lw=1.5)
        style_ax(ax,
                 xlabel=r"Time $t$ [s]",
                 ylabel=r"$\|\mathbf{v}\|$ [m/s]",
                 title=title)
        style_legend(ax)
        out = img_dir / "vk_point" / stem
        save_figure(fig, out, formats=formats)
        plt.close(fig)
        print(f"[plot] wrote {out}.*")

    # Plot 1: PIC vs pure FLIP (coef=0) vs APIC
    _plot_subset(
        [("pic", None), ("pic", 0.0), ("flip", None), ("flip", 0.0), ("apic", None), ("apic", 0.0)],
        "wakePoint_methods",
        r"Wake point $(\frac{L_x}{2},\,\frac{L_y}{4})$: PIC / FLIP / APIC",
    )

    # Plot 2: FLIP α variations only
    flip_coef_keys = sorted(
        [(m, c) for (m, c) in series if m == "flip" and c is not None and c > 1e-12],
        key=lambda k: k[1],
    )
    _plot_subset(
        flip_coef_keys,
        "wakePoint_flip",
        r"Wake point $(\frac{L_x}{2},\,\frac{L_y}{4})$: FLIP $\alpha$ variations",
    )


# ---------------------------------------------------------------------------
# Plot: Rankine vortex — numerical viscosity analysis
# ---------------------------------------------------------------------------

def plot_rankine(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    """Four plots from rankine/:
      KE time series (methods & FLIP variants): normalised kinetic energy E_k(t)/E_k(0)
      Radial profile (methods & FLIP variants): u_θ(r) at t_final vs analytical Rankine
    """
    from collections import defaultdict
    out_base = img_dir / "rankine"
    out_base.mkdir(parents=True, exist_ok=True)

    # ---- 1 & 2: Kinetic energy time series ----
    ke_csv = data_dir / "rankine" / "kinetic_energy.csv"
    ke_rows = _load_or_warn(ke_csv, "rankine/kinetic_energy")

    if ke_rows is not None:
        ke_series: Dict[tuple, list] = defaultdict(list)
        for row in ke_rows:
            t = _try_float(row.get("time"))
            ke = _try_float(row.get("ke_norm"))
            if t is None or ke is None:
                continue
            method = row.get("method", "pic")
            coef_val = _try_float(row.get("flip_coef"))
            ke_series[(method, coef_val)].append((t, ke))

        t_max = max(t for pts in ke_series.values() for t, _ in pts) if ke_series else 1.0

        def _plot_ke(keys, stem, title):
            keys = [k for k in keys if k in ke_series]
            if not keys:
                print(f"[plot] rankine/{stem}: no data")
                return
            fig, ax = plt.subplots()
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                       label="Analytical (inviscid)", zorder=0)
            for method, coef_val in keys:
                pts = sorted(ke_series[(method, coef_val)], key=lambda x: x[0])
                ts, kes = zip(*pts)
                ax.plot(ts, kes, color=method_color(method, coef_val),
                        label=method_label(method, coef_val), lw=1.8)
            ax.set_xlim(0, t_max)
            ax.set_ylim(0, 1.1)
            style_ax(ax, xlabel=r"Time $t$ [s]",
                     ylabel=r"$E_k(t)\,/\,E_k(0)$", title=title)
            style_legend(ax)
            save_figure(fig, out_base / stem, formats=formats)
            plt.close(fig)
            print(f"[plot] wrote {out_base / stem}.*")

        _plot_ke(
            [("pic", None), ("pic", 0.0), ("flip", None), ("flip", 0.0),
             ("apic", None), ("apic", 0.0)],
            "rankine_ke_methods",
            r"Rankine vortex: kinetic energy decay — PIC / FLIP / APIC",
        )
        flip_ke_keys = sorted(
            [(m, c) for (m, c) in ke_series if m == "flip" and c is not None and c > 1e-12],
            key=lambda k: k[1],
        )
        _plot_ke(
            flip_ke_keys,
            "rankine_ke_flip",
            r"Rankine vortex: kinetic energy decay — FLIP $\alpha$ variations",
        )

    # ---- 3 & 4: Azimuthal velocity profile at t_final ----
    rp_csv = data_dir / "rankine" / "radial_profile.csv"
    rp_rows = _load_or_warn(rp_csv, "rankine/radial_profile")

    if rp_rows is not None:
        rp_series: Dict[tuple, list] = defaultdict(list)
        for row in rp_rows:
            r = _try_float(row.get("r"))
            vt = _try_float(row.get("v_theta_sim"))
            if r is None or vt is None:
                continue
            method = row.get("method", "pic")
            coef_val = _try_float(row.get("flip_coef"))
            rp_series[(method, coef_val)].append((r, vt))

        # Reconstruct analytical profile from data range
        if rp_series:
            all_r = [r for pts in rp_series.values() for r, _ in pts]
            r_analytical = np.linspace(0, max(all_r), 400)
            # Infer core_r from analytical peak location if needed;
            # use hard-coded config defaults (core_r=60 cells * dx=0.02 = 1.2 m, omega=2.5)
            OMEGA, CORE_R = 2.5, 1.2
            vt_analytical = np.where(
                r_analytical <= CORE_R,
                OMEGA * r_analytical,
                OMEGA * CORE_R ** 2 / np.maximum(r_analytical, 1e-12),
            )

        def _plot_rp(keys, stem, title):
            keys = [k for k in keys if k in rp_series]
            if not keys:
                print(f"[plot] rankine/{stem}: no data")
                return
            fig, ax = plt.subplots()
            ax.plot(r_analytical, vt_analytical, color="black", linestyle="--",
                    linewidth=1.4, label="Analytical (Rankine)", zorder=0)
            for method, coef_val in keys:
                pts = sorted(rp_series[(method, coef_val)], key=lambda x: x[0])
                rs, vts = zip(*pts)
                ax.plot(rs, vts, color=method_color(method, coef_val),
                        label=method_label(method, coef_val), lw=1.8)
            style_ax(ax, xlabel=r"$r$ [m]",
                     ylabel=r"$u_\theta(r)$ [m/s]", title=title)
            style_legend(ax)
            save_figure(fig, out_base / stem, formats=formats)
            plt.close(fig)
            print(f"[plot] wrote {out_base / stem}.*")

        _plot_rp(
            [("pic", None), ("pic", 0.0), ("flip", None), ("flip", 0.0),
             ("apic", None), ("apic", 0.0)],
            "rankine_profile_methods",
            r"Rankine vortex: azimuthal velocity at $t_\mathrm{final}$ — PIC / FLIP / APIC",
        )
        flip_rp_keys = sorted(
            [(m, c) for (m, c) in rp_series if m == "flip" and c is not None and c > 1e-12],
            key=lambda k: k[1],
        )
        _plot_rp(
            flip_rp_keys,
            "rankine_profile_flip",
            r"Rankine vortex: azimuthal velocity at $t_\mathrm{final}$ — FLIP $\alpha$ variations",
        )

    # ---- 5 & 6: σ²(t) → effective numerical viscosity ----
    sigma_csv = data_dir / "rankine" / "sigma.csv"
    sigma_rows_all = _load_or_warn(sigma_csv, "rankine/sigma")

    if sigma_rows_all is not None:
        sig_series: Dict[tuple, list] = defaultdict(list)
        for row in sigma_rows_all:
            t  = _try_float(row.get("time"))
            se = _try_float(row.get("sigma_sq_excess"))
            if t is None or se is None:
                continue
            method   = row.get("method", "pic")
            coef_val = _try_float(row.get("flip_coef"))
            sig_series[(method, coef_val)].append((t, se))

        # Physical viscosity reference slopes (ν in m²/s)
        # σ²_excess(t) = 4ν t
        _VISCOSITIES = {
            r"Water ($\nu=10^{-6}$)":   1e-6,
            r"Air ($\nu=1.5\times10^{-5}$)": 1.5e-5,
            r"Engine oil ($\nu=10^{-4}$)": 1e-4,
        }

        def _fit_nu_eff(pts):
            """Linear fit σ²_excess = 4ν t → return ν_eff."""
            if len(pts) < 3:
                return float("nan")
            ts = np.array([p[0] for p in pts])
            ses = np.array([p[1] for p in pts])
            # weighted least-squares through origin: ν_eff = Σ(t·se)/(4·Σ(t²))
            nu = float(np.dot(ts, ses) / (4.0 * np.dot(ts, ts))) if np.dot(ts, ts) > 0 else float("nan")
            return nu

        def _plot_sigma(keys, stem, title):
            keys = [k for k in keys if k in sig_series]
            if not keys:
                print(f"[plot] rankine/{stem}: no data")
                return

            t_max_s = max(t for k in keys for t, _ in sig_series[k])
            t_ref   = np.linspace(0, t_max_s, 200)

            fig, ax = plt.subplots()

            # Reference physical viscosity lines (thin, grey shades)
            ref_colors = ["#aaaaaa", "#888888", "#555555"]
            for (label, nu), rc in zip(_VISCOSITIES.items(), ref_colors):
                ax.plot(t_ref, 4 * nu * t_ref, color=rc, lw=1.0,
                        linestyle=":", label=label, zorder=0)

            # Simulation lines
            for method, coef_val in keys:
                pts = sorted(sig_series[(method, coef_val)], key=lambda x: x[0])
                ts, ses = zip(*pts)
                nu_eff = _fit_nu_eff(pts)
                label  = method_label(method, coef_val)
                if not math.isnan(nu_eff):
                    label += f"\n$\\nu_{{eff}}={nu_eff:.2e}$ m²/s"
                ax.plot(ts, ses, color=method_color(method, coef_val),
                        label=label, lw=1.8)

            ax.axhline(0, color="black", linestyle="--", linewidth=1.0,
                       label="Analytical (inviscid)", zorder=0)
            ax.set_xlim(0, t_max_s)
            style_ax(ax,
                     xlabel=r"Time $t$ [s]",
                     ylabel=r"$\sigma^2(t) - \sigma^2(0)$ [m²]",
                     title=title)
            style_legend(ax)
            save_figure(fig, out_base / stem, formats=formats)
            plt.close(fig)
            print(f"[plot] wrote {out_base / stem}.*")

        _plot_sigma(
            [("pic", None), ("pic", 0.0), ("flip", None), ("flip", 0.0),
             ("apic", None), ("apic", 0.0)],
            "rankine_viscosity_methods",
            r"Rankine vortex: effective numerical viscosity — PIC / FLIP / APIC",
        )
        flip_sig_keys = sorted(
            [(m, c) for (m, c) in sig_series if m == "flip" and c is not None and c > 1e-12],
            key=lambda k: k[1],
        )
        _plot_sigma(
            flip_sig_keys,
            "rankine_viscosity_flip",
            r"Rankine vortex: effective numerical viscosity — FLIP $\alpha$ variations",
        )


# ---------------------------------------------------------------------------
# Plot: Lamb-Oseen vortex — analytical viscosity comparison
# ---------------------------------------------------------------------------

# Lamb-Oseen config constants (must match test/PIC/extra/lamb-oseen-vortex.json)
_LO_NU    = 0.002   # physical kinematic viscosity [m²/s]
_LO_T0    = 20.0    # physical_time: simulation starts at this vortex age [s]
_LO_OMEGA = 2.0     # angular velocity
_LO_RC    = 0.4     # core_r in metres (20 cells × 0.02 m/cell)
_LO_GAMMA = 2.0 * math.pi * _LO_OMEGA * _LO_RC ** 2  # circulation Γ


def plot_lamb(data_dir: Path, img_dir: Path, formats: tuple) -> None:
    """Two plots from lamb/:
      lamb_sigma  — σ²(t) for each method vs exact Lamb-Oseen analytical curve
      lamb_profile — u_θ(r) at t_final for each method vs analytical at t=0 and t_final
    Analytical:
      σ²(t_sim) = 4ν(t₀ + t_sim)
      u_θ(r,t)  = (Γ/2πr)·(1 − exp(−r²/σ²(t)))
    """
    from collections import defaultdict
    out_base = img_dir / "lamb"
    out_base.mkdir(parents=True, exist_ok=True)

    # ---- Plot 1 & 2: σ²(t) vs analytical ----
    sig_csv  = data_dir / "lamb" / "sigma.csv"
    sig_rows = _load_or_warn(sig_csv, "lamb/sigma")

    if sig_rows is not None:
        sig_series: Dict[tuple, list] = defaultdict(list)
        for row in sig_rows:
            t  = _try_float(row.get("time"))
            sq = _try_float(row.get("sigma_sq"))
            if t is None or sq is None:
                continue
            method   = row.get("method", "pic")
            coef_val = _try_float(row.get("flip_coef"))
            sig_series[(method, coef_val)].append((t, sq))

        t_max = max(t for pts in sig_series.values() for t, _ in pts) if sig_series else 1.0
        t_ana = np.linspace(0, t_max, 300)
        sigma_sq_analytical = 4.0 * _LO_NU * (_LO_T0 + t_ana)

        def _plot_sig(keys, stem, title):
            keys = [k for k in keys if k in sig_series]
            if not keys:
                return
            fig, ax = plt.subplots()
            ax.plot(t_ana, sigma_sq_analytical, color="black", linestyle="--",
                    linewidth=1.4,
                    label=rf"Analytical $\sigma^2=4\nu(t_0+t)$, $\nu={_LO_NU}$",
                    zorder=0)
            for method, coef_val in keys:
                pts = sorted(sig_series[(method, coef_val)], key=lambda x: x[0])
                ts, sqs = zip(*pts)
                ax.plot(ts, sqs, color=method_color(method, coef_val),
                        label=method_label(method, coef_val), lw=1.8)
            ax.set_xlim(0, t_max)
            style_ax(ax,
                     xlabel=r"Simulation time $t$ [s]",
                     ylabel=r"$\sigma^2(t)$ [m²]",
                     title=title)
            style_legend(ax)
            save_figure(fig, out_base / stem, formats=formats)
            plt.close(fig)
            print(f"[plot] wrote {out_base / stem}.*")

        _plot_sig(
            [("pic", None), ("pic", 0.0), ("flip", None), ("flip", 0.0),
             ("apic", None), ("apic", 0.0)],
            "lamb_sigma_methods",
            r"Lamb-Oseen vortex: $\sigma^2(t)$ — PIC / FLIP / APIC",
        )
        flip_sig_keys = sorted(
            [(m, c) for (m, c) in sig_series if m == "flip" and c is not None and c > 1e-12],
            key=lambda k: k[1],
        )
        _plot_sig(
            flip_sig_keys,
            "lamb_sigma_flip",
            r"Lamb-Oseen vortex: $\sigma^2(t)$ — FLIP $\alpha$ variations",
        )

        # ---- Plot 3 & 4: μ(t) = σ²(t) / 4(t₀+t) — instantaneous effective viscosity ----
        mu_series: Dict[tuple, list] = defaultdict(list)
        for (key, pts) in sig_series.items():
            for t, sq in pts:
                denom = 4.0 * (_LO_T0 + t)
                if denom > 1e-12:
                    mu_series[key].append((t, sq / denom))

        def _plot_mu(keys, stem, title):
            keys = [k for k in keys if k in mu_series]
            if not keys:
                return
            fig, ax = plt.subplots()
            ax.axhline(_LO_NU, color="black", linestyle="--", linewidth=1.4,
                       label=rf"Analytical $\nu = {_LO_NU}$ m²/s", zorder=0)
            for method, coef_val in keys:
                pts = sorted(mu_series[(method, coef_val)], key=lambda x: x[0])
                ts, mus = zip(*pts)
                ax.plot(ts, mus, color=method_color(method, coef_val),
                        label=method_label(method, coef_val), lw=1.8)
            ax.set_xlim(0, t_max)
            style_ax(ax,
                     xlabel=r"Simulation time $t$ [s]",
                     ylabel=r"$\mu(t) = \sigma^2/(4(t_0+t))$ [m²/s]",
                     title=title)
            style_legend(ax)
            save_figure(fig, out_base / stem, formats=formats)
            plt.close(fig)
            print(f"[plot] wrote {out_base / stem}.*")

        _plot_mu(
            [("pic", None), ("pic", 0.0), ("flip", None), ("flip", 0.0),
             ("apic", None), ("apic", 0.0)],
            "lamb_mu_methods",
            r"Lamb-Oseen: effective viscosity $\mu(t)$ — PIC / FLIP / APIC",
        )
        _plot_mu(
            flip_sig_keys,
            "lamb_mu_flip",
            r"Lamb-Oseen: effective viscosity $\mu(t)$ — FLIP $\alpha$ variations",
        )

        # ---- Plot 5 & 6: ν_eff(t) = (σ²(t) − σ²(0)) / 4t — cumulative numerical diffusion ----
        # σ²(0) from each method's first data point
        sig0: Dict[tuple, float] = {}
        for key, pts in sig_series.items():
            pts_sorted = sorted(pts, key=lambda x: x[0])
            if pts_sorted:
                sig0[key] = pts_sorted[0][1]

        nu_eff_series: Dict[tuple, list] = defaultdict(list)
        for key, pts in sig_series.items():
            sq0 = sig0.get(key)
            if sq0 is None:
                continue
            for t, sq in pts:
                if t > 1e-10:
                    nu_eff_series[key].append((t, (sq - sq0) / (4.0 * t)))

        def _plot_nu_eff(keys, stem, title):
            keys = [k for k in keys if k in nu_eff_series]
            if not keys:
                return
            fig, ax = plt.subplots()
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0,
                       label="Perfect inviscid ($\\nu_{eff}=0$)", zorder=0)
            ax.axhline(_LO_NU, color="black", linestyle="--", linewidth=1.4,
                       label=rf"Physical $\nu = {_LO_NU}$ m²/s", zorder=0)
            for method, coef_val in keys:
                pts = sorted(nu_eff_series[(method, coef_val)], key=lambda x: x[0])
                ts, nus = zip(*pts)
                ax.plot(ts, nus, color=method_color(method, coef_val),
                        label=method_label(method, coef_val), lw=1.8)
            ax.set_xlim(0, t_max)
            style_ax(ax,
                     xlabel=r"Simulation time $t$ [s]",
                     ylabel=r"$\nu_{eff}(t) = (\sigma^2(t)-\sigma^2(0))/(4t)$ [m²/s]",
                     title=title)
            style_legend(ax)
            save_figure(fig, out_base / stem, formats=formats)
            plt.close(fig)
            print(f"[plot] wrote {out_base / stem}.*")

        _plot_nu_eff(
            [("pic", None), ("pic", 0.0), ("flip", None), ("flip", 0.0),
             ("apic", None), ("apic", 0.0)],
            "lamb_nueff_methods",
            r"Lamb-Oseen: numerical diffusion $\nu_{eff}(t)$ — PIC / FLIP / APIC",
        )
        _plot_nu_eff(
            flip_sig_keys,
            "lamb_nueff_flip",
            r"Lamb-Oseen: numerical diffusion $\nu_{eff}(t)$ — FLIP $\alpha$ variations",
        )

    # ---- Plot 5 & 6: u_θ(r) at t_final vs analytical ----
    rp_csv  = data_dir / "lamb" / "radial_profile.csv"
    rp_rows = _load_or_warn(rp_csv, "lamb/radial_profile")

    if rp_rows is not None:
        rp_series: Dict[tuple, list] = defaultdict(list)
        t_final_val = 0.0
        for row in rp_rows:
            r   = _try_float(row.get("r"))
            vt  = _try_float(row.get("v_theta_sim"))
            if r is None or vt is None:
                continue
            method   = row.get("method", "pic")
            coef_val = _try_float(row.get("flip_coef"))
            rp_series[(method, coef_val)].append((r, vt))
            t_final_val = max(t_final_val, _try_float(row.get("time")) or 0.0)

        if rp_series:
            all_r = sorted({r for pts in rp_series.values() for r, _ in pts})
            r_arr = np.array(all_r)
            sig_sq_t0   = 4.0 * _LO_NU * _LO_T0
            sig_sq_tfin = 4.0 * _LO_NU * (_LO_T0 + t_final_val)

            def _vtheta_ana(r_arr, sig_sq):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.where(r_arr > 1e-12,
                        (_LO_GAMMA / (2.0 * math.pi * r_arr))
                        * (1.0 - np.exp(-r_arr ** 2 / sig_sq)),
                        0.0)

        def _plot_rp(keys, stem, title):
            keys = [k for k in keys if k in rp_series]
            if not keys:
                return
            fig, ax = plt.subplots()
            ax.plot(r_arr, _vtheta_ana(r_arr, sig_sq_t0), color="black",
                    linestyle=":", linewidth=1.2,
                    label=r"Analytical $t=0$", zorder=0)
            ax.plot(r_arr, _vtheta_ana(r_arr, sig_sq_tfin), color="black",
                    linestyle="--", linewidth=1.4,
                    label=rf"Analytical $t_{{final}}={t_final_val:.1f}$ s", zorder=0)
            for method, coef_val in keys:
                pts = sorted(rp_series[(method, coef_val)], key=lambda x: x[0])
                rs, vts = zip(*pts)
                ax.plot(rs, vts, color=method_color(method, coef_val),
                        label=method_label(method, coef_val), lw=1.8)
            style_ax(ax, xlabel=r"$r$ [m]",
                     ylabel=r"$u_\theta(r)$ [m/s]", title=title)
            style_legend(ax)
            save_figure(fig, out_base / stem, formats=formats)
            plt.close(fig)
            print(f"[plot] wrote {out_base / stem}.*")

        _plot_rp(
            [("pic", None), ("pic", 0.0), ("flip", None), ("flip", 0.0),
             ("apic", None), ("apic", 0.0)],
            "lamb_profile_methods",
            r"Lamb-Oseen vortex: $u_\theta(r)$ at $t_\mathrm{final}$ — PIC / FLIP / APIC",
        )
        flip_rp_keys = sorted(
            [(m, c) for (m, c) in rp_series if m == "flip" and c is not None and c > 1e-12],
            key=lambda k: k[1],
        )
        _plot_rp(
            flip_rp_keys,
            "lamb_profile_flip",
            r"Lamb-Oseen vortex: $u_\theta(r)$ at $t_\mathrm{final}$ — FLIP $\alpha$ variations",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--img", type=Path, default=IMG_DIR)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    formats = parse_formats(args.formats)
    data_dir = args.data
    img_dir = args.img

    plot_energy(data_dir, img_dir, formats)
    plot_vorticity(data_dir, img_dir, formats)
    plot_velocity(data_dir, img_dir, formats)
    plot_volume_count(data_dir, img_dir, formats)
    plot_velocity_methods(data_dir, img_dir, formats)
    plot_ppc_energy(data_dir, img_dir, formats)
    plot_ppc_vorticity(data_dir, img_dir, formats)
    plot_iterative(data_dir, img_dir, formats)
    plot_scaling_strong(data_dir, img_dir, formats)
    plot_scaling_weak(data_dir, img_dir, formats)
    plot_vk_point(data_dir, img_dir, formats)
    plot_rankine(data_dir, img_dir, formats)
    plot_lamb(data_dir, img_dir, formats)


    print("[plot] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
