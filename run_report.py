#!/usr/bin/env python3
from typing import List, Optional, Tuple
"""Run PIC/FLIP/APIC comparison studies and write CSVs. No plotting."""

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from picm_postpro.paths import DATA_DIR, PICM_ROOT
from picm_postpro.core import (
    build_binary,
    drop_run,
    optional_float,
    parse_pvd,
    read_csv,
    read_vti_field,
    read_vtp_point_count,
    run_binary,
    scheduler_threads,
    write_csv,
)

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

TESTS = {
    "falling-block-water": PICM_ROOT / "test" / "PIC" / "extra" / "freeFallInWater.json",
    "freeFallInWater": PICM_ROOT / "test" / "PIC" / "extra" / "freeFallInWater.json",
    "dambreak": PICM_ROOT / "test" / "PIC" / "extra" / "dambreak.json",
    "von-karman": PICM_ROOT / "test" / "PIC" / "section-5-5-1" / "von-karman.json",
    "vases-communicants": (
        PICM_ROOT / "test" / "PIC" / "extra" / "vases-communicants" / "vases-communicants.json"
    ),
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _run_name(test: str, method: str, ppc: int, flip_coef: Optional[float], threads: int, repeat: int) -> str:
    name = f"{test}_{method}_ppc{ppc}"
    if method == "flip" and flip_coef is not None:
        name += f"_coefpic{flip_coef:g}"
    name += f"_t{threads}_r{repeat}"
    return name


def _build_config(
    base: dict,
    method: str,
    ppc: int,
    flip_coef: Optional[float],
    raw_dir: Path,
    analysis: str,
    write_particles: bool,
    sampling_rate: int,
    overrides: dict,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["method"] = method
    cfg["ppcx"] = ppc
    cfg["ppcy"] = ppc
    cfg["folder"] = str(raw_dir)
    cfg["filename"] = "simulation"
    cfg["sampling_rate"] = sampling_rate

    if method == "flip" and flip_coef is not None:
        cfg["coefpic"] = flip_coef

    # Write control
    cfg["write_norm_velocity"] = True
    cfg["write_u"] = analysis == "vorticity"
    cfg["write_v"] = analysis == "vorticity"
    cfg["write_vorticity"] = analysis == "vorticity"
    cfg["write_particles"] = write_particles
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_smoke"] = False

    for key, val in overrides.items():
        cfg[key] = val

    return cfg


# ---------------------------------------------------------------------------
# VTI extraction
# ---------------------------------------------------------------------------

def _extract_kinetic_energy(pvd_path: Path, dx: float, dy: float) -> List[Tuple[int, float, float, float]]:
    """Returns list of (step, time, kinetic_energy, velocity_l2)."""
    if not pvd_path.exists():
        return []
    import xml.etree.ElementTree as ET
    tree = ET.parse(pvd_path)
    rows = []
    for i, dataset in enumerate(tree.getroot().iter("DataSet")):
        t = float(dataset.get("timestep", i))
        fpath = pvd_path.parent / dataset.get("file", "")
        if not fpath.exists():
            continue
        try:
            field = read_vti_field(fpath, "normVelocity")
        except (KeyError, Exception):
            continue
        v2 = np.sum(field ** 2) * dx * dy
        l2 = math.sqrt(float(np.sum(field ** 2)) * dx * dy)
        rows.append((i, t, float(v2), l2))
    return rows


def _compute_vorticity_from_uv(u_pvd: Path, v_pvd: Path, dx: float, dy: float) -> List[Tuple[int, float, float, float]]:
    """Compute vorticity from u and v fields."""
    import xml.etree.ElementTree as ET
    if not u_pvd.exists() or not v_pvd.exists():
        return []
    u_tree = ET.parse(u_pvd)
    v_tree = ET.parse(v_pvd)
    u_datasets = list(u_tree.getroot().iter("DataSet"))
    v_datasets = list(v_tree.getroot().iter("DataSet"))
    rows = []
    for i, (u_ds, v_ds) in enumerate(zip(u_datasets, v_datasets)):
        t = float(u_ds.get("timestep", i))
        u_path = u_pvd.parent / u_ds.get("file", "")
        v_path = v_pvd.parent / v_ds.get("file", "")
        if not u_path.exists() or not v_path.exists():
            continue
        try:
            u = read_vti_field(u_path, "u")
            v = read_vti_field(v_path, "v")
        except (KeyError, Exception):
            continue
        # omega = dv/dx - du/dy (finite differences on staggered grid)
        dv_dx = np.gradient(v, dx, axis=1)
        du_dy = np.gradient(u, dy, axis=0)
        omega = dv_dx - du_dy
        vort_l2 = math.sqrt(float(np.sum(omega ** 2)) * dx * dy)
        enstrophy = float(np.sum(omega ** 2)) * dx * dy * 0.5
        rows.append((i, t, vort_l2, enstrophy))
    return rows


def _extract_vorticity(raw_dir: Path, dx: float, dy: float) -> List[Tuple[int, float, float, float]]:
    """Try vorticity.pvd directly, fall back to u+v computation."""
    vort_pvd = raw_dir / "vorticity.pvd"
    if vort_pvd.exists():
        import xml.etree.ElementTree as ET
        tree = ET.parse(vort_pvd)
        rows = []
        for i, dataset in enumerate(tree.getroot().iter("DataSet")):
            t = float(dataset.get("timestep", i))
            fpath = vort_pvd.parent / dataset.get("file", "")
            if not fpath.exists():
                continue
            try:
                field = read_vti_field(fpath, "vorticity")
            except (KeyError, Exception):
                continue
            vort_l2 = math.sqrt(float(np.sum(field ** 2)) * dx * dy)
            enstrophy = float(np.sum(field ** 2)) * dx * dy * 0.5
            rows.append((i, t, vort_l2, enstrophy))
        if rows:
            return rows
    # Fall back to u+v
    u_pvd = raw_dir / "u.pvd"
    v_pvd = raw_dir / "v.pvd"
    return _compute_vorticity_from_uv(u_pvd, v_pvd, dx, dy)


def _extract_particle_counts(pvd_path: Path) -> List[Tuple[int, float, int]]:
    """Returns (step, time, n_particles) from particles.pvd."""
    if not pvd_path.exists():
        return []
    import xml.etree.ElementTree as ET
    tree = ET.parse(pvd_path)
    rows = []
    for i, dataset in enumerate(tree.getroot().iter("DataSet")):
        t = float(dataset.get("timestep", i))
        fpath = pvd_path.parent / dataset.get("file", "")
        if not fpath.exists():
            continue
        n = read_vtp_point_count(fpath)
        rows.append((i, t, n))
    return rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_one(
    binary: Path,
    name: str,
    config: dict,
    run_dir: Path,
    threads: int,
    dry_run: bool,
) -> Tuple[str, float]:
    """Write config, run binary, return (status, wall_time_s)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / f"{name}.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    if dry_run:
        print(f"[dry-run] would run: {binary} {config_path}")
        return "dry-run", 0.0

    env = {"OMP_NUM_THREADS": str(threads)}
    cmd = [str(binary), str(config_path)]
    print(f"[run] {name} (threads={threads})")
    t0 = time.perf_counter()
    result = run_binary(cmd, env)
    wall_time = time.perf_counter() - t0

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    stdout_log.write_bytes(result.stdout)
    stderr_log.write_bytes(result.stderr)

    if result.returncode != 0:
        print(f"[run] FAILED {name} (exit {result.returncode})")
        return "failed", wall_time

    print(f"[run] OK {name} ({wall_time:.1f}s)")
    return "ok", wall_time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="falling-block-water",
                        choices=list(TESTS.keys()),
                        help="test configuration to use")
    parser.add_argument("--methods", default="pic,flip,apic",
                        help="comma-separated list of methods")
    parser.add_argument("--ppc", default="3",
                        help="comma-separated list of ppc values")
    parser.add_argument("--flip-coef", default="0,0.01,0.05,0.1",
                        help="comma-separated coefpic values for FLIP")
    parser.add_argument("--analysis", choices=["energy", "vorticity"], default="vorticity")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--samples", type=int, default=40,
                        help="number of output samples (sets sampling_rate = nt // samples)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-release")
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="re-run already completed runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--write-particles", action="store_true",
                        help="also write VTP particle frames")
    parser.add_argument("--keep-raw", action="store_true",
                        help="keep raw VTI/VTP output after extraction")
    args = parser.parse_args()

    threads = args.threads if args.threads is not None else scheduler_threads()
    build_jobs = args.build_jobs if args.build_jobs is not None else scheduler_threads()
    out_dir = args.out if args.out is not None else DATA_DIR / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve binary
    binary = args.binary
    if binary is None or not binary.exists():
        print("[build] binary not found, building...")
        binary = build_binary(args.build_dir, build_jobs, skip=False, build_type="Release")
    binary = binary.resolve()

    # Load base config
    config_path = TESTS[args.test]
    if not config_path.exists():
        print(f"[error] config not found: {config_path}", file=sys.stderr)
        return 1
    with open(config_path) as fh:
        base_config = json.load(fh)

    # Config overrides
    overrides: dict = {}
    if args.nt is not None:
        overrides["nt"] = args.nt
    if args.nx is not None:
        overrides["nx"] = args.nx
    if args.ny is not None:
        overrides["ny"] = args.ny
    if args.dt is not None:
        overrides["dt"] = args.dt

    nt = overrides.get("nt", base_config.get("nt", 1000))
    sampling_rate = max(1, nt // args.samples)

    dx = float(base_config.get("dx", 0.05))
    dy = float(base_config.get("dy", 0.05))

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ppc_list = [int(p.strip()) for p in args.ppc.split(",") if p.strip()]
    flip_coefs = [float(c.strip()) for c in args.flip_coef.split(",") if c.strip()]

    # Load existing CSVs
    summary_csv = out_dir / "summary.csv"
    energy_csv = out_dir / "kinetic_energy.csv"
    vorticity_csv = out_dir / "vorticity.csv"
    particle_csv = out_dir / "particle_count.csv"

    summary_rows = read_csv(summary_csv)
    energy_rows = read_csv(energy_csv)
    vorticity_rows = read_csv(vorticity_csv)
    particle_rows = read_csv(particle_csv)

    completed_runs = {
        r["run"] for r in summary_rows if r.get("status") == "ok"
    }

    runs_dir = out_dir / "runs"

    # Build run list
    run_specs = []
    for method in methods:
        for ppc in ppc_list:
            if method == "flip":
                coef_list = flip_coefs
            else:
                coef_list = [None]
            for coef in coef_list:
                for repeat in range(1, args.repeats + 1):
                    name = _run_name(args.test, method, ppc, coef, threads, repeat)
                    run_specs.append((name, method, ppc, coef, repeat))

    for name, method, ppc, coef, repeat in run_specs:
        if not args.force and name in completed_runs:
            print(f"[skip] {name} (already completed)")
            continue

        run_dir = runs_dir / name
        raw_dir = run_dir / "raw"

        config = _build_config(
            base_config,
            method,
            ppc,
            coef,
            raw_dir,
            args.analysis,
            args.write_particles,
            sampling_rate,
            overrides,
        )

        status, wall_time = run_one(binary, name, config, run_dir, threads, args.dry_run)

        if args.dry_run:
            continue

        # Extract data
        final_ke: Optional[float] = None
        final_vel_l2: Optional[float] = None
        final_vort_l2: Optional[float] = None

        # Remove old rows for this run
        energy_rows = drop_run(energy_rows, name)
        vorticity_rows = drop_run(vorticity_rows, name)
        particle_rows = drop_run(particle_rows, name)
        summary_rows = drop_run(summary_rows, name)

        if status == "ok" and np is not None:
            # Kinetic energy
            norm_pvd = raw_dir / "normVelocity.pvd"
            ke_data = _extract_kinetic_energy(norm_pvd, dx, dy)
            for step, t, ke, vel_l2 in ke_data:
                energy_rows.append({
                    "run": name, "step": step, "time": t,
                    "kinetic_energy": ke, "velocity_l2": vel_l2,
                })
            if ke_data:
                final_ke = ke_data[-1][2]
                final_vel_l2 = ke_data[-1][3]

            # Vorticity
            if args.analysis == "vorticity":
                vort_data = _extract_vorticity(raw_dir, dx, dy)
                for step, t, vort_l2, enstrophy in vort_data:
                    vorticity_rows.append({
                        "run": name, "step": step, "time": t,
                        "vorticity_l2": vort_l2, "enstrophy": enstrophy,
                    })
                if vort_data:
                    final_vort_l2 = vort_data[-1][2]

            # Particle count
            if args.write_particles:
                part_pvd = raw_dir / "particles.pvd"
                part_data = _extract_particle_counts(part_pvd)
                volume_per_particle = dx * dy / (ppc * ppc)
                for step, t, n_part in part_data:
                    particle_rows.append({
                        "run": name, "test": args.test,
                        "method": method, "ppc": ppc,
                        "step": step, "time": t,
                        "n_particles": n_part,
                        "volume_m2": n_part * volume_per_particle,
                    })
        elif status == "ok" and np is None:
            print(f"[warn] numpy not available — raw output kept in {raw_dir} for later extraction")
            status = "raw_pending"

        # Summary row
        summary_rows.append({
            "run": name,
            "test": args.test,
            "method": method,
            "ppc": ppc,
            "flip_coef": coef if coef is not None else "",
            "threads": threads,
            "repeat": repeat,
            "status": status,
            "wall_time_s": f"{wall_time:.3f}",
            "final_kinetic_energy": final_ke if final_ke is not None else "",
            "final_velocity_l2": final_vel_l2 if final_vel_l2 is not None else "",
            "final_vorticity_l2": final_vort_l2 if final_vort_l2 is not None else "",
        })

        # Write CSVs incrementally
        write_csv(summary_csv, summary_rows)
        write_csv(energy_csv, energy_rows)
        if args.analysis == "vorticity":
            write_csv(vorticity_csv, vorticity_rows)
        if args.write_particles:
            write_csv(particle_csv, particle_rows)

        # Cleanup raw only when extraction succeeded
        if status == "ok" and not args.keep_raw and raw_dir.exists():
            shutil.rmtree(raw_dir)

    print(f"[done] results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
