#!/usr/bin/env python3
from typing import List, Optional, Tuple
"""Run free-fall studies and write CSVs. No plotting.

Two studies:
  1. Air free-fall  (freeFall.json)      — validates v = v0 + g*t
  2. Water free-fall (freeFallInWater.json) — tracks particle count / volume
"""

import argparse
import copy
import json
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
    read_csv,
    read_vtp_point_array,
    read_vtp_point_count,
    run_binary,
    scheduler_threads,
    write_csv,
)

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

AIR_CONFIG = PICM_ROOT / "test" / "PIC" / "extra" / "freeFall.json"
WATER_CONFIG = PICM_ROOT / "test" / "PIC" / "extra" / "freeFallInWater.json"

GRAVITY = 9.81  # default; overridden by config "gravity" if present


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_name(study: str, method: str, ppc: int, flip_coef: Optional[float], threads: int) -> str:
    name = f"{study}_{method}_ppc{ppc}"
    if method == "flip" and flip_coef is not None:
        name += f"_coefpic{flip_coef:g}"
    name += f"_t{threads}"
    return name


def _sample_time(sample_i: int, dt: float, sampling_rate: int) -> float:
    return float(sample_i) * float(dt) * float(sampling_rate)


def _build_config(
    base: dict,
    method: str,
    ppc: int,
    flip_coef: Optional[float],
    raw_dir: Path,
    nt: int,
    sampling_rate: int,
    write_particles: bool = True,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["method"] = method
    cfg["ppcx"] = ppc
    cfg["ppcy"] = ppc
    cfg["folder"] = str(raw_dir)
    cfg["filename"] = "simulation"
    cfg["nt"] = nt
    cfg["sampling_rate"] = sampling_rate

    if method == "flip" and flip_coef is not None:
        cfg["coefpic"] = flip_coef

    cfg["write_particles"] = write_particles
    cfg["write_u"] = False
    cfg["write_v"] = False
    cfg["write_norm_velocity"] = False
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_smoke"] = False
    cfg["write_vorticity"] = False

    return cfg


def _run_sim(
    binary: Path,
    name: str,
    config: dict,
    run_dir: Path,
    threads: int,
    dry_run: bool = False,
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
    wall = time.perf_counter() - t0

    (run_dir / "stdout.log").write_bytes(result.stdout)
    (run_dir / "stderr.log").write_bytes(result.stderr)

    if result.returncode != 0:
        print(f"[run] FAILED {name} (exit {result.returncode})")
        return "failed", wall

    print(f"[run] OK {name} ({wall:.1f}s)")
    return "ok", wall


# ---------------------------------------------------------------------------
# Air study extraction
# ---------------------------------------------------------------------------

def _extract_air_velocity(
    pvd_path: Path,
    gravity: float,
    v0: float,
    dt: float,
    sampling_rate: int,
) -> List[dict]:
    """Extract per-frame velocity statistics from particles.pvd (air study)."""
    if not pvd_path.exists():
        return []

    import xml.etree.ElementTree as ET
    tree = ET.parse(pvd_path)
    rows = []
    for sample_i, dataset in enumerate(tree.getroot().iter("DataSet")):
        t = _sample_time(sample_i, dt, sampling_rate)
        fpath = pvd_path.parent / dataset.get("file", "")
        if not fpath.exists():
            continue

        # Try to get velocityY directly, fall back to normVelocity
        try:
            vy = read_vtp_point_array(fpath, "velocityY")
        except (KeyError, Exception):
            try:
                vy = read_vtp_point_array(fpath, "normVelocity")
            except (KeyError, Exception):
                vy = np.array([])

        n_part = read_vtp_point_count(fpath)
        if len(vy) == 0 or n_part == 0:
            v_mean = float("nan")
            v_median = float("nan")
            v_p05 = float("nan")
            v_p95 = float("nan")
        else:
            vy_fin = vy[np.isfinite(vy)]
            if len(vy_fin) == 0:
                v_mean = v_median = v_p05 = v_p95 = float("nan")
            else:
                v_mean = float(np.mean(vy_fin))
                v_median = float(np.median(vy_fin))
                v_p05 = float(np.percentile(vy_fin, 5))
                v_p95 = float(np.percentile(vy_fin, 95))

        v_theory = v0 - gravity * t
        rows.append({
            "sample": sample_i,
            "step": sample_i,
            "time": t,
            "v_mean": v_mean,
            "v_median": v_median,
            "v_p05": v_p05,
            "v_p95": v_p95,
            "n_particles": n_part,
            "v_theory": v_theory,
            "v0": v0,
        })
    return rows


# ---------------------------------------------------------------------------
# Water study extraction
# ---------------------------------------------------------------------------

def _extract_water_counts(
    pvd_path: Path,
    dx: float,
    dy: float,
    ppc: int,
    dt: float,
    sampling_rate: int,
) -> List[dict]:
    """Extract particle count and volume from particles.pvd."""
    if not pvd_path.exists():
        return []

    import xml.etree.ElementTree as ET
    tree = ET.parse(pvd_path)
    volume_per_particle = dx * dy / (ppc * ppc)
    rows = []
    for sample_i, dataset in enumerate(tree.getroot().iter("DataSet")):
        t = _sample_time(sample_i, dt, sampling_rate)
        fpath = pvd_path.parent / dataset.get("file", "")
        if not fpath.exists():
            continue
        n = read_vtp_point_count(fpath)
        rows.append({
            "sample": sample_i,
            "step": sample_i,
            "time": t,
            "n_particles": n,
            "volume_m2": n * volume_per_particle,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="pic,flip,apic",
                        help="comma-separated methods")
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--flip-coef", type=float, default=0.0,
                        help="coefpic for FLIP method")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--nt-air", type=int, default=300)
    parser.add_argument("--nt-water", type=int, default=500)
    parser.add_argument("--sampling-rate", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-release")
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-raw", "--raw", action="store_true",
                        help="keep raw VTP output after CSV extraction")
    parser.add_argument("--skip-air", action="store_true")
    parser.add_argument("--skip-water", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    threads = args.threads if args.threads is not None else scheduler_threads()
    build_jobs = args.build_jobs if args.build_jobs is not None else scheduler_threads()
    out_dir = args.out if args.out is not None else DATA_DIR / "free_fall"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve binary
    binary = args.binary
    if binary is None or not binary.exists():
        print("[build] binary not found, building...")
        binary = build_binary(args.build_dir, build_jobs, skip=False, build_type="Release")
    binary = binary.resolve()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    vel_csv = out_dir / "velocity.csv"
    part_csv = out_dir / "particle_count.csv"
    vel_rows = read_csv(vel_csv)
    part_rows = read_csv(part_csv)

    runs_dir = out_dir / "runs"

    # -----------------------------------------------------------------------
    # Air study
    # -----------------------------------------------------------------------
    if not args.skip_air:
        if not AIR_CONFIG.exists():
            print(f"[warn] air config not found: {AIR_CONFIG}", file=sys.stderr)
        else:
            with open(AIR_CONFIG) as fh:
                air_base = json.load(fh)

            gravity = float(air_base.get("gravity", GRAVITY))
            dx = float(air_base.get("dx", 0.05))
            dy = float(air_base.get("dy", 0.05))
            dt = float(air_base.get("dt", 0.01))
            v0 = 0.0  # initial downward velocity

            completed_air = {r["run"] for r in vel_rows if r.get("run", "").startswith("air_")}

            for method in methods:
                coef = args.flip_coef if method == "flip" else None
                name = _run_name("air", method, args.ppc, coef, threads)

                run_dir = runs_dir / name
                raw_dir = run_dir / "raw"

                if not args.force and name in completed_air:
                    if not args.keep_raw or raw_dir.exists():
                        print(f"[skip] {name}")
                        continue
                    print(f"[rerun] {name} (raw output missing)")

                config = _build_config(
                    air_base, method, args.ppc, coef,
                    raw_dir, args.nt_air, args.sampling_rate,
                    write_particles=True,
                )

                status, _ = _run_sim(binary, name, config, run_dir, threads, args.dry_run)

                if args.dry_run or status != "ok":
                    continue

                if np is None:
                    print(f"[warn] numpy not available — raw output kept in {raw_dir}")
                    continue

                # Extract velocity data
                part_pvd = raw_dir / "particles.pvd"
                extracted = _extract_air_velocity(
                    part_pvd, gravity, v0, dt, args.sampling_rate
                )

                # Remove old rows for this run and append new
                vel_rows = [r for r in vel_rows if r.get("run") != name]
                for row in extracted:
                    vel_rows.append({
                        "method": method,
                        "ppc": args.ppc,
                        "flip_coef": coef if coef is not None else "",
                        "run": name,
                        **row,
                    })

                write_csv(vel_csv, vel_rows)

                # Cleanup raw
                if not args.keep_raw and raw_dir.exists():
                    shutil.rmtree(raw_dir)

    # -----------------------------------------------------------------------
    # Water study
    # -----------------------------------------------------------------------
    if not args.skip_water:
        if not WATER_CONFIG.exists():
            print(f"[warn] water config not found: {WATER_CONFIG}", file=sys.stderr)
        else:
            with open(WATER_CONFIG) as fh:
                water_base = json.load(fh)

            dx = float(water_base.get("dx", 0.05))
            dy = float(water_base.get("dy", 0.05))
            dt = float(water_base.get("dt", 0.01))

            completed_water = {r["run"] for r in part_rows if r.get("run", "").startswith("water_")}

            for method in methods:
                coef = args.flip_coef if method == "flip" else None
                name = _run_name("water", method, args.ppc, coef, threads)

                run_dir = runs_dir / name
                raw_dir = run_dir / "raw"

                if not args.force and name in completed_water:
                    if not args.keep_raw or raw_dir.exists():
                        print(f"[skip] {name}")
                        continue
                    print(f"[rerun] {name} (raw output missing)")

                config = _build_config(
                    water_base, method, args.ppc, coef,
                    raw_dir, args.nt_water, args.sampling_rate,
                    write_particles=True,
                )

                status, _ = _run_sim(binary, name, config, run_dir, threads, args.dry_run)

                if args.dry_run or status != "ok":
                    continue

                if np is None:
                    print(f"[warn] numpy not available — raw output kept in {raw_dir}")
                    continue

                # Extract particle count / volume data
                part_pvd = raw_dir / "particles.pvd"
                extracted = _extract_water_counts(
                    part_pvd, dx, dy, args.ppc, dt, args.sampling_rate
                )

                # Remove old rows and append new
                part_rows = [r for r in part_rows if r.get("run") != name]
                for row in extracted:
                    part_rows.append({
                        "method": method,
                        "ppc": args.ppc,
                        "flip_coef": coef if coef is not None else "",
                        "run": name,
                        **row,
                    })

                write_csv(part_csv, part_rows)

                # Cleanup raw
                if not args.keep_raw and raw_dir.exists():
                    shutil.rmtree(raw_dir)

    print(f"[done] results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
