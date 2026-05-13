#!/usr/bin/env python3
from typing import List, Optional, Tuple
"""Run von-Kármán simulations and sample ||v|| at the wake point (nx/2, ny/4).

Produces:
  data/vk_point/wake_point.csv           — combined, all methods
  data/vk_point/per_method/<run>.csv     — one CSV per run (time, normVelocity)
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
    drop_run,
    read_csv,
    read_vti_field,
    run_binary,
    scheduler_threads,
    write_csv,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VK_CONFIG = PICM_ROOT / "test" / "PIC" / "section-5-5-1" / "von-karman.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_name(method: str, ppc: int, flip_coef: Optional[float], threads: int) -> str:
    name = f"vk_{method}_ppc{ppc}"
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
    overrides: dict,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["method"] = method
    cfg["ppcx"] = ppc
    cfg["ppcy"] = ppc
    cfg["folder"] = str(raw_dir.resolve())
    cfg["filename"] = "simulation"
    cfg["nt"] = nt
    cfg["sampling_rate"] = sampling_rate

    if method == "flip" and flip_coef is not None:
        cfg["coefpic"] = flip_coef

    # Only write normVelocity — we only need it for the wake-point extraction
    cfg["write_norm_velocity"] = True
    cfg["write_u"] = False
    cfg["write_v"] = False
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_smoke"] = False
    cfg["write_vorticity"] = False
    cfg["write_particles"] = False

    for key, val in overrides.items():
        cfg[key] = val

    return cfg


def _run_sim(
    binary: Path,
    name: str,
    config: dict,
    run_dir: Path,
    threads: int,
    dry_run: bool,
) -> Tuple[str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)

    config_path = (run_dir / f"{name}.json").resolve()
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
# Extraction
# ---------------------------------------------------------------------------

def _extract_wake_point(
    pvd_path: Path,
    nx: int,
    ny: int,
    dt: float,
    sampling_rate: int,
) -> List[dict]:
    """
    Sample normVelocity at grid cell (nx//2, ny//4) from each VTI frame.
    Returns list of {sample, time, normVelocity}.
    """
    if not pvd_path.exists():
        return []

    import xml.etree.ElementTree as ET
    tree = ET.parse(pvd_path)
    # Wake point: midway along channel, lower-wake quarter
    pi = nx // 2
    pj = ny // 4

    rows = []
    for sample_i, dataset in enumerate(tree.getroot().iter("DataSet")):
        t = _sample_time(sample_i, dt, sampling_rate)
        fpath = pvd_path.parent / dataset.get("file", "")
        if not fpath.exists():
            continue
        try:
            field = read_vti_field(fpath, "normVelocity")
            val = float(field[pj, pi])
        except Exception:
            val = float("nan")

        rows.append({"sample": sample_i, "time": t, "normVelocity": val})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="pic,flip,apic",
                        help="comma-separated methods")
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--flip-coef", default="0,0.01,0.05,0.1",
                        help="comma-separated coefpic values for FLIP")
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--samples", type=int, default=200,
                        help="number of output frames")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-release")
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-raw", "--raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    threads = args.threads if args.threads is not None else scheduler_threads()
    build_jobs = args.build_jobs if args.build_jobs is not None else scheduler_threads()
    out_dir = args.out if args.out is not None else DATA_DIR / "vk_point"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_method_dir = out_dir / "per_method"
    per_method_dir.mkdir(parents=True, exist_ok=True)

    # Resolve binary
    binary = args.binary
    if binary is None or not binary.exists():
        print("[build] binary not found, building...")
        from picm_postpro.core import build_binary as _build
        binary = _build(args.build_dir, build_jobs, skip=False, build_type="Release")
    binary = binary.resolve()

    # Load base config
    if not VK_CONFIG.exists():
        print(f"[error] von-karman config not found: {VK_CONFIG}", file=sys.stderr)
        return 1
    with open(VK_CONFIG) as fh:
        base_config = json.load(fh)

    nx = int(base_config.get("nx", 160))
    ny = int(base_config.get("ny", 80))
    nt = args.nt if args.nt is not None else int(base_config.get("nt", 1000))
    dt = float(base_config.get("dt", 0.01))
    sampling_rate = max(1, nt // args.samples)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    flip_coefs = [float(c.strip()) for c in args.flip_coef.split(",") if c.strip()]

    # Build run list
    run_specs = []
    for method in methods:
        coef_list = flip_coefs if method == "flip" else [None]
        for coef in coef_list:
            name = _run_name(method, args.ppc, coef, threads)
            run_specs.append((name, method, coef))

    # Load combined CSV to know what is already done
    combined_csv = out_dir / "wake_point.csv"
    combined_rows = read_csv(combined_csv)
    completed = {r["run"] for r in combined_rows}

    runs_dir = out_dir / "runs"

    for name, method, coef in run_specs:
        run_dir = runs_dir / name
        raw_dir = run_dir / "raw"

        if not args.force and name in completed:
            if not args.keep_raw or raw_dir.exists():
                print(f"[skip] {name}")
                continue
            print(f"[rerun] {name} (raw output missing)")

        config = _build_config(
            base_config, method, args.ppc, coef,
            raw_dir, nt, sampling_rate, {},
        )

        status, _ = _run_sim(binary, name, config, run_dir, threads, args.dry_run)

        if args.dry_run or status != "ok":
            continue

        if np is None:
            print(f"[warn] numpy not available — raw output kept in {raw_dir}")
            continue

        # Extract wake-point data
        pvd_path = raw_dir / "normVelocity.pvd"
        rows = _extract_wake_point(pvd_path, nx, ny, dt, sampling_rate)

        if not rows:
            print(f"[warn] no data extracted for {name}")
            continue

        # Per-method CSV (compatible with plot_point_study: time, normVelocity)
        per_csv = per_method_dir / f"normVel-{name}.csv"
        write_csv(per_csv, [{"time": r["time"], "normVelocity": r["normVelocity"]} for r in rows])

        # Combined CSV
        combined_rows = drop_run(combined_rows, name)
        for r in rows:
            combined_rows.append({
                "run": name,
                "method": method,
                "ppc": args.ppc,
                "flip_coef": coef if coef is not None else "",
                **r,
            })
        write_csv(combined_csv, combined_rows)

        print(f"[extract] {name}: {len(rows)} frames → {per_csv.name}")

        if not args.keep_raw and raw_dir.exists():
            shutil.rmtree(raw_dir)

    print(f"[done] results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
