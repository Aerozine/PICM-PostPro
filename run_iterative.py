#!/usr/bin/env python3
from typing import List
"""Compare pressure solver iteration counts using a debug build. Writes CSVs only."""

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

from picm_postpro.paths import DATA_DIR, PICM_ROOT
from picm_postpro.core import (
    build_binary,
    drop_run,
    read_csv,
    run_binary,
    scheduler_threads,
    write_csv,
)

DEFAULT_CONFIG = PICM_ROOT / "test" / "PIC" / "extra" / "dambreak.json"

ITER_RE = re.compile(
    r"\[DEBUG\]\s+(\S+).*?(?:converged in\s+(\d+)\s+iters|reached maxIters\s*=\s*(\d+))"
)

DEFAULT_SOLVERS = [
    "jacobi",
    "gauss_seidel",
    "red_black_gauss_seidel",
    "miccg0",
    "cg",
]


def _run_name(solver: str, tolerance: float, threads: int, repeat: int) -> str:
    tol_str = f"{tolerance:g}".replace(".", "p").replace("-", "n")
    return f"iterative_{solver}_tol{tol_str}_t{threads}_r{repeat}"


def _parse_iterations(stderr_bytes: bytes, solver_name: str) -> List[dict]:
    """Parse [DEBUG] lines from stderr to extract iteration counts per step."""
    text = stderr_bytes.decode("utf-8", errors="replace")
    rows = []
    step = 0
    for line in text.splitlines():
        m = ITER_RE.search(line)
        if m:
            iters_converged = m.group(2)
            iters_max = m.group(3)
            if iters_converged is not None:
                iters = int(iters_converged)
                hit_max = False
            elif iters_max is not None:
                iters = int(iters_max)
                hit_max = True
            else:
                continue
            rows.append({
                "step": step,
                "iterations": iters,
                "hit_max": hit_max,
            })
            step += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS),
                        help="comma-separated solver names")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--nt", type=int, default=200)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-debug")
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build_jobs = args.build_jobs if args.build_jobs is not None else scheduler_threads()
    out_dir = args.out if args.out is not None else DATA_DIR / "iterative"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve binary (debug build)
    binary = args.binary
    if binary is None or not binary.exists():
        print("[build] debug binary not found, building...")
        binary = build_binary(args.build_dir, build_jobs, skip=False, build_type="Debug")
    binary = binary.resolve()

    # Load base config
    config_path = args.config if args.config is not None else DEFAULT_CONFIG
    if not config_path.exists():
        print(f"[error] config not found: {config_path}", file=sys.stderr)
        return 1
    with open(config_path) as fh:
        base_config = json.load(fh)

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]

    iter_csv = out_dir / "iterations.csv"
    iter_rows = read_csv(iter_csv)

    runs_dir = out_dir / "runs"

    completed = {
        r["run"]
        for r in iter_rows
        if r.get("run")
    }

    for solver in solvers:
        for repeat in range(1, args.repeats + 1):
            name = _run_name(solver, args.tolerance, args.threads, repeat)

            if not args.force and name in completed:
                print(f"[skip] {name}")
                continue

            run_dir = runs_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)

            config = copy.deepcopy(base_config)
            config["method"] = "pic"
            config["ppcx"] = args.ppc
            config["ppcy"] = args.ppc
            config["nt"] = args.nt
            config["sampling_rate"] = args.nt  # no VTI output
            config["folder"] = str(run_dir / "raw")
            config["filename"] = "simulation"
            config["write_norm_velocity"] = False
            config["write_u"] = False
            config["write_v"] = False
            config["write_p"] = False
            config["write_div"] = False
            config["write_smoke"] = False
            config["write_particles"] = False
            config["write_vorticity"] = False

            # Solver settings
            config["solver"] = {
                "type": solver,
                "max_iterations": args.max_iterations,
                "tolerance": args.tolerance,
            }

            config_out = run_dir / f"{name}.json"
            with open(config_out, "w") as fh:
                json.dump(config, fh, indent=2)

            if args.dry_run:
                print(f"[dry-run] {name}")
                continue

            env = {"OMP_NUM_THREADS": str(args.threads)}
            cmd = [str(binary), str(config_out)]
            print(f"[run] {name}")
            t0 = time.perf_counter()
            result = run_binary(cmd, env)
            wall = time.perf_counter() - t0

            (run_dir / "stdout.log").write_bytes(result.stdout)
            (run_dir / "stderr.log").write_bytes(result.stderr)

            if result.returncode != 0:
                print(f"[run] FAILED {name} (exit {result.returncode})")
                continue

            print(f"[run] OK {name} ({wall:.1f}s)")

            # Parse iteration data
            iter_data = _parse_iterations(result.stderr, solver)
            if not iter_data:
                # Try parsing from stderr log
                stderr_text = result.stderr.decode("utf-8", errors="replace")
                print(f"[warn] no [DEBUG] lines found for {name}; "
                      f"make sure the debug build emits them.")

            # Remove old rows for this run
            iter_rows = [r for r in iter_rows if r.get("run") != name]

            for row in iter_data:
                iter_rows.append({
                    "solver": solver,
                    "tolerance": args.tolerance,
                    "run": name,
                    "step": row["step"],
                    "iterations": row["iterations"],
                    "hit_max": row["hit_max"],
                })

            write_csv(iter_csv, iter_rows)

    print(f"[done] results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
