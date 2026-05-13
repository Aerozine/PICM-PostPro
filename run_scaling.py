#!/usr/bin/env python3
from typing import Dict, List, Tuple
"""Run strong and weak OpenMP scaling studies for PIC. Writes CSVs only."""

import argparse
import copy
import json
import math
import os
import shutil
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

DEFAULT_CONFIG = PICM_ROOT / "test" / "PIC" / "extra" / "freeFallInWater.json"


def _run_name(study: str, binding: str, threads: int, repeat: int) -> str:
    return f"{study}_pic_{binding}_t{threads}_r{repeat}"


def _weak_grid(nx0: int, ny0: int, base_threads: int, threads: int) -> Tuple[int, int]:
    """Scale grid by sqrt(threads/base_threads), rounded to nearest even."""
    scale = math.sqrt(threads / base_threads)
    nx = max(2, round(nx0 * scale))
    ny = max(2, round(ny0 * scale))
    nx += nx % 2
    ny += ny % 2
    return nx, ny


def run_one(
    binary: Path,
    name: str,
    config: dict,
    run_dir: Path,
    threads: int,
    binding: str,
    dry_run: bool = False,
    allow_oversubscribe: bool = False,
) -> Tuple[str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / f"{name}.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    if dry_run:
        print(f"[dry-run] {name} threads={threads} binding={binding}")
        return "dry-run", 0.0

    env = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_PROC_BIND": binding,
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "false",
    }
    if allow_oversubscribe:
        env["OMP_WAIT_POLICY"] = "passive"

    cmd = [str(binary), str(config_path)]
    print(f"[run] {name} threads={threads} binding={binding}")
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


def _compute_scaling(summary_rows: List[dict]) -> List[dict]:
    """Compute speedup and efficiency from summary rows."""
    from collections import defaultdict
    # Group by (study, binding)
    groups: Dict[tuple, list] = defaultdict(list)
    for r in summary_rows:
        if r.get("status") != "ok":
            continue
        key = (r.get("study", ""), r.get("binding", ""))
        try:
            t = float(r["wall_time_s"])
            n = int(r["threads"])
            groups[key].append((n, t))
        except (KeyError, ValueError):
            pass

    rows = []
    for (study, binding), data in groups.items():
        data.sort(key=lambda x: x[0])
        if not data:
            continue
        # baseline = median wall time at min threads
        min_t = min(n for n, _ in data)
        baseline_times = [t for n, t in data if n == min_t]
        if not baseline_times:
            continue
        baseline = sum(baseline_times) / len(baseline_times)

        # Aggregate by thread count
        from collections import defaultdict as dd
        by_threads: Dict[int, List[float]] = dd(list)
        for n, t in data:
            by_threads[n].append(t)

        for n in sorted(by_threads):
            times = by_threads[n]
            mean_t = sum(times) / len(times)
            speedup = baseline / mean_t if mean_t > 0 else float("nan")
            efficiency = speedup / n if n > 0 else float("nan")
            rows.append({
                "study": study,
                "binding": binding,
                "threads": n,
                "runs": len(times),
                "wall_time_s": f"{mean_t:.3f}",
                "speedup": f"{speedup:.4f}",
                "efficiency": f"{efficiency:.4f}",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", default="1,2,4,8,16,32,64",
                        help="comma-separated thread counts")
    parser.add_argument("--bindings", default="close,spread",
                        help="comma-separated OMP_PROC_BIND values")
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-release")
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-oversubscribe", action="store_true")
    args = parser.parse_args()

    build_jobs = args.build_jobs if args.build_jobs is not None else scheduler_threads()
    out_dir = args.out if args.out is not None else DATA_DIR / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve binary
    binary = args.binary
    if binary is None or not binary.exists():
        print("[build] binary not found, building...")
        binary = build_binary(args.build_dir, build_jobs, skip=False, build_type="Release")
    binary = binary.resolve()

    # Load base config
    config_path = args.config if args.config is not None else DEFAULT_CONFIG
    if not config_path.exists():
        print(f"[error] config not found: {config_path}", file=sys.stderr)
        return 1
    with open(config_path) as fh:
        base_config = json.load(fh)

    max_threads = scheduler_threads()
    all_threads = [
        int(t.strip()) for t in args.threads.split(",")
        if t.strip() and int(t.strip()) <= max_threads
    ]
    if not all_threads:
        all_threads = [1]
    bindings = [b.strip() for b in args.bindings.split(",") if b.strip()]

    base_nx = args.nx if args.nx is not None else int(base_config.get("nx", 100))
    base_ny = args.ny if args.ny is not None else int(base_config.get("ny", 100))
    nt = args.nt if args.nt is not None else int(base_config.get("nt", 200))
    base_threads = min(all_threads) if all_threads else 1

    summary_csv = out_dir / "summary.csv"
    scaling_csv = out_dir / "scaling.csv"
    summary_rows = read_csv(summary_csv)

    completed = {
        r["run"] for r in summary_rows if r.get("status") == "ok"
    }

    runs_dir = out_dir / "runs"

    for study in ("strong", "weak"):
        for binding in bindings:
            for threads in all_threads:
                for repeat in range(1, args.repeats + 1):
                    name = _run_name(study, binding, threads, repeat)

                    if not args.force and name in completed:
                        print(f"[skip] {name}")
                        continue

                    run_dir = runs_dir / name
                    raw_dir = run_dir / "raw"

                    if study == "strong":
                        nx, ny = base_nx, base_ny
                    else:
                        nx, ny = _weak_grid(base_nx, base_ny, base_threads, threads)

                    config = copy.deepcopy(base_config)
                    config["method"] = "pic"
                    config["ppcx"] = args.ppc
                    config["ppcy"] = args.ppc
                    config["nx"] = nx
                    config["ny"] = ny
                    config["nt"] = nt
                    config["folder"] = str(raw_dir)
                    config["filename"] = "simulation"
                    config["sampling_rate"] = nt  # no output during run
                    config["write_norm_velocity"] = False
                    config["write_u"] = False
                    config["write_v"] = False
                    config["write_p"] = False
                    config["write_div"] = False
                    config["write_smoke"] = False
                    config["write_particles"] = False
                    config["write_vorticity"] = False

                    status, wall_time = run_one(
                        binary, name, config, run_dir,
                        threads, binding, args.dry_run, args.allow_oversubscribe
                    )

                    if args.dry_run:
                        continue

                    summary_rows = drop_run(summary_rows, name)
                    summary_rows.append({
                        "run": name,
                        "study": study,
                        "binding": binding,
                        "threads": threads,
                        "nx": nx,
                        "ny": ny,
                        "nt": nt,
                        "ppc": args.ppc,
                        "status": status,
                        "wall_time_s": f"{wall_time:.3f}",
                    })
                    write_csv(summary_csv, summary_rows)

                    # Cleanup raw
                    if status == "ok" and raw_dir.exists():
                        shutil.rmtree(raw_dir)

    # Compute scaling metrics
    if not args.dry_run:
        scaling_rows = _compute_scaling(summary_rows)
        write_csv(scaling_csv, scaling_rows)

    print(f"[done] results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
