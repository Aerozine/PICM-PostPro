#!/usr/bin/env python3
"""Compare pressure-solver iteration counts on a CPU debug build."""

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, NamedTuple, Tuple

import report_compare as rc
from picm_postpro.paths import DATA_DIR, PICM_ROOT, default_img_dir, default_misc_dir
from picm_postpro.plots import parse_formats, save_figure

DEFAULT_CONFIG = PICM_ROOT / "test" / "PIC" / "dambreak.json"
DEFAULT_OUT = DATA_DIR / "study_iterative_solvers"
CPU_SOLVERS = ("jacobi", "gauss_seidel", "red_black_gauss_seidel", "miccg0", "cg")

ITER_RE = re.compile(
    r"\[DEBUG\]\s+(\S+).*?(?:converged in\s+(\d+)\s+iters|"
    r"reached maxIters\s*=\s*(\d+))"
)


class SolverSpec(NamedTuple):
    solver: str
    tolerance: float
    repeat: int
    threads: int
    name: str
    config_path: Path
    run_dir: Path
    raw_dir: Path


def parse_csv(value: str) -> Tuple[str, ...]:
    parsed = tuple(chunk.strip() for chunk in value.split(",") if chunk.strip())
    if not parsed:
        raise ValueError("empty CSV list")
    return parsed


def scheduler_threads() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def build_debug_binary(skip_build: bool, build_jobs: int, build_dir: Path) -> Path:
    binary = build_dir / "bin" / "PIC"
    if skip_build:
        if not binary.exists():
            raise FileNotFoundError(f"missing debug binary: {binary}")
        return binary

    rc.prepare_build_dir(build_dir)
    configure_cmd = [
        "cmake",
        "-S",
        str(PICM_ROOT),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DUSE_GPU=OFF",
        "-DUSE_PARALLEL=ON",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), f"-j{build_jobs}"]
    print(f"[build] configuring debug: {' '.join(configure_cmd)}")
    subprocess.run(configure_cmd, cwd=PICM_ROOT, check=True)
    print(f"[build] building debug: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, cwd=PICM_ROOT, check=True)
    if not binary.exists():
        raise FileNotFoundError(f"build succeeded but binary is missing: {binary}")
    return binary


def make_config(base_cfg: dict, spec: SolverSpec, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["method"] = "pic"
    cfg["folder"] = str(spec.raw_dir)
    cfg["filename"] = "simulation"
    if args.nt is not None:
        cfg["nt"] = args.nt
    if args.ppc is not None:
        cfg["ppcx"] = args.ppc
        cfg["ppcy"] = args.ppc
    cfg["sampling_rate"] = max(1, int(cfg.get("nt", 1)))
    cfg["write_u"] = False
    cfg["write_v"] = False
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_norm_velocity"] = False
    cfg["write_smoke"] = False
    cfg["write_particles"] = False
    solver = dict(cfg.get("solver", {}))
    solver["type"] = spec.solver
    solver["tolerance"] = spec.tolerance
    if args.max_iterations is not None:
        solver["max_iterations"] = args.max_iterations
    cfg["solver"] = solver
    return cfg


def make_specs(args: argparse.Namespace) -> List[SolverSpec]:
    solvers = parse_csv(args.solvers)
    unknown = [solver for solver in solvers if solver not in CPU_SOLVERS]
    if unknown:
        raise ValueError(f"unknown CPU solver(s): {unknown}; expected {CPU_SOLVERS}")
    tolerances = tuple(float(value) for value in parse_csv(args.tolerances))
    out_root = args.out.resolve()
    misc_root = args.misc_dir.resolve()
    specs = []
    for solver in solvers:
        for tolerance in tolerances:
            tol_slug = rc.slug_float(tolerance)
            for repeat in range(args.repeats):
                name = f"dambreak_{solver}_tol{tol_slug}_t{args.threads}_r{repeat}"
                specs.append(
                    SolverSpec(
                        solver=solver,
                        tolerance=tolerance,
                        repeat=repeat,
                        threads=args.threads,
                        name=name,
                        config_path=misc_root / "configs" / f"{name}.json",
                        run_dir=misc_root / "runs" / name,
                        raw_dir=misc_root / "runs" / name / "raw",
                    )
                )
    return specs


def base_row(spec: SolverSpec) -> dict:
    return {
        "test": "dambreak",
        "method": "pic",
        "solver": spec.solver,
        "tolerance": spec.tolerance,
        "threads": spec.threads,
        "repeat": spec.repeat,
        "run": spec.name,
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if not rows:
        tmp_path.write_text("")
        tmp_path.replace(path)
        return
    columns = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def read_csv(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def drop_run(rows: List[dict], run: str) -> List[dict]:
    return [row for row in rows if row.get("run") != run]


def write_plan_and_configs(specs: List[SolverSpec], base_cfg: dict, args: argparse.Namespace) -> None:
    rows = []
    for spec in specs:
        spec.config_path.parent.mkdir(parents=True, exist_ok=True)
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        with spec.config_path.open("w") as handle:
            json.dump(make_config(base_cfg, spec, args), handle, indent=2)
            handle.write("\n")
        rows.append(rc.merge_row(base_row(spec), {"config": str(spec.config_path)}))
    write_csv(args.out.resolve() / "plan.csv", rows)


def parse_iterations(spec: SolverSpec, text: str, cfg: dict) -> List[dict]:
    dt = float(cfg.get("dt", 1.0))
    rows = []
    for index, match in enumerate(ITER_RE.finditer(text), start=1):
        rows.append(
            rc.merge_row(
                base_row(spec),
                {
                "solve": index,
                "step": index,
                "time": index * dt,
                "debug_solver_name": match.group(1).rstrip(":"),
                "pressure_iters": int(match.group(2) or match.group(3)),
                "hit_max_iterations": bool(match.group(3)),
                },
            )
        )
    return rows


def run_one(
    spec: SolverSpec,
    binary: Path,
    keep_raw: bool,
    no_run_logs: bool,
) -> Tuple[dict, List[dict]]:
    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(spec.threads)
    env.setdefault("OMP_PROC_BIND", "spread")
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_DYNAMIC", "false")
    command = [str(binary), str(spec.config_path)]
    print(f"[run] {spec.name}: OMP_NUM_THREADS={spec.threads} {' '.join(command)}")
    start = time.perf_counter()
    result = rc.run_process(command, cwd=PICM_ROOT, env=env)
    wall_time = time.perf_counter() - start
    if not no_run_logs or result.returncode != 0:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        (spec.run_dir / "stdout.log").write_text(result.stdout)
        (spec.run_dir / "stderr.log").write_text(result.stderr)
    if not keep_raw and spec.raw_dir.exists():
        shutil.rmtree(spec.raw_dir)
    if no_run_logs and spec.run_dir.exists():
        try:
            spec.run_dir.rmdir()
        except OSError:
            pass
    if no_run_logs:
        try:
            spec.run_dir.parent.rmdir()
        except OSError:
            pass

    combined = result.stdout + "\n" + result.stderr
    cfg = rc.load_config(spec.config_path)
    iter_rows = parse_iterations(spec, combined, cfg)
    counts = [int(row["pressure_iters"]) for row in iter_rows]
    summary = rc.merge_row(base_row(spec), {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "wall_time_s": wall_time,
        "reported_time_s": rc.parse_done_time(combined) or "",
        "pressure_solves": len(counts),
        "mean_pressure_iters": sum(counts) / len(counts) if counts else "",
        "max_pressure_iters": max(counts) if counts else "",
        "min_pressure_iters": min(counts) if counts else "",
        "max_iteration_hits": sum(1 for row in iter_rows if row["hit_max_iterations"]),
        "config": str(spec.config_path),
    })
    if result.returncode != 0:
        print(f"[fail] {spec.name} exited with {result.returncode}")
    if not counts:
        print(f"[warn] {spec.name}: no debug iteration lines found; use a Debug build")
    return summary, iter_rows


def postprocess_csv(out_root: Path) -> None:
    return None


def solver_sort_key(row: dict) -> int:
    try:
        return CPU_SOLVERS.index(row["solver"])
    except ValueError:
        return len(CPU_SOLVERS)


def optional_float(value):
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if rc.math.isfinite(parsed) else None


def plot_iteration_time_series(
    iter_rows: List[dict],
    tolerance: str,
    plot_dir: Path,
    image_formats: Tuple[str, ...],
) -> None:
    tolerance_rows = [row for row in iter_rows if str(row.get("tolerance")) == str(tolerance)]
    if not tolerance_rows:
        return

    grouped = {}
    has_time = any(row.get("time") not in ("", None) for row in tolerance_rows)
    x_key = "time" if has_time else "solve"
    for row in tolerance_rows:
        x_value = optional_float(row.get(x_key))
        y_value = optional_float(row.get("pressure_iters"))
        if x_value is None or y_value is None:
            continue
        grouped.setdefault(row["solver"], {}).setdefault(x_value, []).append(y_value)
    if not grouped:
        return

    fig, ax = rc.plt.subplots(figsize=(10, 5.5))
    for solver in sorted(grouped, key=lambda name: CPU_SOLVERS.index(name) if name in CPU_SOLVERS else 99):
        time_map = grouped[solver]
        xs = sorted(time_map)
        ys = [rc.mean(time_map[x]) for x in xs]
        ax.plot(xs, ys, lw=1.5, label=solver)
    y_values = [value for time_map in grouped.values() for values in time_map.values() for value in values]
    if y_values and max(y_values) / max(min(value for value in y_values if value > 0), 1.0) > 50:
        ax.set_yscale("log")
    ax.set_xlabel("Time t [s]" if has_time else "Pressure solve index")
    ax.set_ylabel("Iterations per pressure solve")
    ax.set_title(f"Dambreak: pressure iterations over time, tolerance={tolerance}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(
        fig,
        plot_dir / f"dambreak_solver_iter_time_tol{rc.slug_float(float(tolerance))}",
        formats=image_formats,
    )
    rc.plt.close(fig)


def plot_iterations(
    out_root: Path,
    img_root: Path,
    image_formats: Tuple[str, ...] = ("png", "svg", "pdf", "jpg"),
) -> None:
    if rc.plt is None:
        return
    summary_rows = [row for row in read_csv(out_root / "summary.csv") if row.get("status") == "ok"]
    if not summary_rows:
        return
    iter_rows = read_csv(out_root / "iterations.csv")
    plot_dir = img_root
    plot_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in summary_rows:
        grouped.setdefault(str(row["tolerance"]), []).append(row)
    for tolerance, rows in grouped.items():
        rows.sort(key=solver_sort_key)
        labels = [row["solver"] for row in rows]
        values = [float(row["mean_pressure_iters"]) for row in rows]
        fig, ax = rc.plt.subplots(figsize=(9, 5))
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_ylabel("Mean iterations per pressure solve")
        ax.set_title(f"Dambreak: mean pressure iterations, tolerance={tolerance}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save_figure(
            fig,
            plot_dir / f"dambreak_solver_iters_tol{rc.slug_float(float(tolerance))}",
            formats=image_formats,
        )
        rc.plt.close(fig)
        plot_iteration_time_series(iter_rows, tolerance, plot_dir, image_formats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--solvers", default=",".join(CPU_SOLVERS))
    parser.add_argument("--tolerances", default=os.environ.get("PICM_SOLVER_TOLERANCES", "1e-2"))
    parser.add_argument("--threads", type=int, default=scheduler_threads())
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "1")))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--misc-dir", type=Path)
    parser.add_argument("--img-dir", type=Path)
    parser.add_argument("--image-formats", default="png,svg,pdf,jpg")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-solver-debug")
    parser.add_argument("--build-jobs", type=int, default=int(os.environ.get("BUILD_JOBS", str(scheduler_threads()))))
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-run-logs", action="store_true")
    parser.add_argument("--nt", type=int)
    parser.add_argument("--ppc", type=int)
    parser.add_argument("--max-iterations", type=int)
    args = parser.parse_args()

    args.out = args.out.resolve()
    args.misc_dir = (args.misc_dir or default_misc_dir(args.out)).resolve()
    args.img_dir = (args.img_dir or default_img_dir(args.out)).resolve()
    args.image_formats = parse_formats(args.image_formats)
    args.build_dir = args.build_dir.resolve()
    if args.plot_only:
        postprocess_csv(args.out)
        if not args.no_plots:
            plot_iterations(args.out, args.img_dir, args.image_formats)
        return 0

    base_cfg = rc.load_config(args.config.resolve())
    specs = make_specs(args)
    write_plan_and_configs(specs, base_cfg, args)
    print(f"[plan] wrote {args.out / 'plan.csv'}")
    for spec in specs:
        print(
            f"  dambreak solver={spec.solver:24s} tol={spec.tolerance:g} "
            f"threads={spec.threads} repeat={spec.repeat}"
        )
    if args.dry_run:
        print("[dry-run] configs generated; no build and no simulation launched")
        return 0

    binary = build_debug_binary(args.skip_build, args.build_jobs, args.build_dir)
    summary_rows = [] if args.force else read_csv(args.out / "summary.csv")
    iter_rows = [] if args.force else read_csv(args.out / "iterations.csv")
    completed = {row["run"] for row in summary_rows if row.get("status") == "ok"}
    failures = 0
    for spec in specs:
        if spec.name in completed:
            print(f"[skip] {spec.name}: already present in summary.csv")
            continue
        summary_rows = drop_run(summary_rows, spec.name)
        iter_rows = drop_run(iter_rows, spec.name)
        summary, rows = run_one(spec, binary, args.keep_raw, args.no_run_logs)
        summary_rows.append(summary)
        iter_rows.extend(rows)
        if summary["status"] != "ok":
            failures += 1
        write_csv(args.out / "summary.csv", summary_rows)
        write_csv(args.out / "iterations.csv", iter_rows)
        print(f"[checkpoint] CSV files updated after {spec.name}")
    write_csv(args.out / "summary.csv", summary_rows)
    write_csv(args.out / "iterations.csv", iter_rows)
    if not args.no_plots:
        plot_iterations(args.out, args.img_dir, args.image_formats)
    print(f"[done] CSV files: {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
