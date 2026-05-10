#!/usr/bin/env python3
"""Run PIC-only weak and strong scaling studies."""

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, NamedTuple, Tuple

import report_compare as rc
from picm_postpro.paths import DATA_DIR, PICM_ROOT, default_img_dir, default_misc_dir
from picm_postpro.plots import parse_formats, save_figure, style_ax, style_legend

DEFAULT_OUT = DATA_DIR / "study_pic_scaling"
DEFAULT_CONFIG = PICM_ROOT / "test" / "PIC" / "extra" / "freeFallInWater.json"


class ScalingSpec(NamedTuple):
    study: str
    binding: str
    threads: int
    repeat: int
    nx: int
    ny: int
    nt: int
    ppc: int
    name: str
    config_path: Path
    run_dir: Path
    raw_dir: Path

    @property
    def cells(self) -> int:
        return self.nx * self.ny


def parse_int_list(value, default: Iterable[int]) -> Tuple[int, ...]:
    if not value:
        return tuple(default)
    parsed = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if chunk:
            item = int(chunk)
            if item < 1:
                raise ValueError("thread counts must be positive")
            parsed.append(item)
    if not parsed:
        raise ValueError("empty integer list")
    return tuple(dict.fromkeys(parsed))


def scheduler_threads() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def rounded_even(value: float) -> int:
    rounded = max(8, int(round(value)))
    return rounded if rounded % 2 == 0 else rounded + 1


def make_config(base_cfg: dict, spec: ScalingSpec) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["method"] = "pic"
    cfg["nx"] = spec.nx
    cfg["ny"] = spec.ny
    cfg["nt"] = spec.nt
    cfg["ppcx"] = spec.ppc
    cfg["ppcy"] = spec.ppc
    cfg["folder"] = str(spec.raw_dir)
    cfg["filename"] = "simulation"
    cfg["sampling_rate"] = max(1, spec.nt)
    cfg["write_u"] = False
    cfg["write_v"] = False
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_norm_velocity"] = False
    cfg["write_smoke"] = False
    cfg["write_particles"] = False
    return cfg


def make_specs(args: argparse.Namespace, base_cfg: dict) -> List[ScalingSpec]:
    threads = parse_int_list(args.threads, (1, 2, 4, 8, 16, 32, 64))
    bindings = tuple(chunk.strip() for chunk in args.bindings.split(",") if chunk.strip())
    if not bindings:
        raise ValueError("empty binding list")
    max_sched = scheduler_threads()
    threads = tuple(t for t in threads if t <= max_sched or args.allow_oversubscribe)
    if not threads:
        threads = (max_sched,)

    base_threads = min(threads)
    base_nx = args.nx or int(base_cfg.get("nx", 160))
    base_ny = args.ny or int(base_cfg.get("ny", 80))
    nt = args.nt or int(base_cfg.get("nt", 1000))
    out_root = args.out.resolve()
    misc_root = args.misc_dir.resolve()
    specs = []

    for study in ("strong", "weak"):
        for binding in bindings:
            for repeat in range(args.repeats):
                for thread_count in threads:
                    if study == "strong":
                        nx, ny = base_nx, base_ny
                    else:
                        scale = math.sqrt(thread_count / base_threads)
                        nx = rounded_even(base_nx * scale)
                        ny = rounded_even(base_ny * scale)
                    name = f"{study}_pic_{binding}_t{thread_count}_r{repeat}"
                    specs.append(
                        ScalingSpec(
                            study=study,
                            binding=binding,
                            threads=thread_count,
                            repeat=repeat,
                            nx=nx,
                            ny=ny,
                            nt=nt,
                            ppc=args.ppc,
                            name=name,
                            config_path=misc_root / "configs" / f"{name}.json",
                            run_dir=misc_root / "runs" / name,
                            raw_dir=misc_root / "runs" / name / "raw",
                        )
                    )
    return specs


def base_row(spec: ScalingSpec) -> dict:
    return {
        "study": spec.study,
        "binding": spec.binding,
        "method": "pic",
        "threads": spec.threads,
        "repeat": spec.repeat,
        "nx": spec.nx,
        "ny": spec.ny,
        "cells": spec.cells,
        "nt": spec.nt,
        "ppc": spec.ppc,
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


def write_plan_and_configs(specs: List[ScalingSpec], base_cfg: dict, out_root: Path) -> None:
    rows = []
    for spec in specs:
        spec.config_path.parent.mkdir(parents=True, exist_ok=True)
        with spec.config_path.open("w") as handle:
            json.dump(make_config(base_cfg, spec), handle, indent=2)
            handle.write("\n")
        rows.append(rc.merge_row(base_row(spec), {"config": str(spec.config_path)}))
    write_csv(out_root / "plan.csv", rows)


def run_one(
    spec: ScalingSpec,
    binary: Path,
    keep_raw: bool,
    no_run_logs: bool,
    use_srun: bool,
) -> dict:
    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(spec.threads)
    env["OMP_PROC_BIND"] = spec.binding
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_DYNAMIC", "false")
    command = [str(binary), str(spec.config_path)]
    if use_srun and os.environ.get("SLURM_JOB_ID"):
        command = [
            "srun",
            "--ntasks=1",
            "--cpus-per-task",
            os.environ.get("SLURM_CPUS_PER_TASK", str(spec.threads)),
            "--cpu-bind=cores",
            *command,
        ]
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
    if result.returncode != 0:
        print(f"[fail] {spec.name} exited with {result.returncode}")
    return rc.merge_row(base_row(spec), {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "wall_time_s": wall_time,
        "reported_time_s": rc.parse_done_time(combined) or "",
        "config": str(spec.config_path),
    })


def save_scaling(summary_rows: List[dict], out_root: Path) -> None:
    rows = []
    for study in ("strong", "weak"):
        grouped = {}
        for row in summary_rows:
            if row.get("study") == study and row.get("status") == "ok":
                grouped.setdefault((row["binding"], int(row["threads"])), []).append(row)
        if not grouped:
            continue
        bindings = sorted({binding for binding, _threads in grouped})
        for binding in bindings:
            binding_threads = sorted(threads for group_binding, threads in grouped if group_binding == binding)
            base_threads = min(binding_threads)
            base_group = grouped[(binding, base_threads)]
            base_time = sum(float(r["wall_time_s"]) for r in base_group) / len(base_group)
            for threads in binding_threads:
                group = grouped[(binding, threads)]
                wall = sum(float(r["wall_time_s"]) for r in group) / len(group)
                speedup = base_time / wall if wall > 0 else float("nan")
                rows.append(
                    {
                        "study": study,
                        "binding": binding,
                        "threads": threads,
                        "runs": len(group),
                        "nx": group[0]["nx"],
                        "ny": group[0]["ny"],
                        "cells": group[0]["cells"],
                        "nt": group[0]["nt"],
                        "ppc": group[0]["ppc"],
                        "wall_time_s": wall,
                        "speedup": speedup,
                        "efficiency": speedup / (threads / base_threads)
                        if study == "strong"
                        else speedup,
                    }
                )
    write_csv(out_root / "scaling.csv", rows)


def postprocess_csv(out_root: Path) -> None:
    save_scaling(read_csv(out_root / "summary.csv"), out_root)


def plot_scaling(
    out_root: Path,
    img_root: Path,
    image_formats: Iterable[str] = ("png", "svg", "pdf", "jpg"),
) -> None:
    if rc.plt is None or rc.np is None:
        return
    rows = read_csv(out_root / "scaling.csv")
    if not rows:
        return
    img_root.mkdir(parents=True, exist_ok=True)
    for study in ("strong", "weak"):
        fig, ax = rc.plt.subplots()
        any_group = False
        for binding in sorted({row["binding"] for row in rows if row.get("study") == study}):
            group = [row for row in rows if row.get("study") == study and row.get("binding") == binding]
            if not group:
                continue
            any_group = True
            group.sort(key=lambda row: int(row["threads"]))
            threads = [int(row["threads"]) for row in group]
            wall_time = [float(row["wall_time_s"]) for row in group]
            ax.plot(threads, wall_time, "o-", label=binding)
        if not any_group:
            rc.plt.close(fig)
            continue
        style_ax(ax, xlabel="OpenMP threads", ylabel="Wall time [s]",
                 title=f"PIC {study} scaling")
        style_legend(ax)
        fig.tight_layout()
        save_figure(fig, img_root / study, formats=image_formats)
        rc.plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--threads", default=os.environ.get("THREADS", "1,2,4,8,16,32,64"))
    parser.add_argument("--bindings", default=os.environ.get("PICM_SCALING_BINDINGS", "close,spread"))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "1")))
    parser.add_argument("--ppc", type=int, default=int(os.environ.get("PICM_SCALING_PPC", "3")))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--misc-dir", type=Path)
    parser.add_argument("--img-dir", type=Path)
    parser.add_argument("--image-formats", default="png,pdf")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-report-release")
    parser.add_argument("--build-jobs", type=int, default=int(os.environ.get("BUILD_JOBS", str(scheduler_threads()))))
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-run-logs", action="store_true")
    parser.add_argument("--use-srun", action="store_true")
    parser.add_argument("--allow-oversubscribe", action="store_true")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nt", type=int)
    args = parser.parse_args()

    args.out = args.out.resolve()
    args.misc_dir = (args.misc_dir or default_misc_dir(args.out)).resolve()
    args.img_dir = (args.img_dir or default_img_dir(args.out)).resolve()
    args.image_formats = parse_formats(args.image_formats)
    args.build_dir = args.build_dir.resolve()

    if args.plot_only:
        postprocess_csv(args.out)
        if not args.no_plots:
            plot_scaling(args.out, args.img_dir, args.image_formats)
        return 0

    base_cfg = rc.load_config(args.config.resolve())
    specs = make_specs(args, base_cfg)
    write_plan_and_configs(specs, base_cfg, args.out)
    print(f"[plan] wrote {args.out / 'plan.csv'}")
    for spec in specs:
        print(
            f"  {spec.study:6s} PIC threads={spec.threads:<3d} "
            f"binding={spec.binding:6s} grid={spec.nx}x{spec.ny:<4d} "
            f"nt={spec.nt:<4d} ppc={spec.ppc}"
        )
    if args.dry_run:
        print("[dry-run] configs generated; no build and no simulation launched")
        return 0

    binary = rc.build_binary(args.skip_build, args.build_jobs, args.build_dir)
    summary_rows = [] if args.force else read_csv(args.out / "summary.csv")
    completed = {row["run"] for row in summary_rows if row.get("status") == "ok"}
    failures = 0
    for spec in specs:
        if spec.name in completed:
            print(f"[skip] {spec.name}: already present in summary.csv")
            continue
        summary_rows = drop_run(summary_rows, spec.name)
        summary = run_one(spec, binary, args.keep_raw, args.no_run_logs, args.use_srun)
        summary_rows.append(summary)
        if summary["status"] != "ok":
            failures += 1
        write_csv(args.out / "summary.csv", summary_rows)
        postprocess_csv(args.out)
        print(f"[checkpoint] CSV files updated after {spec.name}")
    write_csv(args.out / "summary.csv", summary_rows)
    postprocess_csv(args.out)
    if not args.no_plots:
        plot_scaling(args.out, args.img_dir, args.image_formats)
    print(f"[done] CSV files: {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
