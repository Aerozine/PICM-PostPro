#!/usr/bin/env python3
"""Run report-oriented PIC, FLIP, and APIC comparison studies.

The script generates one derived JSON config per run so that the scene is kept
identical and only the selected method/parameters change. It writes CSV files
first, and plots when matplotlib is available.
"""

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

from picm_postpro.paths import (
    DATA_DIR,
    PICM_ROOT,
    SCRIPTS_DIR,
    default_img_dir,
    default_misc_dir,
)
from picm_postpro.plots import parse_formats, save_figure

os.environ.setdefault("MPLCONFIGDIR", "/tmp/picm_matplotlib")

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None


SCRIPT_DIR = SCRIPTS_DIR
DEFAULT_OUT = DATA_DIR / "report_comparisons"

REPORT_TESTS = {
    "falling-block-water": {
        "config": PICM_ROOT / "test" / "PIC" / "freeFallInWater.json",
        "reason": "falling block in water; useful for vorticity and energy conservation",
    },
    "freeFallInWater": {
        "config": PICM_ROOT / "test" / "PIC" / "freeFallInWater.json",
        "reason": "falling block in water; useful for vorticity and energy conservation",
    },
    "von-karman": {
        "config": PICM_ROOT / "test" / "PIC" / "von-karman.json",
        "reason": "wake dynamics behind an obstacle; exposes numerical diffusion",
    },
    "dambreak": {
        "config": PICM_ROOT / "test" / "PIC" / "dambreak.json",
        "reason": "free-surface collapse; exposes interface stability",
    },
    "vases-communicants": {
        "config": PICM_ROOT
        / "test"
        / "PIC"
        / "vases-communicants"
        / "vases-communicants.json",
        "reason": "hydrostatic balancing; exposes damping and long-time stability",
    },
}

REPORT_METHODS = ("pic", "flip", "apic")

MAX_DIV_RE = re.compile(
    r"Step\s+(\d+)\s*/\s*(\d+).*?max\s+\|div\|\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r".*?reached at\s+\((-?\d+),(-?\d+)\)"
)
DONE_RE = re.compile(
    r"Done:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*s"
)


class RunSpec(NamedTuple):
    test: str
    method: str
    ppc: int
    flip_coef_pic: Optional[float]
    threads: int
    repeat: int
    name: str
    config_path: Path
    run_dir: Path
    raw_dir: Path


def merge_row(row, extra):
    merged = dict(row)
    merged.update(extra)
    return merged


def run_process(command, cwd, env=None):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def parse_int_list(value: Optional[str], default: Iterable[int]) -> Tuple[int, ...]:
    if not value:
        return tuple(default)
    parsed = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        item = int(chunk)
        if item < 1:
            raise ValueError("integer list values must be positive")
        parsed.append(item)
    if not parsed:
        raise ValueError("empty integer list")
    return tuple(dict.fromkeys(parsed))


def parse_float_list(value: Optional[str], default: Iterable[float]) -> Tuple[float, ...]:
    if not value:
        return tuple(default)
    parsed = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if chunk:
            parsed.append(float(chunk))
    if not parsed:
        raise ValueError("empty float list")
    return tuple(dict.fromkeys(parsed))


def parse_name_list(value: Optional[str], valid: Iterable[str]) -> Tuple[str, ...]:
    valid_set = set(valid)
    if not value or value == "all":
        return tuple(valid)
    parsed = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if item not in valid_set:
            raise ValueError(f"unknown value '{item}', expected one of {sorted(valid_set)}")
        parsed.append(item)
    if not parsed:
        raise ValueError("empty name list")
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


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_name(
    test: str,
    method: str,
    ppc: int,
    flip_coef_pic: Optional[float],
    threads: int,
    repeat: int,
) -> str:
    parts = [test, method, f"ppc{ppc}"]
    if method == "flip":
        parts.append(f"coefpic{slug_float(flip_coef_pic or 0.0)}")
    parts.extend((f"t{threads}", f"r{repeat}"))
    return "_".join(parts)


def prepare_build_dir(build_dir: Path) -> None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return
    source_dir = None
    for line in cache.read_text(errors="ignore").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            source_dir = Path(line.split("=", 1)[1]).expanduser()
            break
    if source_dir is None:
        return
    try:
        same_source = source_dir.resolve() == PICM_ROOT
    except FileNotFoundError:
        same_source = False
    if not same_source:
        print(f"[build] removing stale CMake build directory: {build_dir}")
        shutil.rmtree(build_dir)


def build_binary(skip_build: bool, build_jobs: int, build_dir: Path) -> Path:
    binary = build_dir / "bin" / "PIC"
    if skip_build:
        if not binary.exists():
            raise FileNotFoundError(f"missing release binary: {binary}")
        return binary

    prepare_build_dir(build_dir)
    configure_cmd = [
        "cmake",
        "-S",
        str(PICM_ROOT),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DUSE_GPU=OFF",
        "-DUSE_PARALLEL=ON",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), f"-j{build_jobs}"]
    print(f"[build] configuring: {' '.join(configure_cmd)}")
    subprocess.run(configure_cmd, cwd=PICM_ROOT, check=True)
    print(f"[build] building: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, cwd=PICM_ROOT, check=True)
    if not binary.exists():
        raise FileNotFoundError(f"build succeeded but binary is missing: {binary}")
    return binary


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def disable_heavy_outputs(cfg: dict, *, need_velocity_fields: bool = False) -> None:
    cfg["write_u"] = need_velocity_fields
    cfg["write_v"] = need_velocity_fields
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_smoke"] = False
    cfg["write_norm_velocity"] = True
    cfg["write_particles"] = False


def make_config(spec: RunSpec, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(load_config(REPORT_TESTS[spec.test]["config"]))
    cfg["method"] = spec.method
    cfg["ppcx"] = spec.ppc
    cfg["ppcy"] = spec.ppc
    cfg["folder"] = str(spec.raw_dir)
    cfg["filename"] = "simulation"
    if spec.method == "flip":
        cfg["coefPic"] = float(spec.flip_coef_pic if spec.flip_coef_pic is not None else 0.05)

    if args.nt is not None:
        cfg["nt"] = args.nt
    if args.nx is not None:
        cfg["nx"] = args.nx
    if args.ny is not None:
        cfg["ny"] = args.ny
    if args.dt is not None:
        cfg["dt"] = args.dt
    if args.kernel_order is not None:
        cfg["kernelOrder"] = args.kernel_order

    if args.samples is not None:
        nt = int(cfg.get("nt", 1))
        cfg["sampling_rate"] = max(1, nt // max(1, args.samples))

    solver = dict(cfg.get("solver", {}))
    if args.solver_type is not None:
        solver["type"] = args.solver_type
    if args.solver_tolerance is not None:
        solver["tolerance"] = args.solver_tolerance
    if args.solver_max_iterations is not None:
        solver["max_iterations"] = args.solver_max_iterations
    cfg["solver"] = solver

    disable_heavy_outputs(cfg, need_velocity_fields=args.analysis in ("vorticity", "ppc"))
    return cfg


def make_specs(args: argparse.Namespace) -> List[RunSpec]:
    tests = parse_name_list(args.test, REPORT_TESTS.keys())
    methods = parse_name_list(args.methods, REPORT_METHODS)
    ppcs = parse_int_list(args.ppc, (3,))
    threads = parse_int_list(args.threads, (scheduler_threads(),))
    flip_coefs = parse_float_list(args.flip_coef_pic, (0.05,))
    out_root = args.out.resolve()
    misc_root = args.misc_dir.resolve()

    specs = []
    for test in tests:
        for method in methods:
            method_coefs = None
            if method == "flip":
                method_coefs = flip_coefs
            else:
                method_coefs = (None,)
            for ppc in ppcs:
                for coef in method_coefs:
                    for thread_count in threads:
                        for repeat in range(args.repeats):
                            name = run_name(test, method, ppc, coef, thread_count, repeat)
                            specs.append(
                                RunSpec(
                                    test=test,
                                    method=method,
                                    ppc=ppc,
                                    flip_coef_pic=coef,
                                    threads=thread_count,
                                    repeat=repeat,
                                    name=name,
                                    config_path=misc_root
                                    / "configs"
                                    / test
                                    / f"{name}.json",
                                    run_dir=misc_root / "runs" / test / name,
                                    raw_dir=misc_root / "runs" / test / name / "raw",
                                )
                            )
    return specs


def write_plan_and_configs(specs: List[RunSpec], args: argparse.Namespace) -> None:
    rows = []
    for spec in specs:
        spec.config_path.parent.mkdir(parents=True, exist_ok=True)
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        cfg = make_config(spec, args)
        with spec.config_path.open("w") as handle:
            json.dump(cfg, handle, indent=2)
            handle.write("\n")
        rows.append(
            {
                "test": spec.test,
                "method": spec.method,
                "ppc": spec.ppc,
                "flip_coef_pic": "" if spec.flip_coef_pic is None else spec.flip_coef_pic,
                "threads": spec.threads,
                "repeat": spec.repeat,
                "run": spec.name,
                "config": str(spec.config_path),
                "raw_folder": str(spec.raw_dir),
                "reference_config": str(REPORT_TESTS[spec.test]["config"]),
                "reason": REPORT_TESTS[spec.test]["reason"],
            }
        )
    write_csv(args.out.resolve() / "plan.csv", rows)


def parse_pvd(pvd_path: Path) -> List[Path]:
    tree = ET.parse(pvd_path)
    base = pvd_path.parent
    paths = []
    for dataset in tree.getroot().iter("DataSet"):
        file_attr = dataset.get("file")
        if file_attr:
            paths.append(base / file_attr)
    return paths


def _xml_without_appended_data(raw: bytes) -> bytes:
    start = raw.find(b"<AppendedData")
    if start == -1:
        return raw
    underscore = raw.find(b"_", start)
    if underscore == -1:
        return raw
    return raw[:underscore] + b"\n  </AppendedData>\n</VTKFile>"


def _appended_data_start(raw: bytes) -> int:
    start = raw.find(b"<AppendedData")
    if start == -1:
        raise ValueError("VTI has no AppendedData block")
    underscore = raw.find(b"_", start)
    if underscore == -1:
        raise ValueError("VTI AppendedData block has no '_' marker")
    return underscore + 1


def read_vti_field(vti_path: Path, field_name: str):
    if np is None:
        raise RuntimeError("numpy is required to extract VTI fields")
    raw = vti_path.read_bytes()
    root = ET.fromstring(_xml_without_appended_data(raw))
    compressed = "compressor" in root.attrib

    image = root.find(".//ImageData")
    if image is None:
        raise ValueError(f"{vti_path} has no ImageData node")
    extent = [int(x) for x in image.get("WholeExtent", "0 0 0 0 0 0").split()]
    nx = extent[1] - extent[0]
    ny = extent[3] - extent[2]
    if nx <= 0 or ny <= 0:
        raise ValueError(f"{vti_path} has invalid extent {extent}")

    arrays = list(root.iter("DataArray"))
    data_array = next((da for da in arrays if da.get("Name") == field_name), None)
    if data_array is None:
        names = [da.get("Name", "") for da in arrays]
        raise KeyError(f"{field_name} not found in {vti_path.name}; available={names}")

    dtype_name = data_array.get("type", "Float32")
    if dtype_name == "Float64":
        dtype = np.dtype("<f8")
    elif dtype_name == "Float32":
        dtype = np.dtype("<f4")
    else:
        raise ValueError(f"unsupported VTI dtype {dtype_name} in {vti_path.name}")

    offset = int(data_array.get("offset", 0))
    chunk = raw[_appended_data_start(raw) + offset :]
    if compressed:
        _num_blocks, raw_size, _last_block_size, comp_size = struct.unpack_from(
            "<IIII", chunk, 0
        )
        payload = zlib.decompress(chunk[16 : 16 + comp_size])
        if len(payload) < raw_size:
            raise ValueError(f"decompressed payload is too small in {vti_path.name}")
    else:
        (raw_size,) = struct.unpack_from("<I", chunk, 0)
        payload = chunk[4 : 4 + raw_size]

    arr = np.frombuffer(payload, dtype=dtype)
    expected = nx * ny
    if arr.size < expected:
        raise ValueError(f"{vti_path.name}: expected {expected} values, got {arr.size}")
    return arr[:expected].reshape(ny, nx)


def base_row(spec: RunSpec) -> dict:
    return {
        "test": spec.test,
        "method": spec.method,
        "ppc": spec.ppc,
        "flip_coef_pic": "" if spec.flip_coef_pic is None else spec.flip_coef_pic,
        "threads": spec.threads,
        "repeat": spec.repeat,
        "run": spec.name,
    }


def extract_kinetic_energy(spec: RunSpec, cfg: dict) -> List[dict]:
    pvd_path = spec.raw_dir / "normVelocity.pvd"
    if not pvd_path.exists() or np is None:
        return []

    dx = float(cfg.get("dx", 1.0))
    dy = float(cfg.get("dy", 1.0))
    density = float(cfg.get("density", 1.0))
    dt = float(cfg.get("dt", 1.0))
    sampling_rate = int(cfg.get("sampling_rate", 1))
    paths = parse_pvd(pvd_path)

    values = []
    for sample_index, vti_path in enumerate(paths):
        if not vti_path.exists():
            continue
        speed = read_vti_field(vti_path, "normVelocity")
        kinetic_energy = 0.5 * density * dx * dy * float(np.sum(speed * speed))
        step = sample_index * sampling_rate
        values.append((sample_index, step, step * dt, kinetic_energy))

    reference = next(
        (ke for _sample, step, _time, ke in values if step > 0 and abs(ke) > 1e-30),
        None,
    )
    if reference is None:
        reference = next((ke for _sample, _step, _time, ke in values if abs(ke) > 1e-30), 1.0)

    rows = []
    for sample_index, step, sim_time, kinetic_energy in values:
        rows.append(
            merge_row(
                base_row(spec),
                {
                "sample": sample_index,
                "step": step,
                "time": sim_time,
                "kinetic_energy": kinetic_energy,
                "normalized_ke": kinetic_energy / reference,
                },
            )
        )
    return rows


def compute_vorticity(u_field, v_field, dx: float, dy: float):
    if np is None:
        raise RuntimeError("numpy is required to compute vorticity")
    ny = min(u_field.shape[0], v_field.shape[0] - 1)
    nx = min(u_field.shape[1] - 1, v_field.shape[1])
    if nx <= 1 or ny <= 1:
        raise ValueError(f"invalid u/v shapes for vorticity: u={u_field.shape}, v={v_field.shape}")
    dv_dx = (v_field[1:ny, 1:nx] - v_field[1:ny, 0 : nx - 1]) / dx
    du_dy = (u_field[1:ny, 1:nx] - u_field[0 : ny - 1, 1:nx]) / dy
    return dv_dx - du_dy


def extract_vorticity(spec: RunSpec, cfg: dict) -> Tuple[List[dict], List[Tuple[float, object]]]:
    u_pvd = spec.raw_dir / "u.pvd"
    v_pvd = spec.raw_dir / "v.pvd"
    if not u_pvd.exists() or not v_pvd.exists() or np is None:
        return [], []

    dx = float(cfg.get("dx", 1.0))
    dy = float(cfg.get("dy", 1.0))
    dt = float(cfg.get("dt", 1.0))
    sampling_rate = int(cfg.get("sampling_rate", 1))
    u_paths = parse_pvd(u_pvd)
    v_paths = parse_pvd(v_pvd)
    rows = []
    frames = []
    for sample_index, (u_path, v_path) in enumerate(zip(u_paths, v_paths)):
        if not u_path.exists() or not v_path.exists():
            continue
        u_field = read_vti_field(u_path, "u")
        v_field = read_vti_field(v_path, "v")
        omega = compute_vorticity(u_field, v_field, dx, dy)
        step = sample_index * sampling_rate
        abs_omega = np.abs(omega)
        rows.append(
            merge_row(
                base_row(spec),
                {
                "sample": sample_index,
                "step": step,
                "time": step * dt,
                "mean_abs_vorticity": float(np.mean(abs_omega)),
                "max_abs_vorticity": float(np.max(abs_omega)),
                "enstrophy": float(0.5 * dx * dy * np.sum(omega * omega)),
                },
            )
        )
        frames.append((step * dt, omega))
    return rows, frames


def parse_max_div(spec: RunSpec, text: str, cfg: dict) -> List[dict]:
    dt = float(cfg.get("dt", 1.0))
    rows = []
    for match in MAX_DIV_RE.finditer(text):
        step = int(match.group(1))
        rows.append(
            merge_row(
                base_row(spec),
                {
                "step": step,
                "time": step * dt,
                "max_div": float(match.group(3)),
                "i": int(match.group(4)),
                "j": int(match.group(5)),
                },
            )
        )
    return rows


def parse_done_time(text: str) -> Optional[float]:
    matches = list(DONE_RE.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group(1))


def save_vorticity_images(
    spec: RunSpec,
    frames: List[Tuple[float, object]],
    img_root: Path,
    image_formats: Iterable[str],
) -> None:
    if plt is None or np is None or not frames:
        return
    image_dir = img_root / "vorticity"
    image_dir.mkdir(parents=True, exist_ok=True)
    sim_time, omega = frames[-1]
    vmax = float(np.nanmax(np.abs(omega))) if omega.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(
        omega,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_title(f"{label_for(base_row(spec))}, t={sim_time:g}s")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    fig.colorbar(image, ax=ax, label="vorticity")
    fig.tight_layout()
    save_figure(fig, image_dir / f"{spec.name}_final_vorticity", formats=image_formats)
    plt.close(fig)


def run_one(
    spec: RunSpec,
    binary: Path,
    keep_raw: bool,
    analysis: str,
    no_run_logs: bool,
    write_images: bool,
    img_root: Path,
    image_formats: Iterable[str],
) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(spec.config_path)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(spec.threads)
    env.setdefault("OMP_PROC_BIND", "spread")
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_DYNAMIC", "false")

    command = [str(binary), str(spec.config_path)]
    print(f"[run] {spec.name}: OMP_NUM_THREADS={spec.threads} {' '.join(command)}")
    start = time.perf_counter()
    result = run_process(command, cwd=PICM_ROOT, env=env)
    wall_time = time.perf_counter() - start

    if not no_run_logs or result.returncode != 0:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        (spec.run_dir / "stdout.log").write_text(result.stdout)
        (spec.run_dir / "stderr.log").write_text(result.stderr)
    combined = result.stdout + "\n" + result.stderr

    kinetic_rows = []
    div_rows = []
    vorticity_rows = []
    try:
        if result.returncode == 0:
            try:
                kinetic_rows = extract_kinetic_energy(spec, cfg)
            except Exception as exc:
                print(f"[warn] {spec.name}: could not extract kinetic energy: {exc}")
            if analysis in ("vorticity", "ppc"):
                try:
                    vorticity_rows, vorticity_frames = extract_vorticity(spec, cfg)
                    if write_images:
                        save_vorticity_images(spec, vorticity_frames, img_root, image_formats)
                except Exception as exc:
                    print(f"[warn] {spec.name}: could not extract vorticity: {exc}")
            div_rows = parse_max_div(spec, combined, cfg)
    finally:
        if not keep_raw and spec.raw_dir.exists():
            shutil.rmtree(spec.raw_dir)
        if no_run_logs and spec.run_dir.exists():
            try:
                spec.run_dir.rmdir()
            except OSError:
                pass
        if no_run_logs:
            for directory in (spec.run_dir.parent, spec.run_dir.parent.parent):
                try:
                    directory.rmdir()
                except OSError:
                    pass

    done_time = parse_done_time(combined)
    summary = merge_row(base_row(spec), {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "wall_time_s": wall_time,
        "reported_time_s": done_time if done_time is not None else "",
        "final_kinetic_energy": kinetic_rows[-1]["kinetic_energy"] if kinetic_rows else "",
        "final_normalized_ke": kinetic_rows[-1]["normalized_ke"] if kinetic_rows else "",
        "final_enstrophy": vorticity_rows[-1]["enstrophy"] if vorticity_rows else "",
        "final_mean_abs_vorticity": vorticity_rows[-1]["mean_abs_vorticity"] if vorticity_rows else "",
        "final_max_abs_vorticity": vorticity_rows[-1]["max_abs_vorticity"] if vorticity_rows else "",
        "max_div": max((row["max_div"] for row in div_rows), default=""),
        "config": str(spec.config_path),
    })
    if result.returncode != 0:
        print(f"[fail] {spec.name} exited with {result.returncode}")
    return summary, kinetic_rows, div_rows, vorticity_rows


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if not rows:
        tmp_path.write_text("")
        tmp_path.replace(path)
        return
    columns = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
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


def checkpoint(
    out_root: Path,
    summary_rows: List[dict],
    ke_rows: List[dict],
    div_rows: List[dict],
    vorticity_rows: List[dict],
) -> None:
    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "kinetic_energy.csv", ke_rows)
    write_csv(out_root / "max_div.csv", div_rows)
    write_csv(out_root / "vorticity.csv", vorticity_rows)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: List[float]) -> float:
    if not values:
        return float("nan")
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def optional_floats(rows: List[dict], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in ("", None):
            values.append(float(value))
    return values


def save_comparison_summary(summary_rows: List[dict], out_root: Path) -> None:
    grouped = {}
    for row in summary_rows:
        if row.get("status") != "ok":
            continue
        key = (
            row["test"],
            row["method"],
            str(row["ppc"]),
            str(row.get("flip_coef_pic", "")),
        )
        grouped.setdefault(key, []).append(row)

    rows = []
    for (test, method, ppc, flip_coef_pic), group in sorted(grouped.items()):
        wall = optional_floats(group, "wall_time_s")
        final_ke = optional_floats(group, "final_kinetic_energy")
        final_norm_ke = optional_floats(group, "final_normalized_ke")
        max_div = optional_floats(group, "max_div")
        enstrophy = optional_floats(group, "final_enstrophy")
        mean_abs_vorticity = optional_floats(group, "final_mean_abs_vorticity")
        max_abs_vorticity = optional_floats(group, "final_max_abs_vorticity")
        rows.append(
            {
                "test": test,
                "method": method,
                "ppc": ppc,
                "flip_coef_pic": flip_coef_pic,
                "runs": len(group),
                "wall_time_mean_s": mean(wall),
                "wall_time_std_s": std(wall),
                "final_kinetic_energy_mean": mean(final_ke),
                "final_kinetic_energy_std": std(final_ke),
                "final_normalized_ke_mean": mean(final_norm_ke),
                "final_normalized_ke_std": std(final_norm_ke),
                "max_div_max": max(max_div) if max_div else "",
                "final_enstrophy_mean": mean(enstrophy) if enstrophy else "",
                "final_enstrophy_std": std(enstrophy) if enstrophy else "",
                "final_mean_abs_vorticity_mean": mean(mean_abs_vorticity)
                if mean_abs_vorticity
                else "",
                "final_mean_abs_vorticity_std": std(mean_abs_vorticity)
                if mean_abs_vorticity
                else "",
                "final_max_abs_vorticity_max": max(max_abs_vorticity)
                if max_abs_vorticity
                else "",
            }
        )
    write_csv(out_root / "comparison_summary.csv", rows)


def label_for(row: dict) -> str:
    label = f"{row['method'].upper()} ppc={row['ppc']}"
    if row.get("flip_coef_pic") not in ("", None):
        label += f" coefPic={row['flip_coef_pic']}"
    return label


def postprocess_csv(out_root: Path) -> None:
    summary_rows = [row for row in read_csv(out_root / "summary.csv") if row.get("status") == "ok"]
    if summary_rows:
        save_comparison_summary(summary_rows, out_root)


def finite_values(rows: List[dict], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            values.append(parsed)
    return values


def rows_for_test_ppc(rows: List[dict], test: str, ppc: str) -> List[dict]:
    return [
        row
        for row in rows
        if row.get("test") == test
        and row.get("ppc") == ppc
        and int(row.get("repeat", 0)) == 0
    ]


def group_rows_by_run(rows: List[dict]) -> Dict[str, List[dict]]:
    by_run: Dict[str, List[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    return by_run


def mark_no_data(ax, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def plot_metric_series(
    ax,
    rows: List[dict],
    y_key: str,
    y_label: str,
    title: str,
    missing_message: str,
) -> bool:
    if not rows:
        mark_no_data(ax, title, missing_message)
        return False

    plotted = False
    for _run, group in sorted(group_rows_by_run(rows).items()):
        points = []
        for row in sorted(group, key=lambda item: float(item["time"])):
            value = row.get(y_key)
            if value in ("", None):
                continue
            parsed = float(value)
            if math.isfinite(parsed):
                points.append((float(row["time"]), parsed))
        if not points:
            continue
        ax.plot(
            [time for time, _value in points],
            [value for _time, value in points],
            lw=1.5,
            label=label_for(group[0]),
        )
        plotted = True

    if not plotted:
        mark_no_data(ax, title, missing_message)
        return False

    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return True


def plot_energy_vorticity_ppc(
    test: str,
    ppc: str,
    ke_rows: List[dict],
    vorticity_rows: List[dict],
    plot_dir: Path,
    image_formats: Iterable[str],
) -> bool:
    energy_rows = rows_for_test_ppc(ke_rows, test, ppc)
    vort_rows = rows_for_test_ppc(vorticity_rows, test, ppc)
    if not energy_rows and not vort_rows:
        return False

    fig, (energy_ax, vort_ax) = plt.subplots(1, 2, figsize=(15, 5))
    plot_metric_series(
        energy_ax,
        energy_rows,
        "normalized_ke",
        "Normalized kinetic energy",
        "Energy",
        "No energy CSV rows",
    )
    plot_metric_series(
        vort_ax,
        vort_rows,
        "enstrophy",
        "Enstrophy",
        "Vorticity",
        "No vorticity CSV rows",
    )
    fig.suptitle(f"{test}: energy and vorticity, ppc={ppc}")
    fig.tight_layout()
    save_figure(fig, plot_dir / f"{test}_energy_vorticity_ppc{ppc}", formats=image_formats)
    plt.close(fig)
    return True


def plot_all(
    out_root: Path,
    img_root: Path,
    image_formats: Iterable[str] = ("png", "svg", "pdf", "jpg"),
) -> None:
    if plt is None:
        print("[plot] matplotlib not available; skipping plots")
        return

    summary_rows = [row for row in read_csv(out_root / "summary.csv") if row.get("status") == "ok"]
    ke_rows = read_csv(out_root / "kinetic_energy.csv")
    vorticity_rows = read_csv(out_root / "vorticity.csv")
    div_rows = read_csv(out_root / "max_div.csv")
    if not summary_rows:
        return
    if div_rows and not ke_rows and not vorticity_rows:
        print(
            f"[plot] {out_root}: max_div.csv has rows, but kinetic_energy.csv "
            "and vorticity.csv are empty; skipping max-div-only plots"
        )
    postprocess_csv(out_root)

    plot_dir = img_root
    plot_dir.mkdir(parents=True, exist_ok=True)

    for test in sorted({row["test"] for row in summary_rows}):
        test_summary = [row for row in summary_rows if row["test"] == test]
        groups = {}
        for row in test_summary:
            key = (row["method"], row["ppc"], row.get("flip_coef_pic", ""))
            groups.setdefault(key, []).append(row)

        labels = []
        wall = []
        final_ke = []
        for _key, group in sorted(groups.items()):
            labels.append(label_for(group[0]))
            wall.append(mean([float(row["wall_time_s"]) for row in group]))
            ke_values = finite_values(group, "final_normalized_ke")
            final_ke.append(mean(ke_values))

        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(labels)), 5))
        ax.bar(range(len(labels)), wall)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_ylabel("Wall time [s]")
        ax.set_title(f"{test}: runtime")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save_figure(fig, plot_dir / f"{test}_runtime", formats=image_formats)
        plt.close(fig)

        if any(math.isfinite(value) for value in final_ke):
            fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(labels)), 5))
            ax.bar(range(len(labels)), final_ke)
            ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
            ax.set_ylabel("Final normalized kinetic energy")
            ax.set_title(f"{test}: final kinetic energy")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            save_figure(fig, plot_dir / f"{test}_final_ke", formats=image_formats)
            plt.close(fig)

        for ppc in sorted({row["ppc"] for row in test_summary}, key=lambda value: int(value)):
            plot_energy_vorticity_ppc(
                test,
                ppc,
                ke_rows,
                vorticity_rows,
                plot_dir,
                image_formats,
            )
    print(f"[plot] wrote plots under {plot_dir}")


def print_report_tests(tests: Iterable[str]) -> None:
    print("[tests] selected report comparisons:")
    for test in tests:
        info = REPORT_TESTS[test]
        print(f"  {test}: {info['config']}")
        print(f"    {info['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="all", help="comma list, or all")
    parser.add_argument("--methods", default="pic,flip,apic", help="comma list")
    parser.add_argument(
        "--analysis",
        choices=("energy", "vorticity", "ppc"),
        default="energy",
        help="controls output fields and plot emphasis",
    )
    parser.add_argument("--ppc", default="3", help="comma list for ppcx=ppcy")
    parser.add_argument(
        "--flip-coef-pic",
        default="0.05",
        help="comma list for FLIP PIC blending coefficient",
    )
    parser.add_argument("--threads", default=os.environ.get("THREADS"))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "1")))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--misc-dir", type=Path)
    parser.add_argument("--img-dir", type=Path)
    parser.add_argument("--image-formats", default="png,svg,pdf,jpg")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-report-release")
    parser.add_argument(
        "--build-jobs",
        type=int,
        default=int(os.environ.get("BUILD_JOBS", str(scheduler_threads()))),
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-run-logs", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--kernel-order", type=int)
    parser.add_argument("--solver-type")
    parser.add_argument("--solver-tolerance", type=float)
    parser.add_argument("--solver-max-iterations", type=int)
    args = parser.parse_args()

    args.out = args.out.resolve()
    args.misc_dir = (args.misc_dir or default_misc_dir(args.out)).resolve()
    args.img_dir = (args.img_dir or default_img_dir(args.out)).resolve()
    args.image_formats = parse_formats(args.image_formats)
    args.build_dir = args.build_dir.resolve()

    if args.plot_only:
        postprocess_csv(args.out)
        if not args.no_plots:
            plot_all(args.out, args.img_dir, args.image_formats)
        return 0

    specs = make_specs(args)
    if not specs:
        print("[error] no runs selected")
        return 1
    write_plan_and_configs(specs, args)
    print_report_tests(dict.fromkeys(spec.test for spec in specs))
    print(f"[plan] wrote {args.out / 'plan.csv'}")
    for spec in specs:
        print(
            f"  {spec.test:20s} {spec.method:4s} ppc={spec.ppc:<2d} "
            f"coefPic={'' if spec.flip_coef_pic is None else spec.flip_coef_pic} "
            f"threads={spec.threads:<3d} repeat={spec.repeat}"
        )

    if args.dry_run:
        print("[dry-run] configs generated; no build and no simulation launched")
        return 0

    binary = build_binary(args.skip_build, args.build_jobs, args.build_dir)
    summary_rows = [] if args.force else read_csv(args.out / "summary.csv")
    ke_rows = [] if args.force else read_csv(args.out / "kinetic_energy.csv")
    div_rows = [] if args.force else read_csv(args.out / "max_div.csv")
    vorticity_rows = [] if args.force else read_csv(args.out / "vorticity.csv")
    completed = {
        row["run"]
        for row in summary_rows
        if row.get("run") and row.get("status") == "ok"
    }

    failures = 0
    for spec in specs:
        if spec.name in completed:
            print(f"[skip] {spec.name}: already present in summary.csv")
            continue
        summary_rows = drop_run(summary_rows, spec.name)
        ke_rows = drop_run(ke_rows, spec.name)
        div_rows = drop_run(div_rows, spec.name)
        vorticity_rows = drop_run(vorticity_rows, spec.name)
        summary, ke, div, vort = run_one(
            spec,
            binary,
            args.keep_raw,
            args.analysis,
            args.no_run_logs,
            not args.no_plots,
            args.img_dir,
            args.image_formats,
        )
        summary_rows.append(summary)
        ke_rows.extend(ke)
        div_rows.extend(div)
        vorticity_rows.extend(vort)
        if summary["status"] != "ok":
            failures += 1
        checkpoint(args.out, summary_rows, ke_rows, div_rows, vorticity_rows)
        save_comparison_summary(summary_rows, args.out)
        print(f"[checkpoint] CSV files updated after {spec.name}")

    checkpoint(args.out, summary_rows, ke_rows, div_rows, vorticity_rows)
    save_comparison_summary(summary_rows, args.out)
    if not args.no_plots:
        plot_all(args.out, args.img_dir, args.image_formats)
    print(f"[done] CSV files: {args.out}")
    if failures:
        print(f"[error] {failures} simulation(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
