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
    VIDEO_DIR,
    default_img_dir,
    default_misc_dir,
)
from picm_postpro.plots import parse_formats, save_figure, PALETTE, style_ax, style_legend

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


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


REPORT_TESTS = {
    "falling-block-water": {
        "config": first_existing_path(
            PICM_ROOT / "test" / "PIC" / "extra" / "freeFallInWater.json",
        ),
        "reason": "falling block in water; useful for vorticity and energy conservation",
    },
    "freeFallInWater": {
        "config": first_existing_path(
            PICM_ROOT / "test" / "PIC" / "extra" / "freeFallInWater.json",
        ),
        "reason": "falling block in water; useful for vorticity and energy conservation",
    },
    "von-karman": {
        "config": first_existing_path(
            PICM_ROOT / "test" / "PIC" / "extra" / "von-karman.json",
            PICM_ROOT / "test" / "PIC" / "section-5-5-1" / "von-karman.json",
        ),
        "reason": "wake dynamics behind an obstacle; exposes numerical diffusion",
    },
    "dambreak": {
        "config": first_existing_path(
            PICM_ROOT / "test" / "PIC" / "extra" / "dambreak.json",
        ),
        "reason": "free-surface collapse; exposes interface stability",
    },
    "vases-communicants": {
        "config": first_existing_path(
            PICM_ROOT
            / "test"
            / "PIC"
            / "extra"
            / "vases-communicants"
            / "vases-communicants.json",
            PICM_ROOT / "test" / "PIC" / "section-5-5-3" / "vases-communicants.json",
        ),
        "reason": "hydrostatic balancing; exposes damping and long-time stability",
    },
}

REPORT_METHODS = ("pic", "flip", "apic")
VIDEO_METHODS = ("pic", "flip", "apic", "mixed")
DEFAULT_FLIP_COEF_PIC = "0,0.01,0.05,0.1"
METHOD_COLORS = {
    "pic":   PALETTE["blue"],
    "flip":  PALETTE["orange"],
    "apic":  PALETTE["green"],
    "mixed": PALETTE["purple"],
}
MIXED_FLIP_COLORS = {
    0.01: PALETTE["purple"],
    0.05: PALETTE["pink"],
    0.1:  PALETTE["grey"],
}
METHOD_ORDER = {"pic": 0, "flip": 1, "apic": 2, "mixed": 3}

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


class VideoOptions(NamedTuple):
    enabled: bool
    methods: Tuple[str, ...]
    root: Path
    fps: int
    sample: int
    width: int
    height: int
    workers: int
    cmap: str
    mode: str


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


def parse_int_list(
    value: Optional[str],
    default: Iterable[int],
    *,
    min_value: int = 1,
    value_name: str = "integer list",
) -> Tuple[int, ...]:
    if not value:
        return tuple(default)
    parsed = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        item = int(chunk)
        if item < min_value:
            raise ValueError(f"{value_name} values must be >= {min_value}")
        parsed.append(item)
    if not parsed:
        raise ValueError(f"empty {value_name}")
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


def parse_video_methods(value: Optional[str]) -> Tuple[str, ...]:
    if not value or value == "all":
        return VIDEO_METHODS
    parsed = []
    valid = set(VIDEO_METHODS)
    for chunk in value.split(","):
        item = chunk.strip().lower()
        if not item:
            continue
        if item not in valid:
            raise ValueError(f"unknown video method '{item}', expected one of {VIDEO_METHODS}")
        parsed.append(item)
    if not parsed:
        raise ValueError("empty video method list")
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


def default_run_threads() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 1


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


def disable_heavy_outputs(
    cfg: dict,
    *,
    need_vorticity_field: bool = False,
    need_particles: bool = False,
) -> None:
    cfg["write_u"] = False
    cfg["write_v"] = False
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_smoke"] = False
    cfg["write_norm_velocity"] = True
    cfg["write_vorticity"] = need_vorticity_field
    cfg["write_particles"] = need_particles


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

    disable_heavy_outputs(
        cfg,
        need_vorticity_field=args.analysis in ("vorticity", "ppc"),
        need_particles=args.write_particles or args.make_videos,
    )
    return cfg


def make_specs(args: argparse.Namespace) -> List[RunSpec]:
    tests = parse_name_list(args.test, REPORT_TESTS.keys())
    methods = parse_name_list(args.methods, REPORT_METHODS)
    ppcs = parse_int_list(args.ppc, (3,), min_value=0, value_name="ppc")
    threads = parse_int_list(args.threads, (default_run_threads(),), value_name="thread")
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


def require_numpy(task: str) -> None:
    if np is None:
        raise RuntimeError(f"numpy is required to extract {task}")


def extract_kinetic_energy(spec: RunSpec, cfg: dict) -> List[dict]:
    require_numpy("kinetic energy")
    pvd_path = spec.raw_dir / "normVelocity.pvd"
    if not pvd_path.exists():
        raise FileNotFoundError(f"missing velocity norm PVD: {pvd_path}")

    dx = float(cfg.get("dx", 1.0))
    dy = float(cfg.get("dy", 1.0))
    density = float(cfg.get("density", 1.0))
    dt = float(cfg.get("dt", 1.0))
    sampling_rate = int(cfg.get("sampling_rate", 1))
    paths = parse_pvd(pvd_path)
    if not paths:
        raise RuntimeError(f"{pvd_path} contains no VTK frames")

    values = []
    missing = 0
    for sample_index, vti_path in enumerate(paths):
        if not vti_path.exists():
            missing += 1
            continue
        speed = read_vti_field(vti_path, "normVelocity")
        speed_l2_sq = dx * dy * float(np.sum(speed * speed))
        velocity_l2 = math.sqrt(max(0.0, speed_l2_sq))
        kinetic_energy = 0.5 * density * speed_l2_sq
        step = sample_index * sampling_rate
        values.append((sample_index, step, step * dt, kinetic_energy, velocity_l2))
    if not values:
        raise RuntimeError(
            f"no readable normVelocity frames under {spec.raw_dir} "
            f"({missing} missing)"
        )

    reference = next(
        (ke for _sample, step, _time, ke, _l2 in values if step > 0 and abs(ke) > 1e-30),
        None,
    )
    if reference is None:
        reference = next((ke for _sample, _step, _time, ke, _l2 in values if abs(ke) > 1e-30), 1.0)
    reference_l2 = next(
        (
            l2
            for _sample, step, _time, _ke, l2 in values
            if step > 0 and abs(l2) > 1e-30
        ),
        None,
    )
    if reference_l2 is None:
        reference_l2 = next(
            (l2 for _sample, _step, _time, _ke, l2 in values if abs(l2) > 1e-30),
            1.0,
        )

    rows = []
    for sample_index, step, sim_time, kinetic_energy, velocity_l2 in values:
        rows.append(
            merge_row(
                base_row(spec),
                {
                "sample": sample_index,
                "step": step,
                "time": sim_time,
                "kinetic_energy": kinetic_energy,
                "normalized_ke": kinetic_energy / reference,
                "velocity_l2": velocity_l2,
                "normalized_velocity_l2": velocity_l2 / reference_l2,
                },
            )
        )
    return rows


def compute_vorticity(u_field, v_field, dx: float, dy: float):
    require_numpy("vorticity")
    ny = min(u_field.shape[0], v_field.shape[0] - 1)
    nx = min(u_field.shape[1] - 1, v_field.shape[1])
    if nx <= 1 or ny <= 1:
        raise ValueError(f"invalid u/v shapes for vorticity: u={u_field.shape}, v={v_field.shape}")
    dv_dx = (v_field[1:ny, 1:nx] - v_field[1:ny, 0 : nx - 1]) / dx
    du_dy = (u_field[1:ny, 1:nx] - u_field[0 : ny - 1, 1:nx]) / dy
    return dv_dx - du_dy


def vorticity_rows_from_frames(
    spec: RunSpec,
    cfg: dict,
    frames_by_sample: List[Tuple[int, object]],
) -> Tuple[List[dict], List[Tuple[float, object]]]:
    dx = float(cfg.get("dx", 1.0))
    dy = float(cfg.get("dy", 1.0))
    dt = float(cfg.get("dt", 1.0))
    sampling_rate = int(cfg.get("sampling_rate", 1))
    rows = []
    frames = []
    for sample_index, omega in frames_by_sample:
        step = sample_index * sampling_rate
        abs_omega = np.abs(omega)
        vorticity_l2 = math.sqrt(max(0.0, dx * dy * float(np.sum(omega * omega))))
        rows.append(
            merge_row(
                base_row(spec),
                {
                "sample": sample_index,
                "step": step,
                "time": step * dt,
                "mean_abs_vorticity": float(np.mean(abs_omega)),
                "max_abs_vorticity": float(np.max(abs_omega)),
                "vorticity_l2": vorticity_l2,
                "enstrophy": float(0.5 * vorticity_l2 * vorticity_l2),
                },
            )
        )
        frames.append((step * dt, omega))
    return rows, frames


def extract_vorticity_field(
    spec: RunSpec,
    cfg: dict,
) -> Tuple[List[dict], List[Tuple[float, object]]]:
    pvd_path = spec.raw_dir / "vorticity.pvd"
    if not pvd_path.exists():
        raise FileNotFoundError(f"missing vorticity PVD: {pvd_path}")
    paths = parse_pvd(pvd_path)
    if not paths:
        raise RuntimeError(f"{pvd_path} contains no VTK frames")

    frames_by_sample = []
    missing = 0
    for sample_index, vti_path in enumerate(paths):
        if not vti_path.exists():
            missing += 1
            continue
        frames_by_sample.append(
            (sample_index, read_vti_field(vti_path, "vorticity"))
        )
    if not frames_by_sample:
        raise RuntimeError(
            f"no readable vorticity frames under {spec.raw_dir} ({missing} missing)"
        )
    return vorticity_rows_from_frames(spec, cfg, frames_by_sample)


def extract_vorticity_from_velocity_components(
    spec: RunSpec,
    cfg: dict,
) -> Tuple[List[dict], List[Tuple[float, object]]]:
    u_pvd = spec.raw_dir / "u.pvd"
    v_pvd = spec.raw_dir / "v.pvd"
    missing_pvds = [str(path) for path in (u_pvd, v_pvd) if not path.exists()]
    if missing_pvds:
        raise FileNotFoundError(
            f"missing velocity component PVD(s): {', '.join(missing_pvds)}"
        )
    u_paths = parse_pvd(u_pvd)
    v_paths = parse_pvd(v_pvd)
    if not u_paths or not v_paths:
        raise RuntimeError(f"{u_pvd} or {v_pvd} contains no VTK frames")

    dx = float(cfg.get("dx", 1.0))
    dy = float(cfg.get("dy", 1.0))
    frames_by_sample = []
    missing = 0
    for sample_index, (u_path, v_path) in enumerate(zip(u_paths, v_paths)):
        if not u_path.exists() or not v_path.exists():
            missing += 1
            continue
        u_field = read_vti_field(u_path, "u")
        v_field = read_vti_field(v_path, "v")
        frames_by_sample.append(
            (sample_index, compute_vorticity(u_field, v_field, dx, dy))
        )
    if not frames_by_sample:
        raise RuntimeError(
            f"no readable u/v frames under {spec.raw_dir} ({missing} missing pairs)"
        )
    return vorticity_rows_from_frames(spec, cfg, frames_by_sample)


def extract_vorticity(spec: RunSpec, cfg: dict) -> Tuple[List[dict], List[Tuple[float, object]]]:
    require_numpy("vorticity")
    try:
        return extract_vorticity_field(spec, cfg)
    except FileNotFoundError as field_error:
        try:
            return extract_vorticity_from_velocity_components(spec, cfg)
        except FileNotFoundError as component_error:
            raise FileNotFoundError(
                f"{field_error}; fallback also failed: {component_error}"
            ) from component_error


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
    style_ax(ax, xlabel="$i$", ylabel="$j$",
             title=f"{label_for(base_row(spec))}, $t={sim_time:g}$ s",
             grid=False, minorticks=False)
    fig.colorbar(image, ax=ax, label="vorticity")
    fig.tight_layout()
    save_figure(fig, image_dir / f"{spec.name}_final_vorticity", formats=image_formats)
    plt.close(fig)


def video_title_for(spec: RunSpec) -> str:
    row = base_row(spec)
    return f"{spec.test} - {label_for(row, include_ppc=True)}"


def should_make_video(spec: RunSpec, options: VideoOptions) -> bool:
    if not options.enabled:
        return False
    return method_kind(base_row(spec)) in options.methods


def write_particle_video(spec: RunSpec, options: VideoOptions) -> None:
    pvd_path = spec.raw_dir / "particles.pvd"
    if not pvd_path.exists():
        print(f"[video] {spec.name}: particles.pvd not found; skipping")
        return
    try:
        import particles_to_mp4

        vtp_paths = [path for path in particles_to_mp4.parse_pvd(pvd_path) if path.exists()]
        if options.sample > 1:
            vtp_paths = vtp_paths[:: options.sample]
        if not vtp_paths:
            print(f"[video] {spec.name}: no VTP frames found; skipping")
            return
        out_path = options.root / f"{spec.name}.mp4"
        particles_to_mp4.build_mp4(
            vtp_paths,
            out_path,
            fps=options.fps,
            cmap_name=options.cmap,
            width=options.width,
            height=options.height,
            title=video_title_for(spec),
            mode=options.mode,
            n_workers=options.workers,
        )
    except Exception as exc:
        print(f"[warn] {spec.name}: could not write particle video: {exc}")


def run_one(
    spec: RunSpec,
    binary: Path,
    keep_raw: bool,
    defer_extraction: bool,
    analysis: str,
    no_run_logs: bool,
    write_images: bool,
    img_root: Path,
    image_formats: Iterable[str],
    video_options: VideoOptions,
) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(spec.config_path)
    effective_keep_raw = keep_raw or defer_extraction
    effective_no_run_logs = no_run_logs and not defer_extraction

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

    if not effective_no_run_logs or result.returncode != 0:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        (spec.run_dir / "stdout.log").write_text(result.stdout)
        (spec.run_dir / "stderr.log").write_text(result.stderr)
    combined = result.stdout + "\n" + result.stderr

    kinetic_rows = []
    div_rows = []
    vorticity_rows = []
    extraction_errors = []
    try:
        if result.returncode == 0:
            if defer_extraction:
                print(
                    f"[defer] {spec.name}: raw output kept for later "
                    "--extract-only post-processing"
                )
            else:
                try:
                    kinetic_rows = extract_kinetic_energy(spec, cfg)
                except Exception as exc:
                    extraction_errors.append(f"kinetic energy: {exc}")
                    print(f"[warn] {spec.name}: could not extract kinetic energy: {exc}")
                if analysis in ("vorticity", "ppc"):
                    try:
                        vorticity_rows, vorticity_frames = extract_vorticity(spec, cfg)
                        if write_images:
                            save_vorticity_images(spec, vorticity_frames, img_root, image_formats)
                    except Exception as exc:
                        extraction_errors.append(f"vorticity: {exc}")
                        print(f"[warn] {spec.name}: could not extract vorticity: {exc}")
            div_rows = parse_max_div(spec, combined, cfg)
            if not defer_extraction and should_make_video(spec, video_options):
                write_particle_video(spec, video_options)
    finally:
        if not effective_keep_raw and not extraction_errors and spec.raw_dir.exists():
            shutil.rmtree(spec.raw_dir)
        if effective_no_run_logs and spec.run_dir.exists():
            try:
                spec.run_dir.rmdir()
            except OSError:
                pass
        if effective_no_run_logs:
            for directory in (spec.run_dir.parent, spec.run_dir.parent.parent):
                try:
                    directory.rmdir()
                except OSError:
                    pass

    done_time = parse_done_time(combined)
    if result.returncode != 0:
        status = "failed"
        error = ""
    elif defer_extraction:
        status = "raw_pending"
        error = "analysis extraction deferred; raw data kept for --extract-only"
    elif extraction_errors:
        status = "failed"
        error = "; ".join(extraction_errors)
    else:
        status = "ok"
        error = ""
    summary = merge_row(base_row(spec), {
        "status": status,
        "returncode": result.returncode,
        "wall_time_s": wall_time,
        "reported_time_s": done_time if done_time is not None else "",
        "final_kinetic_energy": kinetic_rows[-1]["kinetic_energy"] if kinetic_rows else "",
        "final_normalized_ke": kinetic_rows[-1]["normalized_ke"] if kinetic_rows else "",
        "final_velocity_l2": kinetic_rows[-1]["velocity_l2"] if kinetic_rows else "",
        "final_normalized_velocity_l2": kinetic_rows[-1]["normalized_velocity_l2"] if kinetic_rows else "",
        "final_enstrophy": vorticity_rows[-1]["enstrophy"] if vorticity_rows else "",
        "final_vorticity_l2": vorticity_rows[-1]["vorticity_l2"] if vorticity_rows else "",
        "final_mean_abs_vorticity": vorticity_rows[-1]["mean_abs_vorticity"] if vorticity_rows else "",
        "final_max_abs_vorticity": vorticity_rows[-1]["max_abs_vorticity"] if vorticity_rows else "",
        "max_div": max((row["max_div"] for row in div_rows), default=""),
        "config": str(spec.config_path),
        "error": error,
    })
    if result.returncode != 0:
        print(f"[fail] {spec.name} exited with {result.returncode}")
    elif extraction_errors:
        print(f"[fail] {spec.name}: required analysis extraction failed")
    return summary, kinetic_rows, div_rows, vorticity_rows


def read_run_logs(spec: RunSpec) -> str:
    chunks = []
    for name in ("stdout.log", "stderr.log"):
        path = spec.run_dir / name
        if path.exists():
            try:
                chunks.append(path.read_text(errors="replace"))
            except Exception as exc:
                chunks.append(f"[warn] could not read {path}: {exc}")
    return "\n".join(chunks)


def extract_one_from_raw(
    spec: RunSpec,
    analysis: str,
    existing_summary: dict,
    write_images: bool,
    img_root: Path,
    image_formats: Iterable[str],
) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    cfg = load_config(spec.config_path)
    kinetic_rows = []
    div_rows = []
    vorticity_rows = []
    extraction_errors = []

    if not spec.raw_dir.exists():
        extraction_errors.append(f"missing raw directory: {spec.raw_dir}")
    else:
        try:
            kinetic_rows = extract_kinetic_energy(spec, cfg)
        except Exception as exc:
            extraction_errors.append(f"kinetic energy: {exc}")
            print(f"[warn] {spec.name}: could not extract kinetic energy: {exc}")
        if analysis in ("vorticity", "ppc"):
            try:
                vorticity_rows, vorticity_frames = extract_vorticity(spec, cfg)
                if write_images:
                    save_vorticity_images(spec, vorticity_frames, img_root, image_formats)
            except Exception as exc:
                extraction_errors.append(f"vorticity: {exc}")
                print(f"[warn] {spec.name}: could not extract vorticity: {exc}")
        combined_logs = read_run_logs(spec)
        div_rows = parse_max_div(spec, combined_logs, cfg)
    done_time = parse_done_time(combined_logs) if spec.raw_dir.exists() else None
    wall_time_s = existing_summary.get("wall_time_s", "")
    parsed_wall_time = optional_float(wall_time_s)
    if (wall_time_s in ("", None) or parsed_wall_time == 0.0) and done_time is not None:
        wall_time_s = done_time
    if wall_time_s in ("", None):
        wall_time_s = 0.0

    summary = merge_row(
        base_row(spec),
        {
            "status": "ok" if not extraction_errors else "failed",
            "returncode": existing_summary.get("returncode", 0),
            "wall_time_s": wall_time_s,
            "reported_time_s": existing_summary.get("reported_time_s", "")
            or (done_time if done_time is not None else ""),
            "final_kinetic_energy": kinetic_rows[-1]["kinetic_energy"] if kinetic_rows else "",
            "final_normalized_ke": kinetic_rows[-1]["normalized_ke"] if kinetic_rows else "",
            "final_velocity_l2": kinetic_rows[-1]["velocity_l2"] if kinetic_rows else "",
            "final_normalized_velocity_l2": kinetic_rows[-1]["normalized_velocity_l2"]
            if kinetic_rows
            else "",
            "final_enstrophy": vorticity_rows[-1]["enstrophy"] if vorticity_rows else "",
            "final_vorticity_l2": vorticity_rows[-1]["vorticity_l2"] if vorticity_rows else "",
            "final_mean_abs_vorticity": vorticity_rows[-1]["mean_abs_vorticity"]
            if vorticity_rows
            else "",
            "final_max_abs_vorticity": vorticity_rows[-1]["max_abs_vorticity"]
            if vorticity_rows
            else "",
            "max_div": max((row["max_div"] for row in div_rows), default=""),
            "config": str(spec.config_path),
            "error": "; ".join(extraction_errors),
        },
    )
    if extraction_errors:
        print(f"[fail] {spec.name}: required analysis extraction failed")
    return summary, kinetic_rows, div_rows, vorticity_rows


def extract_from_raw(
    specs: List[RunSpec],
    args: argparse.Namespace,
) -> int:
    existing = {row.get("run"): row for row in read_csv(args.out / "summary.csv")}
    summary_rows = []
    ke_rows = []
    div_rows = []
    vorticity_rows = []
    failures = 0
    for spec in specs:
        summary, ke, div, vort = extract_one_from_raw(
            spec,
            args.analysis,
            existing.get(spec.name, {}),
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
    if not args.no_plots:
        plot_all(args.out, args.img_dir, args.image_formats)
    print(f"[extract] rebuilt CSV files from raw data under {args.out}")
    return 1 if failures else 0


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


def config_density(row: dict, cache: Dict[str, float]) -> float:
    config = row.get("config", "")
    if not config:
        return 1.0
    if config not in cache:
        try:
            cache[config] = float(load_config(Path(config)).get("density", 1.0))
        except Exception:
            cache[config] = 1.0
    return cache[config]


def backfill_l2_columns(
    summary_rows: List[dict],
    ke_rows: List[dict],
    vorticity_rows: List[dict],
) -> None:
    summary_by_run = {row.get("run"): row for row in summary_rows if row.get("run")}
    density_cache: Dict[str, float] = {}

    ke_by_run: Dict[str, List[dict]] = {}
    for row in ke_rows:
        run = row.get("run")
        if not run:
            continue
        summary = summary_by_run.get(run, {})
        density = config_density(summary, density_cache)
        kinetic_energy = optional_float(row.get("kinetic_energy"))
        if row.get("velocity_l2") in ("", None) and kinetic_energy is not None and density > 0.0:
            row["velocity_l2"] = math.sqrt(max(0.0, 2.0 * kinetic_energy / density))
        ke_by_run.setdefault(run, []).append(row)

    for rows in ke_by_run.values():
        reference = next(
            (
                optional_float(row.get("velocity_l2"))
                for row in rows
                if optional_float(row.get("step"), 0.0) > 0.0
                and optional_float(row.get("velocity_l2")) not in (None, 0.0)
            ),
            None,
        )
        if reference is None:
            reference = next(
                (
                    optional_float(row.get("velocity_l2"))
                    for row in rows
                    if optional_float(row.get("velocity_l2")) not in (None, 0.0)
                ),
                None,
            )
        for row in rows:
            velocity_l2 = optional_float(row.get("velocity_l2"))
            if velocity_l2 is not None and reference:
                row["normalized_velocity_l2"] = velocity_l2 / reference

    vort_by_run: Dict[str, List[dict]] = {}
    for row in vorticity_rows:
        run = row.get("run")
        if not run:
            continue
        enstrophy = optional_float(row.get("enstrophy"))
        if row.get("vorticity_l2") in ("", None) and enstrophy is not None:
            row["vorticity_l2"] = math.sqrt(max(0.0, 2.0 * enstrophy))
        vort_by_run.setdefault(run, []).append(row)

    for row in summary_rows:
        run = row.get("run")
        if not run:
            continue
        if ke_by_run.get(run):
            last_ke = ke_by_run[run][-1]
            row["final_velocity_l2"] = last_ke.get("velocity_l2", "")
            row["final_normalized_velocity_l2"] = last_ke.get(
                "normalized_velocity_l2", ""
            )
        if vort_by_run.get(run):
            last_vort = vort_by_run[run][-1]
            row["final_vorticity_l2"] = last_vort.get("vorticity_l2", "")


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
        final_velocity_l2 = optional_floats(group, "final_velocity_l2")
        final_norm_velocity_l2 = optional_floats(group, "final_normalized_velocity_l2")
        max_div = optional_floats(group, "max_div")
        enstrophy = optional_floats(group, "final_enstrophy")
        vorticity_l2 = optional_floats(group, "final_vorticity_l2")
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
                "final_velocity_l2_mean": mean(final_velocity_l2)
                if final_velocity_l2
                else "",
                "final_velocity_l2_std": std(final_velocity_l2)
                if final_velocity_l2
                else "",
                "final_normalized_velocity_l2_mean": mean(final_norm_velocity_l2)
                if final_norm_velocity_l2
                else "",
                "final_normalized_velocity_l2_std": std(final_norm_velocity_l2)
                if final_norm_velocity_l2
                else "",
                "max_div_max": max(max_div) if max_div else "",
                "final_enstrophy_mean": mean(enstrophy) if enstrophy else "",
                "final_enstrophy_std": std(enstrophy) if enstrophy else "",
                "final_vorticity_l2_mean": mean(vorticity_l2)
                if vorticity_l2
                else "",
                "final_vorticity_l2_std": std(vorticity_l2)
                if vorticity_l2
                else "",
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


def optional_float(value, default: Optional[float] = None) -> Optional[float]:
    if value in ("", None):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def method_kind(row_or_spec) -> str:
    method = row_or_spec["method"] if isinstance(row_or_spec, dict) else row_or_spec.method
    if method != "flip":
        return method
    coef = (
        optional_float(row_or_spec.get("flip_coef_pic"))
        if isinstance(row_or_spec, dict)
        else row_or_spec.flip_coef_pic
    )
    if coef is None or abs(float(coef)) < 1e-14:
        return "flip"
    return "mixed"


def display_float(value) -> str:
    parsed = optional_float(value)
    if parsed is None:
        return ""
    return f"{parsed:g}"


def label_for(row: dict, *, include_ppc: bool = True) -> str:
    method = row.get("method", "")
    kind = method_kind(row)
    if method == "pic":
        label = "PIC"
    elif method == "apic":
        label = "APIC"
    elif kind == "flip":
        label = "FLIP"
    elif kind == "mixed":
        label = f"FLIP {display_float(row.get('flip_coef_pic'))}"
    else:
        label = str(method).upper()
    if include_ppc and row.get("ppc") not in ("", None):
        label += f" ppc={row['ppc']}"
    return label


def method_sort_key(row: dict) -> Tuple[int, float, int]:
    kind = method_kind(row)
    coef = optional_float(row.get("flip_coef_pic"), 0.0) or 0.0
    ppc = int(float(row.get("ppc", 0) or 0))
    return METHOD_ORDER.get(kind, 99), coef, ppc


def method_color(row: dict):
    kind = method_kind(row)
    if kind == "mixed":
        coef = optional_float(row.get("flip_coef_pic"), 0.0) or 0.0
        for target, color in MIXED_FLIP_COLORS.items():
            if abs(coef - target) < 1e-12:
                return color
        return METHOD_COLORS["mixed"]
    return METHOD_COLORS.get(kind)


def postprocess_csv(out_root: Path) -> None:
    summary_rows = read_csv(out_root / "summary.csv")
    ke_rows = read_csv(out_root / "kinetic_energy.csv")
    vorticity_rows = read_csv(out_root / "vorticity.csv")
    if summary_rows:
        backfill_l2_columns(summary_rows, ke_rows, vorticity_rows)
        write_csv(out_root / "summary.csv", summary_rows)
        write_csv(out_root / "kinetic_energy.csv", ke_rows)
        write_csv(out_root / "vorticity.csv", vorticity_rows)
        save_comparison_summary(
            [row for row in summary_rows if row.get("status") == "ok"],
            out_root,
        )


def has_run_rows(rows: List[dict], run: str) -> bool:
    return any(row.get("run") == run for row in rows)


def missing_required_metrics(
    run: str,
    analysis: str,
    kinetic_rows: List[dict],
    vorticity_rows: List[dict],
) -> List[str]:
    missing = []
    if not has_run_rows(kinetic_rows, run):
        missing.append("kinetic_energy")
    if analysis in ("vorticity", "ppc") and not has_run_rows(vorticity_rows, run):
        missing.append("vorticity")
    return missing


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
    ]


def group_rows_by_key(rows: List[dict], y_key: str, key_func):
    grouped: Dict[Tuple, dict] = {}
    for row in rows:
        value = row.get(y_key)
        if value in ("", None):
            continue
        try:
            time_value = float(row["time"])
            y_value = float(value)
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(time_value) and math.isfinite(y_value)):
            continue
        key = key_func(row)
        bucket = grouped.setdefault(key, {"row": row, "times": {}})
        bucket["times"].setdefault(time_value, []).append(y_value)
    return grouped


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
    key_func,
    label_func,
    color_func=None,
    legend_columns: int = 1,
) -> bool:
    if not rows:
        mark_no_data(ax, title, missing_message)
        return False

    plotted = False
    grouped = group_rows_by_key(rows, y_key, key_func)
    for _key, bucket in sorted(grouped.items(), key=lambda item: method_sort_key(item[1]["row"])):
        times = sorted(bucket["times"])
        if not times:
            continue
        means = [mean(bucket["times"][time]) for time in times]
        deviations = [std(bucket["times"][time]) for time in times]
        row = bucket["row"]
        color = color_func(row) if color_func else None
        ax.plot(
            times,
            means,
            lw=1.5,
            label=label_func(row),
            color=color,
        )
        if any(math.isfinite(value) and value > 0.0 for value in deviations):
            lower = [m - s for m, s in zip(means, deviations)]
            upper = [m + s for m, s in zip(means, deviations)]
            ax.fill_between(times, lower, upper, color=color, alpha=0.12)
        plotted = True

    if not plotted:
        mark_no_data(ax, title, missing_message)
        return False

    style_ax(ax, xlabel="Time $t$ [s]", ylabel=y_label, title=title)
    style_legend(ax, many_threshold=4, max_columns=legend_columns)
    return True


def plot_method_metric_for_ppc(
    test: str,
    ppc: str,
    rows: List[dict],
    y_key: str,
    y_label: str,
    title: str,
    output_name: str,
    plot_dir: Path,
    image_formats: Iterable[str],
) -> bool:
    metric_rows = rows_for_test_ppc(rows, test, ppc)
    if not metric_rows:
        return False

    fig, ax = plt.subplots()
    ok = plot_metric_series(
        ax,
        metric_rows,
        y_key,
        y_label,
        title,
        f"No {y_label} CSV rows",
        key_func=lambda row: (row["method"], row.get("flip_coef_pic", "")),
        label_func=lambda row: label_for(row, include_ppc=False),
        color_func=method_color,
    )
    if not ok:
        plt.close(fig)
        return False
    fig.tight_layout()
    save_figure(fig, plot_dir / output_name, formats=image_formats)
    plt.close(fig)
    return True


def _ppc_errbar(
    test: str,
    summary_rows: List[dict],
    final_key: str,
    y_label: str,
    title: str,
    out_stem: Path,
    image_formats: Iterable[str],
) -> None:
    """Bar chart of a final scalar metric vs PPC, with std errorbars."""
    groups: Dict[int, List[float]] = {}
    for row in summary_rows:
        if row.get("test") != test or row.get("method") != "pic" or row.get("status") != "ok":
            continue
        value = optional_float(row.get(final_key))
        if value is None:
            continue
        groups.setdefault(int(float(row["ppc"])), []).append(value)
    if not groups:
        return
    xs = sorted(groups)
    ys = [mean(groups[ppc]) for ppc in xs]
    yerr = [std(groups[ppc]) for ppc in xs]
    fig, ax = plt.subplots()
    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", capsize=5, color=PALETTE["blue"])
    style_ax(ax, xlabel="Particles per cell (ppcx = ppcy)", ylabel=y_label, title=title)
    fig.tight_layout()
    save_figure(fig, out_stem, formats=image_formats)
    plt.close(fig)


def plot_all(
    out_root: Path,
    img_root: Path,
    image_formats: Iterable[str] = ("png", "pdf"),
) -> None:
    if plt is None:
        print("[plot] matplotlib not available; skipping plots")
        return

    postprocess_csv(out_root)
    summary_rows = [row for row in read_csv(out_root / "summary.csv") if row.get("status") == "ok"]
    ke_rows = read_csv(out_root / "kinetic_energy.csv")
    vorticity_rows = read_csv(out_root / "vorticity.csv")
    if not summary_rows:
        return

    for test in sorted({row["test"] for row in summary_rows}):
        test_summary = [row for row in summary_rows if row["test"] == test]
        ppcs = sorted({row["ppc"] for row in test_summary}, key=lambda v: int(v))

        for ppc in ppcs:
            plot_method_metric_for_ppc(
                test, ppc, ke_rows,
                "kinetic_energy", "Kinetic energy $E_k$",
                f"{test}: kinetic energy, ppc={ppc}",
                "energyL2",
                img_root / "energy",
                image_formats,
            )
            plot_method_metric_for_ppc(
                test, ppc, vorticity_rows,
                "vorticity_l2", r"Vorticity $\|\omega\|_2$",
                f"{test}: vorticity $L^2$ norm, ppc={ppc}",
                "vorticityL2",
                img_root / "vorticity",
                image_formats,
            )

        _ppc_errbar(
            test, summary_rows,
            "final_kinetic_energy", "Final kinetic energy $E_k$",
            f"{test}: PIC kinetic energy vs PPC",
            img_root / "ppc" / "ppc_energyL2",
            image_formats,
        )
        _ppc_errbar(
            test, summary_rows,
            "final_vorticity_l2", r"Final vorticity $\|\omega\|_2$",
            f"{test}: PIC vorticity $L^2$ vs PPC",
            img_root / "ppc" / "ppc_vorticityL2",
            image_formats,
        )

    print(f"[plot] wrote plots under {img_root}")


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
        default=DEFAULT_FLIP_COEF_PIC,
        help="comma list for FLIP PIC blending coefficient",
    )
    parser.add_argument("--threads", default=os.environ.get("THREADS"))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "1")))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--misc-dir", type=Path)
    parser.add_argument("--img-dir", type=Path)
    parser.add_argument("--image-formats", default="png,pdf")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-report-release")
    parser.add_argument(
        "--build-jobs",
        type=int,
        default=int(os.environ.get("BUILD_JOBS", str(scheduler_threads()))),
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument(
        "--defer-extraction",
        action="store_true",
        default=os.environ.get("PICM_DEFER_EXTRACTION", "").lower()
        in ("1", "true", "yes", "on"),
        help=(
            "run simulations only, keep raw output/logs, and leave CSV metrics "
            "for a later --extract-only run"
        ),
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="rebuild CSV metrics and plots from existing raw output without running PICM",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-run-logs", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--write-particles",
        action="store_true",
        help="write particles.pvd/VTP during runs",
    )
    parser.add_argument(
        "--make-videos",
        action="store_true",
        help="encode particle MP4 clips before raw output is cleaned",
    )
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-methods", default="pic,flip,apic")
    parser.add_argument("--video-cmap", default="viridis")
    parser.add_argument("--video-mode", choices=("speed", "density"), default="speed")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-sample", type=int, default=1)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
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
    args.video_dir = (args.video_dir or (VIDEO_DIR / args.out.name)).resolve()
    args.video_methods = parse_video_methods(args.video_methods)
    args.image_formats = parse_formats(args.image_formats)
    args.build_dir = args.build_dir.resolve()
    if args.make_videos:
        args.write_particles = True

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

    if args.extract_only:
        return extract_from_raw(specs, args)

    binary = build_binary(args.skip_build, args.build_jobs, args.build_dir)
    defer_extraction = args.defer_extraction
    if np is None and not defer_extraction:
        defer_extraction = True
        print("[defer] numpy is not available; keeping raw output for later extraction")
    if defer_extraction and args.make_videos:
        print("[defer] video encoding is skipped while extraction is deferred")
    video_options = VideoOptions(
        enabled=args.make_videos,
        methods=args.video_methods,
        root=args.video_dir,
        fps=max(1, args.video_fps),
        sample=max(1, args.video_sample),
        width=max(2, args.video_width),
        height=max(2, args.video_height),
        workers=max(1, args.video_workers),
        cmap=args.video_cmap,
        mode=args.video_mode,
    )
    summary_rows = [] if args.force else read_csv(args.out / "summary.csv")
    ke_rows = [] if args.force else read_csv(args.out / "kinetic_energy.csv")
    div_rows = [] if args.force else read_csv(args.out / "max_div.csv")
    vorticity_rows = [] if args.force else read_csv(args.out / "vorticity.csv")
    completed = {
        row["run"]
        for row in summary_rows
        if row.get("run") and row.get("status") == "ok"
        and not missing_required_metrics(
            row["run"],
            args.analysis,
            ke_rows,
            vorticity_rows,
        )
    }

    failures = 0
    for spec in specs:
        if spec.name in completed:
            print(f"[skip] {spec.name}: already present in summary.csv")
            continue
        missing_metrics = missing_required_metrics(
            spec.name,
            args.analysis,
            ke_rows,
            vorticity_rows,
        )
        if missing_metrics and any(
            row.get("run") == spec.name and row.get("status") == "ok"
            for row in summary_rows
        ):
            print(
                f"[rerun] {spec.name}: summary is ok but missing "
                f"{', '.join(missing_metrics)} rows"
            )
        summary_rows = drop_run(summary_rows, spec.name)
        ke_rows = drop_run(ke_rows, spec.name)
        div_rows = drop_run(div_rows, spec.name)
        vorticity_rows = drop_run(vorticity_rows, spec.name)
        summary, ke, div, vort = run_one(
            spec,
            binary,
            args.keep_raw,
            defer_extraction,
            args.analysis,
            args.no_run_logs,
            not args.no_plots,
            args.img_dir,
            args.image_formats,
            video_options,
        )
        summary_rows.append(summary)
        ke_rows.extend(ke)
        div_rows.extend(div)
        vorticity_rows.extend(vort)
        if summary["status"] == "failed":
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
