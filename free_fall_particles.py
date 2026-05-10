#!/usr/bin/env python3
"""Compare no-water particle free fall against v_th = v0 + g t."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from picm_postpro.paths import DATA_DIR, PICM_ROOT, default_img_dir, default_misc_dir
from picm_postpro.plots import parse_formats, save_figure

os.environ.setdefault("MPLCONFIGDIR", "/tmp/picm_matplotlib")

try:
    import numpy as np
except ImportError:  # pragma: no cover - required for this script
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - required for plotting
    plt = None


DEFAULT_OUT = DATA_DIR / "study_free_fall_particles"


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_CONFIG = first_existing_path(
    PICM_ROOT / "test" / "PIC" / "freeFall.json",
    PICM_ROOT / "test" / "PIC" / "extra" / "freeFall.json",
)


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def run_process(command: List[str], cwd: Path, env: Dict[str, str]):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def build_binary(skip_build: bool, build_dir: Path, build_jobs: int) -> Path:
    binary = build_dir / "bin" / "PIC"
    if skip_build:
        if not binary.exists():
            raise FileNotFoundError(f"missing binary: {binary}")
        return binary

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
    build_cmd = ["cmake", "--build", str(build_dir), f"-j{max(1, build_jobs)}"]
    subprocess.run(configure_cmd, cwd=PICM_ROOT, check=True)
    subprocess.run(build_cmd, cwd=PICM_ROOT, check=True)
    if not binary.exists():
        raise FileNotFoundError(f"build succeeded but binary is missing: {binary}")
    return binary


def parse_method_list(value: str) -> List[str]:
    valid = {"pic", "flip", "apic"}
    methods = []
    for item in value.split(","):
        method = item.strip().lower()
        if not method:
            continue
        if method not in valid:
            raise ValueError(f"unknown method '{method}', expected one of {sorted(valid)}")
        methods.append(method)
    if not methods:
        raise ValueError("empty method list")
    return list(dict.fromkeys(methods))


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_name(method: str, ppc: int, flip_coef_pic: Optional[float], threads: int) -> str:
    parts = ["free-fall-no-water", method, f"ppc{ppc}"]
    if method == "flip":
        parts.append(f"coefpic{slug_float(flip_coef_pic or 0.0)}")
    parts.append(f"t{threads}")
    return "_".join(parts)


def make_config(
    base_cfg: dict,
    method: str,
    ppc: int,
    flip_coef_pic: Optional[float],
    raw_dir: Path,
    args: argparse.Namespace,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["method"] = method
    cfg["ppcx"] = ppc
    cfg["ppcy"] = ppc
    cfg["folder"] = str(raw_dir)
    cfg["filename"] = "simulation"
    cfg["nt"] = int(args.nt)
    cfg["sampling_rate"] = int(args.sampling_rate)
    cfg["write_u"] = False
    cfg["write_v"] = False
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_smoke"] = False
    cfg["write_norm_velocity"] = False
    cfg["write_vorticity"] = False
    cfg["write_particles"] = True
    if method == "flip":
        cfg["coefPic"] = float(flip_coef_pic if flip_coef_pic is not None else 0.0)
    return cfg


def parse_pvd(pvd_path: Path) -> List[Path]:
    tree = ET.parse(pvd_path)
    base = pvd_path.parent
    return [
        base / dataset.get("file")
        for dataset in tree.getroot().iter("DataSet")
        if dataset.get("file")
    ]


def xml_without_appended_data(raw: bytes) -> bytes:
    start = raw.find(b"<AppendedData")
    if start == -1:
        return raw
    underscore = raw.find(b"_", start)
    if underscore == -1:
        return raw
    return raw[:underscore] + b"\n  </AppendedData>\n</VTKFile>"


def appended_data_start(raw: bytes) -> int:
    start = raw.find(b"<AppendedData")
    if start == -1:
        raise ValueError("VTP has no AppendedData block")
    underscore = raw.find(b"_", start)
    if underscore == -1:
        raise ValueError("VTP AppendedData block has no '_' marker")
    return underscore + 1


def decode_array(raw: bytes, data_start: int, offset: int, compressed: bool, dtype):
    chunk = raw[data_start + offset :]
    if compressed:
        _num_blocks, raw_size, _last_block_size, comp_size = struct.unpack_from(
            "<IIII", chunk, 0
        )
        payload = zlib.decompress(chunk[16 : 16 + comp_size])
        if len(payload) < raw_size:
            raise ValueError("decompressed VTP payload is too small")
        return np.frombuffer(payload, dtype=dtype)
    (raw_size,) = struct.unpack_from("<I", chunk, 0)
    return np.frombuffer(chunk[4 : 4 + raw_size], dtype=dtype)


def read_vtp_point_array(vtp_path: Path, name: str):
    if np is None:
        raise RuntimeError("numpy is required for particle statistics")
    raw = vtp_path.read_bytes()
    root = ET.fromstring(xml_without_appended_data(raw))
    compressed = "compressor" in root.attrib
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{vtp_path} has no Piece node")
    point_count = int(piece.get("NumberOfPoints", 0))
    data_array = next(
        (array for array in root.iter("DataArray") if array.get("Name") == name),
        None,
    )
    if data_array is None:
        names = [array.get("Name", "") for array in root.iter("DataArray")]
        raise KeyError(f"{name} not found in {vtp_path.name}; available={names}")
    dtype = np.float32 if data_array.get("type", "Float32") == "Float32" else np.float64
    values = decode_array(
        raw,
        appended_data_start(raw),
        int(data_array.get("offset", 0)),
        compressed,
        dtype,
    )
    if values.size < point_count:
        raise ValueError(f"{vtp_path.name}: expected {point_count} values, got {values.size}")
    return values[:point_count].astype(np.float64)


def particle_velocity_rows(
    method: str,
    ppc: int,
    flip_coef_pic: Optional[float],
    run: str,
    raw_dir: Path,
    cfg: dict,
    v0: Optional[float],
) -> List[dict]:
    pvd_path = raw_dir / "particles.pvd"
    if not pvd_path.exists():
        raise FileNotFoundError(f"missing particle PVD: {pvd_path}")
    paths = [path for path in parse_pvd(pvd_path) if path.exists()]
    if not paths:
        raise RuntimeError(f"{pvd_path} contains no readable particle frames")

    gravity = float(cfg.get("gravity", 0.0))
    gravity_sign = -1.0 if gravity >= 0.0 else 1.0
    g_abs = abs(gravity)
    dt = float(cfg.get("dt", 1.0))
    sampling_rate = int(cfg.get("sampling_rate", 1))
    frame_stats = []
    for sample, path in enumerate(paths):
        velocity_y = read_vtp_point_array(path, "velocityY")
        observed = gravity_sign * velocity_y
        finite = observed[np.isfinite(observed)]
        if finite.size == 0:
            continue
        step = sample * sampling_rate
        sim_time = step * dt
        frame_stats.append(
            {
                "sample": sample,
                "step": step,
                "time": sim_time,
                "v_mean": float(np.mean(finite)),
                "v_median": float(np.percentile(finite, 50.0)),
                "v_p05": float(np.percentile(finite, 5.0)),
                "v_p95": float(np.percentile(finite, 95.0)),
                "n_particles": int(finite.size),
            }
        )
    if not frame_stats:
        return []

    reference_v0 = float(v0) if v0 is not None else float(frame_stats[0]["v_median"])
    rows = []
    for stats in frame_stats:
        sim_time = float(stats["time"])
        rows.append(
            {
                "method": method,
                "ppc": ppc,
                "flip_coef_pic": "" if flip_coef_pic is None else flip_coef_pic,
                "run": run,
                "v0": reference_v0,
                "v_theory": reference_v0 + g_abs * sim_time,
                **stats,
            }
        )
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def method_color(method: str, flip_coef_pic) -> str:
    if method == "pic":
        return "#1f77b4"
    if method == "apic":
        return "#2ca02c"
    coef = "" if flip_coef_pic in ("", None) else f" {float(flip_coef_pic):g}"
    return "#ff7f0e" if not coef or abs(float(flip_coef_pic)) < 1e-14 else "#9467bd"


def method_label(method: str, flip_coef_pic) -> str:
    if method == "pic":
        return "PIC"
    if method == "apic":
        return "APIC"
    coef = None if flip_coef_pic in ("", None) else float(flip_coef_pic)
    if coef is None or abs(coef) < 1e-14:
        return "FLIP"
    return f"Mixed PIC-FLIP {coef:g}"


def plot_rows(rows: List[dict], img_dir: Path, image_formats: Iterable[str]) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot the free-fall study")
    if not rows:
        raise RuntimeError("no free-fall rows to plot")

    img_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    residual_ax = ax.twinx()

    theory_times = sorted({float(row["time"]) for row in rows})
    theory_by_time: Dict[float, List[float]] = {}
    for row in rows:
        theory_by_time.setdefault(float(row["time"]), []).append(float(row["v_theory"]))
    ax.plot(
        theory_times,
        [sum(theory_by_time[time]) / len(theory_by_time[time]) for time in theory_times],
        color="#111111",
        lw=2.0,
        linestyle="--",
        label="v_th = v0 + g t",
    )

    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row.get("flip_coef_pic", "")), []).append(row)

    for (method, coef), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: float(row["time"]))
        times = [float(row["time"]) for row in group]
        theory = [float(row["v_theory"]) for row in group]
        median = [float(row["v_median"]) for row in group]
        median_error = [value - ref for value, ref in zip(median, theory)]
        color = method_color(method, coef)
        label = method_label(method, coef)
        ax.plot(times, median, color=color, lw=1.8, label=f"{label} median")
        residual_ax.plot(
            times,
            median_error,
            color="#d62728",
            lw=1.4,
            linestyle="-.",
            label=f"{label} median - v_th",
        )

    ax.set_xlabel("Time t [s]")
    ax.set_ylabel("Downward particle velocity")
    residual_ax.set_ylabel("Observed - theory", color="#d62728")
    residual_ax.tick_params(axis="y", colors="#d62728")
    residual_ax.axhline(0.0, color="#d62728", lw=0.8, alpha=0.35)
    ax.set_title("No-water falling block: particle velocity versus theory")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    residual_handles, residual_labels = residual_ax.get_legend_handles_labels()
    ax.legend(handles + residual_handles, labels + residual_labels, ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    out_base = img_dir / "free_fall_no_water_particle_velocity"
    save_figure(fig, out_base, formats=image_formats)
    plt.close(fig)
    return out_base.with_suffix(".png")


def graph_has_data(image_path: Path) -> None:
    if np is None:
        return
    try:
        image = plt.imread(image_path)
    except Exception as exc:
        raise RuntimeError(f"could not read generated plot {image_path}: {exc}") from exc
    if image.size == 0:
        raise RuntimeError(f"generated plot is empty: {image_path}")
    rgb = image[..., :3] if image.ndim == 3 else image
    if float(np.nanstd(rgb)) < 1e-5:
        raise RuntimeError(f"generated plot appears blank: {image_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="pic", help="comma list: pic, flip, apic")
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--flip-coef-pic", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--nt", type=int, default=300, help="default is pre-impact")
    parser.add_argument("--sampling-rate", type=int, default=5)
    parser.add_argument(
        "--v0",
        type=float,
        default=None,
        help="initial downward velocity; defaults to the first observed median",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--misc-dir", type=Path)
    parser.add_argument("--img-dir", type=Path)
    parser.add_argument("--image-formats", default="png,svg,pdf,jpg")
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-local-report-release")
    parser.add_argument("--build-jobs", type=int, default=1)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if np is None:
        raise RuntimeError("numpy is required for the free-fall particle study")

    args.out = args.out.resolve()
    args.misc_dir = (args.misc_dir or default_misc_dir(args.out)).resolve()
    args.img_dir = (args.img_dir or default_img_dir(args.out)).resolve()
    image_formats = parse_formats(args.image_formats)
    if args.plot_only:
        rows = read_csv(args.out / "particle_velocity.csv")
        plot_path = plot_rows(rows, args.img_dir, image_formats)
        graph_has_data(plot_path)
        print(f"[plot] wrote {plot_path}")
        return 0

    methods = parse_method_list(args.methods)
    binary = build_binary(args.skip_build, args.build_dir.resolve(), args.build_jobs)

    base_cfg = load_config(args.config.resolve())
    rows: List[dict] = []
    summary_rows: List[dict] = []
    for method in methods:
        coef = args.flip_coef_pic if method == "flip" else None
        name = run_name(method, args.ppc, coef, args.threads)
        run_dir = args.misc_dir / "runs" / name
        raw_dir = run_dir / "raw"
        config_path = args.misc_dir / "configs" / f"{name}.json"
        raw_dir.mkdir(parents=True, exist_ok=True)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = make_config(base_cfg, method, args.ppc, coef, raw_dir, args)
        with config_path.open("w") as handle:
            json.dump(cfg, handle, indent=2)
            handle.write("\n")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(max(1, args.threads))
        env.setdefault("OMP_DYNAMIC", "false")
        command = [str(binary), str(config_path)]
        print(f"[run] {name}: OMP_NUM_THREADS={env['OMP_NUM_THREADS']} {' '.join(command)}")
        start = time.perf_counter()
        result = run_process(command, PICM_ROOT, env)
        wall_time = time.perf_counter() - start
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stdout.log").write_text(result.stdout)
        (run_dir / "stderr.log").write_text(result.stderr)
        if result.returncode != 0:
            summary_rows.append(
                {
                    "method": method,
                    "ppc": args.ppc,
                    "flip_coef_pic": "" if coef is None else coef,
                    "run": name,
                    "status": "failed",
                    "returncode": result.returncode,
                    "wall_time_s": wall_time,
                    "config": str(config_path),
                }
            )
            continue

        method_rows = particle_velocity_rows(
            method, args.ppc, coef, name, raw_dir, cfg, args.v0
        )
        rows.extend(method_rows)
        final_row = method_rows[-1] if method_rows else {}
        summary_rows.append(
            {
                "method": method,
                "ppc": args.ppc,
                "flip_coef_pic": "" if coef is None else coef,
                "run": name,
                "status": "ok" if method_rows else "failed",
                "returncode": result.returncode,
                "wall_time_s": wall_time,
                "frames": len(method_rows),
                "final_time": final_row.get("time", ""),
                "final_v_theory": final_row.get("v_theory", ""),
                "final_v_mean": final_row.get("v_mean", ""),
                "final_v_median": final_row.get("v_median", ""),
                "final_v_p05": final_row.get("v_p05", ""),
                "final_v_p95": final_row.get("v_p95", ""),
                "config": str(config_path),
            }
        )
        if not args.keep_raw and raw_dir.exists():
            shutil.rmtree(raw_dir)

    write_csv(args.out / "particle_velocity.csv", rows)
    write_csv(args.out / "summary.csv", summary_rows)
    plot_path = plot_rows(rows, args.img_dir, image_formats)
    graph_has_data(plot_path)
    print(f"[plot] wrote {plot_path}")
    print(f"[done] CSV files: {args.out}")
    return 0 if all(row.get("status") == "ok" for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
