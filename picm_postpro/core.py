"""Shared utilities for PICM post-processing scripts."""

import csv
import math
import os
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

from picm_postpro.paths import PICM_ROOT


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    """Atomic write: write to .tmp then rename."""
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fieldnames = list(rows[0].keys())
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> List[dict]:
    """Read a CSV and return list of dicts. Returns [] if file missing/empty."""
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def drop_run(rows: List[dict], run: str) -> List[dict]:
    """Remove all rows where rows['run'] == run."""
    return [r for r in rows if r.get("run") != run]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Return float(value) unless value is empty/None/non-finite."""
    if value is None or value == "":
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def scheduler_threads() -> int:
    """Detect available parallelism from SLURM or OS."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
        val = os.environ.get(var)
        if val:
            try:
                n = int(val)
                if n > 0:
                    return n
            except ValueError:
                pass
    return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Binary runner
# ---------------------------------------------------------------------------

def run_binary(command: List[str], env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run command with stdout/stderr captured, cwd=PICM_ROOT."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PICM_ROOT),
        env=merged_env,
    )


def build_binary(
    build_dir: Path,
    jobs: int,
    skip: bool = False,
    build_type: str = "Release",
) -> Path:
    """cmake configure + build; returns path to PIC binary."""
    binary = build_dir / "bin" / "PIC"
    if skip and binary.exists():
        print(f"[build] skipping build, using {binary}")
        return binary
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_args = [
        "cmake",
        "-S", str(PICM_ROOT),
        "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DUSE_GPU=OFF",
        "-DUSE_PARALLEL=ON",
    ]
    print(f"[build] configuring: {' '.join(cmake_args)}")
    result = subprocess.run(cmake_args, cwd=str(PICM_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"cmake configure failed (exit {result.returncode})")
    build_args = ["cmake", "--build", str(build_dir), f"-j{jobs}"]
    print(f"[build] building: {' '.join(build_args)}")
    result = subprocess.run(build_args, cwd=str(PICM_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"cmake build failed (exit {result.returncode})")
    if not binary.exists():
        raise FileNotFoundError(f"build succeeded but binary not found at {binary}")
    return binary


# ---------------------------------------------------------------------------
# PVD parsing
# ---------------------------------------------------------------------------

def parse_pvd(pvd_path: Path) -> List[Path]:
    """Parse a .pvd file and return list of referenced VTP/VTI paths."""
    tree = ET.parse(pvd_path)
    base = pvd_path.parent
    return [
        base / dataset.get("file")
        for dataset in tree.getroot().iter("DataSet")
        if dataset.get("file")
    ]


# ---------------------------------------------------------------------------
# VTK binary appended data helpers
# ---------------------------------------------------------------------------

def _xml_stub(raw: bytes) -> bytes:
    """Strip binary AppendedData for XML parsing."""
    start = raw.find(b"<AppendedData")
    if start == -1:
        return raw
    underscore = raw.find(b"_", start)
    if underscore == -1:
        return raw
    return raw[:underscore] + b"\n  </AppendedData>\n</VTKFile>"


def _appended_start(raw: bytes) -> int:
    """Find the offset where appended binary data begins."""
    start = raw.find(b"<AppendedData")
    if start == -1:
        raise ValueError("no AppendedData block found")
    underscore = raw.find(b"_", start)
    if underscore == -1:
        raise ValueError("AppendedData block has no '_' marker")
    return underscore + 1


def _decode_array(
    raw: bytes,
    data_start: int,
    offset: int,
    compressed: bool,
    dtype,
):
    """Decompress zlib or read plain binary array from VTK appended data."""
    chunk = raw[data_start + offset:]
    if compressed:
        num_blocks, raw_size, last_size, comp_size = struct.unpack_from("<IIII", chunk, 0)
        payload = zlib.decompress(chunk[16: 16 + comp_size])
        return np.frombuffer(payload, dtype=dtype).copy()
    (raw_size,) = struct.unpack_from("<I", chunk, 0)
    return np.frombuffer(chunk[4: 4 + raw_size], dtype=dtype).copy()


# ---------------------------------------------------------------------------
# VTI reader
# ---------------------------------------------------------------------------

def read_vti_field(vti_path: Path, field_name: str):
    """Read a scalar field from a VTI file. Returns 2D array shape (ny, nx)."""
    raw = vti_path.read_bytes()
    root = ET.fromstring(_xml_stub(raw))
    compressed = "compressor" in root.attrib

    # Get grid dimensions from ImageData extent
    image_data = root.find(".//ImageData")
    if image_data is None:
        image_data = root.find("ImageData")
    if image_data is None:
        raise ValueError(f"no ImageData element in {vti_path}")

    extent = image_data.get("WholeExtent", "").split()
    if len(extent) < 6:
        piece = root.find(".//Piece")
        extent = (piece.get("Extent", "") if piece is not None else "").split()
    x0, x1, y0, y1 = int(extent[0]), int(extent[1]), int(extent[2]), int(extent[3])
    nx = x1 - x0
    ny = y1 - y0

    data_start = _appended_start(raw)

    # Search CellData and PointData for the field
    for da in root.iter("DataArray"):
        if da.get("Name") != field_name:
            continue
        dtype_str = da.get("type", "Float64")
        dtype = np.float32 if dtype_str == "Float32" else np.float64
        offset = int(da.get("offset", 0))
        arr = _decode_array(raw, data_start, offset, compressed, dtype)
        return arr[: nx * ny].reshape(ny, nx)

    raise KeyError(f"field '{field_name}' not found in {vti_path}")


# ---------------------------------------------------------------------------
# VTP reader
# ---------------------------------------------------------------------------

def read_vtp_point_count(vtp_path: Path) -> int:
    """Fast read: return NumberOfPoints from the Piece element."""
    raw = vtp_path.read_bytes()
    root = ET.fromstring(_xml_stub(raw))
    piece = root.find(".//Piece")
    if piece is None:
        return 0
    return int(piece.get("NumberOfPoints", 0))


def read_vtp_point_array(vtp_path: Path, name: str):
    """Read a per-point DataArray from a VTP file."""
    raw = vtp_path.read_bytes()
    root = ET.fromstring(_xml_stub(raw))
    compressed = "compressor" in root.attrib

    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"no Piece element in {vtp_path}")
    point_count = int(piece.get("NumberOfPoints", 0))
    if point_count == 0:
        return np.empty(0, np.float32)

    data_start = _appended_start(raw)

    for da in root.iter("DataArray"):
        if da.get("Name") != name:
            continue
        dtype_str = da.get("type", "Float32")
        dtype = np.float32 if dtype_str == "Float32" else np.float64
        offset = int(da.get("offset", 0))
        arr = _decode_array(raw, data_start, offset, compressed, dtype)
        n_components = int(da.get("NumberOfComponents", 1))
        total = point_count * n_components
        return arr[:total].reshape(point_count, n_components) if n_components > 1 else arr[:point_count]

    raise KeyError(f"array '{name}' not found in {vtp_path}")
