#!/usr/bin/env python3
"""Convert a PICM particle PVD/VTP sequence to an MP4 movie."""


import argparse
import os
import shutil
import struct
import subprocess
import tempfile
import zlib
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/picm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc="", **_kwargs):
            self.iterable = iterable
            self.total = total if total is not None else len(iterable or [])
            self.count = 0
            self.desc = desc

        def __iter__(self):
            for item in self.iterable:
                self.update(1)
                yield item

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            print()

        def update(self, n=1):
            self.count += n
            print(f"\r{self.desc} {self.count}/{self.total}", end="", flush=True)

        def set_postfix(self, **_kwargs):
            return None

        def close(self):
            print()

        @staticmethod
        def write(message):
            print(message)


def _cmap(name: str):
    return matplotlib.colormaps[name]


def _background_palette(name: str):
    if name == "black":
        return (0, 0, 0), "black", "white"
    return (255, 255, 255), "white", "black"


def parse_pvd(pvd_path: Path) -> List[Path]:
    tree = ET.parse(pvd_path)
    base = pvd_path.parent
    return [
        base / dataset.get("file")
        for dataset in tree.getroot().iter("DataSet")
        if dataset.get("file")
    ]


@dataclass(frozen=True)
class _EncoderCandidate:
    label: str
    codec: str
    global_args: Tuple[str, ...]
    output_args: Tuple[str, ...]


def _ffmpeg_binary() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    return ffmpeg


@lru_cache(maxsize=1)
def _ffmpeg_encoders() -> frozenset[str]:
    result = subprocess.run(
        [_ffmpeg_binary(), "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not list ffmpeg encoders")
    names = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


def _vaapi_global_args() -> Tuple[str, ...]:
    device = os.environ.get("PICM_FFMPEG_VAAPI_DEVICE")
    if device:
        return ("-vaapi_device", device)
    dri = Path("/dev/dri")
    if dri.is_dir():
        render_nodes = sorted(dri.glob("renderD*"))
        if render_nodes:
            return ("-vaapi_device", str(render_nodes[0]))
    return ()


def _software_candidates(crf: int, preset: str) -> List[_EncoderCandidate]:
    return [
        _EncoderCandidate(
            "software AV1 (SVT)",
            "libsvtav1",
            (),
            (
                "-vcodec", "libsvtav1",
                "-crf", str(crf),
                "-preset", "6",   # SVT-AV1 preset: 0=slowest 13=fastest; 6 = good balance
                "-pix_fmt", "yuv420p",
                "-svtav1-params", "tune=0",  # tune=0: PSNR/visual quality
            ),
        ),
        _EncoderCandidate(
            "software AV1 (libaom)",
            "libaom-av1",
            (),
            (
                "-vcodec", "libaom-av1",
                "-crf", str(crf),
                "-b:v", "0",
                "-cpu-used", "4",   # 0=best quality 8=fastest
                "-row-mt", "1",
                "-pix_fmt", "yuv420p",
            ),
        ),
        _EncoderCandidate(
            "software HEVC",
            "libx265",
            (),
            (
                "-vcodec", "libx265",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1",
                "-x265-params", "log-level=error",
            ),
        ),
        _EncoderCandidate(
            "software H.264",
            "libx264",
            (),
            (
                "-vcodec", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
            ),
        ),
    ]


def _hardware_candidates(quality: int) -> List[_EncoderCandidate]:
    candidates = [
        _EncoderCandidate(
            "hardware HEVC",
            "hevc_nvenc",
            (),
            (
                "-vcodec",
                "hevc_nvenc",
                "-cq",
                str(quality),
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-tag:v",
                "hvc1",
            ),
        ),
        _EncoderCandidate(
            "hardware HEVC",
            "hevc_qsv",
            (),
            (
                "-vcodec",
                "hevc_qsv",
                "-global_quality",
                str(quality),
                "-tag:v",
                "hvc1",
            ),
        ),
        _EncoderCandidate(
            "hardware HEVC",
            "hevc_videotoolbox",
            (),
            (
                "-vcodec",
                "hevc_videotoolbox",
                "-q:v",
                str(quality),
                "-tag:v",
                "hvc1",
            ),
        ),
        _EncoderCandidate(
            "hardware H.264",
            "h264_nvenc",
            (),
            (
                "-vcodec",
                "h264_nvenc",
                "-cq",
                str(quality),
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
            ),
        ),
        _EncoderCandidate(
            "hardware H.264",
            "h264_qsv",
            (),
            (
                "-vcodec",
                "h264_qsv",
                "-global_quality",
                str(quality),
            ),
        ),
        _EncoderCandidate(
            "hardware H.264",
            "h264_videotoolbox",
            (),
            (
                "-vcodec",
                "h264_videotoolbox",
                "-q:v",
                str(quality),
            ),
        ),
    ]
    vaapi_args = _vaapi_global_args()
    if vaapi_args:
        candidates[2:2] = [
            _EncoderCandidate(
                "hardware HEVC",
                "hevc_vaapi",
                vaapi_args,
                (
                    "-vf",
                    "format=nv12,hwupload",
                    "-vcodec",
                    "hevc_vaapi",
                    "-qp",
                    str(quality),
                    "-tag:v",
                    "hvc1",
                ),
            ),
            _EncoderCandidate(
                "hardware H.264",
                "h264_vaapi",
                vaapi_args,
                (
                    "-vf",
                    "format=nv12,hwupload",
                    "-vcodec",
                    "h264_vaapi",
                    "-qp",
                    str(quality),
                ),
            ),
        ]
    return candidates


def _normalise_encoder_name(encoder: str) -> str:
    return encoder.strip().lower().replace("_", "-")


def _encoder_candidates(encoder: str, quality: int, preset: str) -> List[_EncoderCandidate]:
    encoder = _normalise_encoder_name(encoder)
    hardware = _hardware_candidates(quality)
    software = _software_candidates(quality, preset)
    by_codec = {candidate.codec.replace("_", "-"): candidate for candidate in hardware + software}

    if encoder == "auto":
        return hardware + software
    if encoder in {"hardware", "hw"}:
        return hardware
    if encoder in {"software", "cpu"}:
        return software
    if encoder in {"av1"}:
        return [c for c in software if "av1" in c.codec.lower()]
    if encoder in {"hevc", "h265", "x265"}:
        return [c for c in software if c.codec == "libx265"]
    if encoder in {"h264", "x264"}:
        return [c for c in software if c.codec == "libx264"]
    if encoder in by_codec:
        return [by_codec[encoder]]
    raise ValueError(
        "unknown encoder "
        f"'{encoder}', expected auto, av1, hardware, software, libx265, libx264"
    )


def _ffmpeg_command(
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    candidate: _EncoderCandidate,
    *,
    extra_output_args: Tuple[str, ...] = (),
) -> List[str]:
    return [
        _ffmpeg_binary(),
        "-y",
        *candidate.global_args,
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-framerate",
        str(fps),
        "-loglevel",
        "error",
        "-i",
        "pipe:",
        *candidate.output_args,
        *extra_output_args,
        "-movflags",
        "+faststart",
        str(out_path),
    ]


@lru_cache(maxsize=None)
def _probe_encoder(candidate: _EncoderCandidate) -> bool:
    if candidate.codec not in _ffmpeg_encoders():
        return False
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        probe_path = Path(handle.name)
    try:
        result = subprocess.run(
            _ffmpeg_command(
                probe_path,
                16,
                16,
                1,
                candidate,
                extra_output_args=("-frames:v", "1"),
            ),
            input=bytes(16 * 16 * 3),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.returncode == 0 and probe_path.stat().st_size > 0
    finally:
        try:
            probe_path.unlink()
        except FileNotFoundError:
            pass


def _select_encoder(encoder: str, quality: int, preset: str) -> _EncoderCandidate:
    candidates = _encoder_candidates(encoder, quality, preset)
    tried = []
    for candidate in candidates:
        tried.append(candidate.codec)
        if _probe_encoder(candidate):
            print(f"[video] ffmpeg encoder: {candidate.label} ({candidate.codec})")
            return candidate
    raise RuntimeError(f"no working ffmpeg encoder found; tried: {', '.join(tried)}")


def _extract_xml(raw: bytes) -> bytes:
    start = raw.find(b"<AppendedData")
    if start == -1:
        return raw
    underscore = raw.find(b"_", start)
    if underscore == -1:
        return raw
    return raw[:underscore] + b"\n  </AppendedData>\n</VTKFile>"


def _bin_start(raw: bytes) -> int:
    start = raw.find(b"<AppendedData")
    if start == -1:
        raise ValueError("VTP has no AppendedData block")
    underscore = raw.find(b"_", start)
    if underscore == -1:
        raise ValueError("VTP AppendedData block has no '_' marker")
    return underscore + 1


def _decode(raw: bytes, bstart: int, offset: int, compressed: bool, dtype):
    chunk = raw[bstart + offset :]
    try:
        if compressed:
            _num_blocks, _raw_size, _last_size, comp_size = struct.unpack_from(
                "<IIII", chunk, 0
            )
            payload = zlib.decompress(chunk[16 : 16 + comp_size])
            return np.frombuffer(payload, dtype=dtype)
        (raw_size,) = struct.unpack_from("<I", chunk, 0)
        return np.frombuffer(chunk[4 : 4 + raw_size], dtype=dtype)
    except Exception as exc:  # pragma: no cover - corrupt file handling
        tqdm.write(f"  [warn] decode offset {offset}: {exc}")
        return None


def _load_vtp(path: Path):
    raw = path.read_bytes()
    try:
        root = ET.fromstring(_extract_xml(raw))
    except ET.ParseError as exc:
        tqdm.write(f"  [warn] {path.name}: {exc}")
        return None, None, None

    compressed = "compressor" in root.attrib
    piece = root.find(".//Piece")
    if piece is None:
        return None, None, None
    point_count = int(piece.get("NumberOfPoints", 0))
    if point_count == 0:
        empty = np.empty(0, np.float32)
        return empty, empty, empty

    data_start = _bin_start(raw)
    points_da = root.find(".//Points/DataArray")
    if points_da is None:
        return None, None, None
    points_dtype = np.float32 if points_da.get("type", "Float32") == "Float32" else np.float64
    points = _decode(raw, data_start, int(points_da.get("offset", 0)), compressed, points_dtype)
    if points is None or points.size < point_count * 3:
        return None, None, None
    points = points[: point_count * 3].reshape(point_count, 3)

    speed = None
    for data_array in root.iter("DataArray"):
        if data_array.get("Name") != "normVelocity":
            continue
        speed_dtype = (
            np.float32 if data_array.get("type", "Float32") == "Float32" else np.float64
        )
        speed = _decode(
            raw,
            data_start,
            int(data_array.get("offset", 0)),
            compressed,
            speed_dtype,
        )
        if speed is not None:
            speed = speed[:point_count].astype(np.float32)
        break
    if speed is None:
        speed = np.zeros(point_count, np.float32)

    return (
        points[:, 0].astype(np.float32),
        points[:, 1].astype(np.float32),
        speed,
    )


def _expand_limits(limits: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = limits
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if abs(hi - lo) < 1e-30:
        center = 0.5 * (lo + hi)
        return center - 0.5, center + 0.5
    return lo, hi


def _fit_limits(xlim, ylim, width: int, height: int, margin: float):
    xmin, xmax = _expand_limits((float(xlim[0]), float(xlim[1])))
    ymin, ymax = _expand_limits((float(ylim[0]), float(ylim[1])))
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    xspan = (xmax - xmin) * (1.0 + 2.0 * max(0.0, margin))
    yspan = (ymax - ymin) * (1.0 + 2.0 * max(0.0, margin))
    canvas_aspect = width / max(1, height)
    data_aspect = xspan / max(yspan, 1e-30)
    if data_aspect > canvas_aspect:
        yspan = xspan / canvas_aspect
    else:
        xspan = yspan * canvas_aspect
    return (
        (xmid - 0.5 * xspan, xmid + 0.5 * xspan),
        (ymid - 0.5 * yspan, ymid + 0.5 * yspan),
    )


def _scan(paths: Iterable[Path], n_scan: int = 12):
    paths = list(paths)
    if not paths:
        return (0.0, 1.0), (0.0, 1.0), 0.0, 1.0
    indices = sorted(
        set([0] + list(range(0, len(paths), max(1, len(paths) // n_scan))) + [len(paths) - 1])
    )
    xs, ys, speeds = [], [], []
    with tqdm(indices, desc="  scanning", unit="frame", leave=False) as progress:
        for index in progress:
            x, y, speed = _load_vtp(paths[index])
            if x is not None and len(x) > 0:
                xs.append(x)
                ys.append(y)
                speeds.append(speed)
    if not xs:
        return (0.0, 1.0), (0.0, 1.0), 0.0, 1.0
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    speed_all = np.concatenate(speeds)
    finite_speed = speed_all[np.isfinite(speed_all)]
    if finite_speed.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(finite_speed))
        vmax = float(np.nanpercentile(finite_speed, 99.5))
        if abs(vmax - vmin) < 1e-30:
            vmax = float(np.nanmax(finite_speed))
        if abs(vmax - vmin) < 1e-30:
            vmax = vmin + 1.0
    return (
        (float(np.nanmin(x_all)), float(np.nanmax(x_all))),
        (float(np.nanmin(y_all)), float(np.nanmax(y_all))),
        vmin,
        vmax,
    )


def _rasterise(
    x,
    y,
    speed,
    width,
    height,
    xlim,
    ylim,
    cmap_name,
    vmin,
    vmax,
    mode,
    background_rgb,
):
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = np.asarray(background_rgb, dtype=np.uint8)
    if x is None or len(x) == 0:
        return image.tobytes()

    px = ((x - xlim[0]) / (xlim[1] - xlim[0]) * (width - 1)).astype(np.int32)
    py = ((y - ylim[0]) / (ylim[1] - ylim[0]) * (height - 1)).astype(np.int32)
    py = height - 1 - py
    mask = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[mask]
    py = py[mask]
    speed = speed[mask]
    if len(px) == 0:
        return image.tobytes()

    xbins = np.arange(width + 1)
    ybins = np.arange(height + 1)
    counts, _, _ = np.histogram2d(px, py, bins=[xbins, ybins])
    counts = counts.T

    colormap = _cmap(cmap_name)
    if mode == "density":
        data = np.log1p(counts)
        max_value = float(np.nanmax(data)) if data.size else 0.0
        data = data / (max_value + 1e-30)
        rgba = colormap(data)
    else:
        weights, _, _ = np.histogram2d(px, py, bins=[xbins, ybins], weights=speed)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_speed = np.where(counts > 0, weights.T / counts, 0.0)
        rgba = colormap(mcolors.Normalize(vmin=vmin, vmax=vmax)(mean_speed))

    occupied = counts > 0
    image[occupied] = (rgba[..., :3][occupied] * 255).astype(np.uint8)
    return image.tobytes()


def _worker(args):
    path, width, height, xlim, ylim, cmap_name, global_vmin, global_vmax, mode, background_rgb = args
    x, y, speed = _load_vtp(Path(path))

    # Per-frame speed range (dynamic colorbar)
    if speed is not None and len(speed) > 0:
        finite = speed[np.isfinite(speed)]
        if finite.size > 0:
            frame_vmin = float(np.min(finite))
            frame_vmax = float(np.percentile(finite, 99.5))
            if abs(frame_vmax - frame_vmin) < 1e-30:
                frame_vmax = frame_vmin + 1.0
        else:
            frame_vmin, frame_vmax = global_vmin, global_vmax
    else:
        frame_vmin, frame_vmax = global_vmin, global_vmax

    rgb_bytes = _rasterise(
        x, y, speed, width, height, xlim, ylim,
        cmap_name, frame_vmin, frame_vmax, mode, background_rgb,
    )
    return rgb_bytes, frame_vmin, frame_vmax


_cb_fig_cache: dict = {}  # key → (fig, scalar, cb)


def _colorbar_left(
    height: int,
    cb_width: int,
    vmin: float,
    vmax: float,
    cmap_name: str,
    label: str,
    title: str,
    background_color: str,
    foreground_color: str,
    background_rgb,
    dpi: int = 100,
) -> np.ndarray:
    """Render a vertical colorbar strip (height × cb_width, 3) per-frame, with caching."""
    key = (height, cb_width, cmap_name, dpi, background_color, foreground_color)
    if key not in _cb_fig_cache:
        fig, ax = plt.subplots(figsize=(cb_width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor(background_color)
        ax.set_visible(False)
        scalar = plt.cm.ScalarMappable(
            cmap=_cmap(cmap_name), norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
        )
        scalar.set_array([])
        cb = fig.colorbar(
            scalar, ax=ax, orientation="vertical",
            fraction=0.55, pad=0.04,
            aspect=max(6, height // max(1, cb_width) * 5),
        )
        cb.set_label(label, color=foreground_color, fontsize=6, rotation=90, labelpad=3)
        cb.ax.tick_params(color=foreground_color, labelsize=6, labelcolor=foreground_color)
        cb.outline.set_edgecolor(foreground_color)
        fig.tight_layout(pad=0.2)
        _cb_fig_cache[key] = (fig, scalar, cb)

    fig, scalar, cb = _cb_fig_cache[key]
    scalar.norm.vmin = vmin
    scalar.norm.vmax = vmax
    cb.update_normal(scalar)
    # small title at top
    for txt in fig.texts:
        txt.remove()
    if title:
        fig.text(0.5, 0.995, title, ha="center", va="top",
                 color=foreground_color, fontsize=5, transform=fig.transFigure,
                 clip_on=True)
    fig.canvas.draw()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    panel = rgba[..., :3].copy()
    # Resize to exact (height, cb_width) if needed
    if panel.shape[:2] != (height, cb_width):
        try:
            from PIL import Image
            panel = np.array(Image.fromarray(panel).resize((cb_width, height), Image.LANCZOS))
        except ImportError:
            out = np.full((height, cb_width, 3), np.asarray(background_rgb, np.uint8), np.uint8)
            h = min(height, panel.shape[0])
            w = min(cb_width, panel.shape[1])
            out[:h, :w] = panel[:h, :w]
            panel = out
    return panel


def _open_ffmpeg(
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    encoder: str,
    quality: int,
    preset: str,
):
    candidate = _select_encoder(encoder, quality, preset)
    return subprocess.Popen(
        _ffmpeg_command(out_path, width, height, fps, candidate),
        stdin=subprocess.PIPE,
    )


def build_mp4(
    vtp_paths: Iterable[Path],
    out_path: Path,
    *,
    fps: int = 30,
    cmap_name: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    width: int = 1280,
    height: int = 720,
    dpi: int = 120,
    title: str = "PICM particles",
    mode: str = "speed",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    n_workers: Optional[int] = None,
    prefetch: Optional[int] = None,
    margin: float = 0.08,
    encoder: str = "av1",
    crf: int = 28,
    preset: str = "veryslow",
    background: str = "white",
    colorbar_width: int = 80,
) -> None:
    paths = list(vtp_paths)
    if not paths:
        raise ValueError("no VTP frames to encode")
    width += width % 2
    height += height % 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    background_rgb, background_color, foreground_color = _background_palette(background)

    print("[video] scanning particle frames for spatial extent")
    scanned_xlim, scanned_ylim, scanned_vmin, scanned_vmax = _scan(paths)
    xlim = tuple(xlim) if xlim is not None else scanned_xlim
    ylim = tuple(ylim) if ylim is not None else scanned_ylim
    # global vmin/vmax used as fallback when a frame has no particles
    global_vmin = scanned_vmin if vmin is None else float(vmin)
    global_vmax = scanned_vmax if vmax is None else float(vmax)
    if abs(global_vmax - global_vmin) < 1e-30:
        global_vmax = global_vmin + 1.0

    # Layout: colorbar on the left, particles on the right
    cb_width = max(60, colorbar_width)
    cb_width += cb_width % 2
    frame_width = max(2, width - cb_width)
    frame_width += frame_width % 2
    total_width = cb_width + frame_width

    xlim, ylim = _fit_limits(xlim, ylim, frame_width, height, margin)
    label = "Mean speed |v|" if mode == "speed" else "log(1 + count)"

    n_workers = max(1, int(n_workers or os.cpu_count() or 1))
    prefetch = max(1, int(prefetch or n_workers * 2))
    worker_args = [
        (
            str(path),
            frame_width,
            height,
            xlim,
            ylim,
            cmap_name,
            global_vmin,
            global_vmax,
            mode,
            background_rgb,
        )
        for path in paths
    ]
    process = _open_ffmpeg(out_path, total_width, height, fps, encoder, crf, preset)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is not available")

    def _assemble(rgb_bytes, frame_vmin, frame_vmax):
        """Combine left colorbar + right particle frame into one row."""
        cb = _colorbar_left(
            height, cb_width, frame_vmin, frame_vmax, cmap_name,
            label, title, background_color, foreground_color, background_rgb, dpi,
        )
        particles = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, frame_width, 3)
        return np.concatenate([cb, particles], axis=1)

    outer = tqdm(total=len(worker_args), desc="  encoding", unit="frame")
    if n_workers == 1:
        try:
            for args in worker_args:
                rgb_bytes, fvmin, fvmax = _worker(args)
                process.stdin.write(_assemble(rgb_bytes, fvmin, fvmax).tobytes())
                outer.update(1)
        finally:
            outer.close()
            process.stdin.close()
            process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {process.returncode}")
        print(f"[video] wrote {out_path}")
        return

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pending: deque[Tuple[int, Future]] = deque()

        def submit_next(index: int) -> None:
            if index < len(worker_args):
                pending.append((index, pool.submit(_worker, worker_args[index])))

        for index in range(min(prefetch, len(worker_args))):
            submit_next(index)
        next_submit = prefetch

        for _index in range(len(worker_args)):
            _, future = pending.popleft()
            rgb_bytes, fvmin, fvmax = future.result()
            submit_next(next_submit)
            next_submit += 1
            process.stdin.write(_assemble(rgb_bytes, fvmin, fvmax).tobytes())
            outer.set_postfix(queue=len(pending))
            outer.update(1)

    outer.close()
    process.stdin.close()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {process.returncode}")
    print(f"[video] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pvd", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--title")
    parser.add_argument("--mode", choices=("speed", "density"), default="speed")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--prefetch", type=int)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"))
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"))
    parser.add_argument(
        "--encoder",
        default="av1",
        help="ffmpeg encoder: av1 (default), auto, hardware, software, libx265, libx264",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="constant-quality value; lower = higher quality. AV1: 28≈good, 18≈near-lossless",
    )
    parser.add_argument("--preset", default="veryslow")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--colorbar-width", type=int, default=80,
                        help="width in pixels of the left-side colorbar strip")
    args = parser.parse_args()

    if not args.pvd.exists():
        raise FileNotFoundError(args.pvd)
    paths = [path for path in parse_pvd(args.pvd) if path.exists()]
    if args.sample > 1:
        paths = paths[:: args.sample]
    if not paths:
        raise RuntimeError("no existing VTP frames found")
    out_path = args.out if args.out else args.pvd.with_suffix(".mp4")
    build_mp4(
        paths,
        out_path,
        fps=args.fps,
        cmap_name=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        title=args.title or args.pvd.stem,
        mode=args.mode,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        n_workers=args.workers,
        prefetch=args.prefetch,
        encoder=args.encoder,
        crf=args.crf,
        preset=args.preset,
        background=args.background,
        colorbar_width=args.colorbar_width,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
