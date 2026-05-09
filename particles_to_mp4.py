#!/usr/bin/env python3
"""Convert a PICM particle PVD/VTP sequence to an MP4 movie."""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import zlib
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

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


def parse_pvd(pvd_path: Path) -> list[Path]:
    tree = ET.parse(pvd_path)
    base = pvd_path.parent
    return [
        base / dataset.get("file")
        for dataset in tree.getroot().iter("DataSet")
        if dataset.get("file")
    ]


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


def _expand_limits(limits: tuple[float, float]) -> tuple[float, float]:
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


def _rasterise(x, y, speed, width, height, xlim, ylim, cmap_name, vmin, vmax, mode):
    image = np.zeros((height, width, 3), dtype=np.uint8)
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
    path, width, height, xlim, ylim, cmap_name, vmin, vmax, mode = args
    x, y, speed = _load_vtp(Path(path))
    return _rasterise(x, y, speed, width, height, xlim, ylim, cmap_name, vmin, vmax, mode)


def _make_panel(width, panel_height, dpi, title, cmap_name, vmin, vmax, label):
    fig, ax = plt.subplots(figsize=(width / dpi, panel_height / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_visible(False)
    scalar = plt.cm.ScalarMappable(cmap=_cmap(cmap_name), norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, orientation="horizontal", fraction=1.0, pad=0.0, aspect=40)
    colorbar.set_label(label, color="white", fontsize=8)
    colorbar.ax.xaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(colorbar.ax.xaxis.get_ticklabels(), color="white")
    colorbar.outline.set_edgecolor("white")
    fig.suptitle(title, color="white", fontsize=9, y=0.98)
    fig.canvas.draw()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    panel = rgba[..., :3].copy()
    plt.close(fig)
    if panel.shape[1] != width or panel.shape[0] != panel_height:
        try:
            from PIL import Image

            panel = np.array(Image.fromarray(panel).resize((width, panel_height), Image.LANCZOS))
        except ImportError:
            resized = np.zeros((panel_height, width, 3), np.uint8)
            h = min(panel_height, panel.shape[0])
            w = min(width, panel.shape[1])
            resized[:h, :w] = panel[:h, :w]
            panel = resized
    return panel


def _open_ffmpeg(out_path: Path, width: int, height: int, fps: int, crf: int, preset: str):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
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
            "-vcodec",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )


def build_mp4(
    vtp_paths: Iterable[Path],
    out_path: Path,
    *,
    fps: int = 30,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    width: int = 1280,
    height: int = 720,
    dpi: int = 120,
    title: str = "PICM particles",
    mode: str = "speed",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    n_workers: int | None = None,
    prefetch: int | None = None,
    margin: float = 0.08,
    crf: int = 20,
    preset: str = "fast",
) -> None:
    paths = list(vtp_paths)
    if not paths:
        raise ValueError("no VTP frames to encode")
    width += width % 2
    height += height % 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[video] scanning particle frames")
    scanned_xlim, scanned_ylim, scanned_vmin, scanned_vmax = _scan(paths)
    xlim = tuple(xlim) if xlim is not None else scanned_xlim
    ylim = tuple(ylim) if ylim is not None else scanned_ylim
    vmin = scanned_vmin if vmin is None else vmin
    vmax = scanned_vmax if vmax is None else vmax
    if abs(float(vmax) - float(vmin)) < 1e-30:
        vmax = float(vmin) + 1.0

    panel_height = max(60, int(width * 0.06))
    panel_height += panel_height % 2
    frame_height = max(2, height - panel_height)
    frame_height += frame_height % 2
    total_height = panel_height + frame_height
    xlim, ylim = _fit_limits(xlim, ylim, width, frame_height, margin)
    label = "Mean particle speed |v|" if mode == "speed" else "log(1 + particles per pixel)"
    panel = _make_panel(width, panel_height, dpi, title, cmap_name, float(vmin), float(vmax), label)
    template = np.concatenate(
        [np.zeros((frame_height, width, 3), np.uint8), panel],
        axis=0,
    )

    n_workers = max(1, int(n_workers or os.cpu_count() or 1))
    prefetch = max(1, int(prefetch or n_workers * 2))
    worker_args = [
        (str(path), width, frame_height, xlim, ylim, cmap_name, float(vmin), float(vmax), mode)
        for path in paths
    ]
    process = _open_ffmpeg(out_path, width, total_height, fps, crf, preset)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is not available")

    outer = tqdm(total=len(worker_args), desc="  encoding", unit="frame")
    if n_workers == 1:
        try:
            for args in worker_args:
                rgb_bytes = _worker(args)
                frame = template.copy()
                frame[:frame_height] = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                    frame_height, width, 3
                )
                process.stdin.write(frame.tobytes())
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
        pending: deque[tuple[int, Future]] = deque()

        def submit_next(index: int) -> None:
            if index < len(worker_args):
                pending.append((index, pool.submit(_worker, worker_args[index])))

        for index in range(min(prefetch, len(worker_args))):
            submit_next(index)
        next_submit = prefetch

        for _index in range(len(worker_args)):
            _, future = pending.popleft()
            rgb_bytes = future.result()
            submit_next(next_submit)
            next_submit += 1
            frame = template.copy()
            frame[:frame_height] = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                frame_height, width, 3
            )
            process.stdin.write(frame.tobytes())
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
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="fast")
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
        crf=args.crf,
        preset=args.preset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
