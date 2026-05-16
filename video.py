#!/usr/bin/env python3
from typing import List
"""Discover test configs, run simulations with particles, generate MP4s."""

import argparse
import copy
import json
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from picm_postpro.paths import PICM_ROOT, POSTPRO_ROOT, VIDEO_DIR
from picm_postpro.core import build_binary, run_binary, scheduler_threads, write_csv, read_csv

try:
    from particles_to_mp4 import parse_pvd, build_mp4, build_mp4_vti
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from particles_to_mp4 import parse_pvd, build_mp4, build_mp4_vti


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def _discover_configs(roots: List[Path]) -> List[Path]:
    """Recursively find all *.json files in given roots."""
    found = []
    for root in roots:
        if root.is_dir():
            found.extend(sorted(root.rglob("*.json")))
        elif root.is_file() and root.suffix == ".json":
            found.append(root)
    return found


def _is_sim_config(path: Path) -> bool:
    """Check if a JSON file looks like a PICM simulation config."""
    try:
        with open(path) as fh:
            data = json.load(fh)
        return "method" in data or "nx" in data or "nt" in data
    except (json.JSONDecodeError, OSError):
        return False


def _config_name(config_path: Path) -> str:
    """Generate a short name from a config path, mirroring the section folder structure."""
    for root in (POSTPRO_ROOT, PICM_ROOT):
        try:
            rel = config_path.relative_to(root)
            parts = list(rel.parts)
            skip = {"test", "PIC", "FLIP", "APIC", "GPIC", "FLAPIC", "SL", "extra"}
            name_parts = [p for p in parts[:-1] if p not in skip]
            name_parts.append(config_path.stem)
            return "_".join(name_parts) if name_parts else config_path.stem
        except ValueError:
            continue
    return config_path.stem


# ---------------------------------------------------------------------------
# Async GPU encoding
# ---------------------------------------------------------------------------

_manifest_lock = threading.Lock()


def _create_archive(archive_path: Path, source_dir: Path) -> None:
    """Pack source_dir into a .tar.xz. Uses xz -T0 (all cores) when available."""
    files = sorted(f.name for f in source_dir.iterdir() if f.is_file())
    xz_bin = shutil.which("xz")
    if xz_bin:
        tar_cmd = ["tar", "-c", "-C", str(source_dir)] + files
        xz_cmd = [xz_bin, "-9e", f"-T{os.cpu_count() or 1}", "--block-size=8MiB", "-c"]
        with open(archive_path, "wb") as out_fh:
            tar_proc = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
            xz_proc = subprocess.Popen(xz_cmd, stdin=tar_proc.stdout, stdout=out_fh)
            tar_proc.stdout.close()
            xz_proc.wait()
            tar_proc.wait()
        if xz_proc.returncode != 0 or tar_proc.returncode != 0:
            raise RuntimeError(f"archive failed (tar={tar_proc.returncode}, xz={xz_proc.returncode})")
    else:
        with lzma.open(archive_path, "wb",
                       preset=9 | lzma.PRESET_EXTREME,
                       format=lzma.FORMAT_XZ) as fh:
            with tarfile.open(fileobj=fh, mode="w|") as tar:
                for f in files:
                    tar.add(source_dir / f, arcname=f)


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.xz archive into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(dest_dir)


_HTML_HEAD = """\
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MATH0471-3 - Multiphysics integrated computational project</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌊</text></svg>">
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main class="page">
    <header class="header">
      <h1>MATH0471-3</h1>
      <p>Multiphysics integrated computational project</p>
    </header>
    <section aria-label="Videos">
      <div class="video-grid">
"""

_HTML_TAIL = """\
      </div>
    </section>
  </main>
</body>
</html>
"""


def _render_index_html(out_dir: Path, rows: list) -> None:
    # Build a dict keyed by mp4 filename — deduplicates any stale CSV rows
    by_file: dict = {}
    for r in rows:
        if r.get("status") == "ok" and r.get("mp4_path"):
            by_file[Path(r["mp4_path"]).name] = r["name"]
    videos = sorted(by_file.items())
    print(f"[video] index.html updated ({len(videos)} video(s))")
    if videos:
        cards = "\n".join(
            f'        <article class="video-item">\n'
            f'          <h2>{name.replace("-", " ").replace("_", " ")}</h2>\n'
            f'          <video controls preload="metadata">\n'
            f'            <source src="./{mp4_name}" type="video/mp4">\n'
            f'          </video>\n'
            f'        </article>'
            for mp4_name, name in videos
        )
    else:
        cards = '        <p class="status">Aucune video disponible.</p>'
    tmp = out_dir / "index.html.tmp"
    tmp.write_text(_HTML_HEAD + cards + "\n" + _HTML_TAIL, encoding="utf-8")
    tmp.replace(out_dir / "index.html")


def _update_manifest(manifest_csv: Path, name: str, config_path: Path,
                     status: str, mp4_path: str) -> None:
    with _manifest_lock:
        rows = read_csv(manifest_csv)
        rows = [r for r in rows if r.get("name") != name]
        rows.append({"name": name, "config": str(config_path),
                     "status": status, "mp4_path": mp4_path})
        write_csv(manifest_csv, rows)
        _render_index_html(manifest_csv.parent, rows)


def _encode_job(job: dict, manifest_csv: Path) -> None:
    """Run in the GPU thread pool while the CPU is running the next simulation."""
    name = job["name"]
    mp4_path = Path(job["mp4_path"])
    run_dir = Path(job["run_dir"])
    use_vti = job["use_vti"]
    paths = [Path(p) for p in job["paths"]]
    config_path = Path(job["config_path"])
    kwargs = {
        "fps": job["fps"],
        "cmap_name": job["cmap"],
        "width": job["width"],
        "height": job["height"],
        "title": name,
        "n_workers": job["n_workers"],
        "encoder": job["encoder"],
        "crf": job["crf"],
    }

    print(f"[video] GPU encoding {name}...")
    try:
        if use_vti:
            build_mp4_vti(paths, mp4_path, **kwargs)
        else:
            build_mp4(paths, mp4_path, **kwargs, particle_radius=job["particle_radius"])
        status = "ok"
    except Exception as exc:
        print(f"[video] encoding failed for {name}: {exc}")
        status = "encode_failed"

    # Always delete extracted raw files after encoding — archive in run_dir is kept.
    raw_dir = run_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Full cleanup (archive included) only when --delete-raw.
    if job.get("delete_raw", False) and run_dir.exists():
        shutil.rmtree(run_dir)

    _update_manifest(manifest_csv, name, config_path,
                     status, str(mp4_path) if status == "ok" else "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--tests", default=None,
                        help="comma-separated JSON config paths or search roots; "
                             "default: auto-discover from PostPro/test/")
    parser.add_argument("--json", type=Path, default=None,
                        help="shorthand: run a single simulation config JSON")
    parser.add_argument("--out", type=Path, default=VIDEO_DIR)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4,
                        help="CPU workers for frame rendering (runs while GPU encodes)")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--encoder", default="av1",
                        help="encoder: av1 (default, GPU), h264, lossless, auto, libx265, libx264")
    parser.add_argument("--crf", type=int, default=20,
                        help="CQ/CRF quality (lower=better); av1_nvenc: 0=max quality, 51=worst")
    parser.add_argument("--particle-radius", type=int, default=2,
                        help="particle display radius in pixels (2 = 5x5 square block)")
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="re-run simulation even if raw data already exists")
    parser.add_argument("--delete-raw", action="store_true",
                        help="delete raw simulation data after encoding")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--methods", default=None,
                        help="filter by method: pic,flip,apic")
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-release")
    parser.add_argument("--build-jobs", type=int, default=None)
    args = parser.parse_args()

    threads = args.threads if args.threads is not None else scheduler_threads()
    build_jobs = args.build_jobs if args.build_jobs is not None else scheduler_threads()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    style_src = POSTPRO_ROOT / "style.css"
    if style_src.exists():
        shutil.copy2(style_src, out_dir / "style.css")
    manifest_csv = out_dir / "manifest.csv"
    _render_index_html(out_dir, read_csv(manifest_csv))

    binary = args.binary
    if binary is None or not binary.exists():
        print("[video] binary not found, building...")
        binary = build_binary(args.build_dir, build_jobs, skip=False, build_type="Release")
    binary = binary.resolve()

    if args.json is not None:
        roots = [args.json]
    elif args.tests is not None:
        roots = [Path(p.strip()) for p in args.tests.split(",") if p.strip()]
    else:
        roots = [PICM_ROOT / "test"]

    all_configs = _discover_configs(roots)
    sim_configs = [c for c in all_configs if _is_sim_config(c)]

    if not sim_configs:
        print(f"[video] no simulation configs found in: {roots}")
        return 0

    method_filter = None
    if args.methods:
        method_filter = {m.strip().lower() for m in args.methods.split(",") if m.strip()}

    runs_base = out_dir / ".runs"
    runs_base.mkdir(parents=True, exist_ok=True)

    encode_futures = []

    # max_workers=1: one GPU encoding job at a time, overlapping with CPU simulation
    with ThreadPoolExecutor(max_workers=1) as encode_pool:
        for config_path in sim_configs:
            name = _config_name(config_path)

            if method_filter:
                try:
                    with open(config_path) as fh:
                        cfg_method = json.load(fh).get("method", "pic").lower()
                    if cfg_method not in method_filter:
                        continue
                except (json.JSONDecodeError, OSError):
                    continue

            mp4_path = out_dir / f"{name}.mp4"

            if args.dry_run:
                print(f"[video] dry-run: would process {config_path} -> {mp4_path}")
                continue

            try:
                with open(config_path) as fh:
                    base_cfg = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[video] skip {name}: cannot read config: {exc}")
                continue

            run_dir = runs_base / name
            run_dir.mkdir(parents=True, exist_ok=True)
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            archive_path = run_dir / f"{name}.tar.xz"

            # ── CPU simulation (skip if raw data already present) ───────────
            existing_pvd = list(raw_dir.rglob("particles.pvd")) + \
                           list(raw_dir.rglob("normVelocity.pvd"))

            # If raw files were cleaned up but archive exists, extract it.
            if not existing_pvd and archive_path.exists() and not args.force:
                print(f"[video] extracting archive for {name}...")
                t0 = time.perf_counter()
                _extract_archive(archive_path, raw_dir)
                print(f"[video] extracted in {time.perf_counter() - t0:.1f}s")
                existing_pvd = list(raw_dir.rglob("particles.pvd")) + \
                               list(raw_dir.rglob("normVelocity.pvd"))

            cfg = copy.deepcopy(base_cfg)
            _method_val = cfg.get("method", "sl")
            is_sl = not _method_val or str(_method_val).lower() == "sl"
            cfg["write_particles"] = not is_sl
            cfg["write_norm_velocity"] = is_sl
            cfg["write_u"] = False
            cfg["write_v"] = False
            cfg["write_p"] = False
            cfg["write_div"] = False
            cfg["write_smoke"] = False
            cfg["write_vorticity"] = False
            cfg["sampling_rate"] = 1
            cfg["folder"] = str(raw_dir)
            cfg["filename"] = "simulation"

            if args.nt is not None:
                cfg["nt"] = args.nt

            mod_config_path = run_dir / f"{name}.json"
            with open(mod_config_path, "w") as fh:
                json.dump(cfg, fh, indent=2)

            if existing_pvd and not args.force:
                n_frames = len(list(existing_pvd[0].parent.rglob("*.vt[pi]")))
                print(f"[video] reusing existing data for {name} ({n_frames} frames found)")
            else:
                env = {"OMP_NUM_THREADS": str(threads)}
                cmd = [str(binary), str(mod_config_path)]
                print(f"[video] simulating {name}  (GPU may be encoding previous job)")
                t0 = time.perf_counter()
                result = run_binary(cmd, env)
                wall = time.perf_counter() - t0

                (run_dir / "stdout.log").write_bytes(result.stdout)
                (run_dir / "stderr.log").write_bytes(result.stderr)

                if result.returncode != 0:
                    print(f"[video] FAILED {name} (exit {result.returncode})")
                    _update_manifest(manifest_csv, name, config_path, "failed", "")
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    continue
                print(f"[video] simulation done ({wall:.1f}s)")
                # Keep only normVelocity* and particles* — remove label, u, v, p, div, etc.
                for f in raw_dir.iterdir():
                    if not (f.name.startswith("normVelocity") or f.name.startswith("particles")):
                        f.unlink(missing_ok=True)

                print(f"[video] archiving {name} (xz --extreme, please wait)...")
                t0 = time.perf_counter()
                _create_archive(archive_path, raw_dir)
                sz_mb = archive_path.stat().st_size / 1024**2
                print(f"[video] archive done in {time.perf_counter() - t0:.1f}s → {sz_mb:.1f} MB")

            # Resolve PVD paths
            particles_pvd = raw_dir / "particles.pvd"
            norm_vel_pvd = raw_dir / "normVelocity.pvd"
            if not particles_pvd.exists():
                found = list(raw_dir.rglob("particles.pvd"))
                if found:
                    particles_pvd = found[0]
            if not norm_vel_pvd.exists():
                found = list(raw_dir.rglob("normVelocity.pvd"))
                if found:
                    norm_vel_pvd = found[0]

            use_vti = is_sl and norm_vel_pvd.exists()
            use_vtp = not is_sl and particles_pvd.exists()

            if not use_vti and not use_vtp:
                print(f"[video] ERROR: no PVD output for {name} "
                      f"(raw_dir={raw_dir}, is_sl={is_sl}, "
                      f"particles_pvd_exists={particles_pvd.exists()}, "
                      f"norm_vel_pvd_exists={norm_vel_pvd.exists()})")
                _update_manifest(manifest_csv, name, config_path, "no_pvd", "")
                if args.delete_raw and run_dir.exists():
                    shutil.rmtree(run_dir)
                continue

            pvd_file = norm_vel_pvd if use_vti else particles_pvd
            all_pvd_paths = parse_pvd(pvd_file)
            frame_paths = [str(p) for p in all_pvd_paths if p.exists()]
            if not frame_paths:
                print(f"[video] ERROR: no frames found for {name} "
                      f"(pvd={pvd_file}, {len(all_pvd_paths)} entries in PVD, "
                      f"none exist on disk)")
                _update_manifest(manifest_csv, name, config_path, "no_frames", "")
                if args.delete_raw and run_dir.exists():
                    shutil.rmtree(run_dir)
                continue

            # ── Submit GPU encoding job (async, overlaps with next simulation) ──
            print(f"[video] queuing GPU encode for {name}")
            future = encode_pool.submit(
                _encode_job,
                {
                    "name": name,
                    "mp4_path": str(mp4_path),
                    "run_dir": str(run_dir),
                    "use_vti": use_vti,
                    "paths": frame_paths,
                    "config_path": str(config_path),
                    "fps": args.fps,
                    "cmap": args.cmap,
                    "width": args.width,
                    "height": args.height,
                    "n_workers": args.workers,
                    "encoder": args.encoder,
                    "crf": args.crf,
                    "particle_radius": args.particle_radius,
                    "delete_raw": args.delete_raw,
                },
                manifest_csv,
            )
            encode_futures.append(future)

    # Thread pool has shut down — all encoding jobs are complete.
    # Re-raise any unexpected exceptions from encoding jobs.
    for future in encode_futures:
        exc = future.exception()
        if exc is not None:
            print(f"[video] unexpected encoding error: {exc}")

    print(f"[video] done, manifest: {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
