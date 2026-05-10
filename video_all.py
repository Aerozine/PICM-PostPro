#!/usr/bin/env python3
"""Run particle-output test JSON files and encode one MP4 per config."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/picm_matplotlib")

import particles_to_mp4
from picm_postpro.paths import MISC_DIR, PICM_ROOT, VIDEO_DIR


DEFAULT_CONFIG_ROOTS = (
    PICM_ROOT / "test" / "PIC",
    PICM_ROOT / "test" / "FLIP",
    PICM_ROOT / "test" / "APIC",
)
HEAVY_OUTPUT_FLAGS = (
    "write_u",
    "write_v",
    "write_p",
    "write_div",
    "write_smoke",
    "write_norm_velocity",
)


def scheduler_threads() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def parse_roots(value: Optional[str]) -> Tuple[Path, ...]:
    if not value:
        return DEFAULT_CONFIG_ROOTS
    roots = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = PICM_ROOT / path
        roots.append(path.resolve())
    if not roots:
        raise ValueError("empty config root list")
    return tuple(dict.fromkeys(roots))


def load_config(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def config_method(cfg: Dict) -> str:
    return str(cfg.get("method", "")).lower()


def relative_to_test(config_path: Path) -> Path:
    test_root = PICM_ROOT / "test"
    try:
        return config_path.resolve().relative_to(test_root.resolve())
    except ValueError:
        return Path(config_path.stem)


def slug_for(relative: Path) -> Path:
    return relative.with_suffix("")


def discover_configs(roots: Iterable[Path], include_no_write_particles: bool) -> List[Tuple[Path, Dict]]:
    selected = []
    for root in roots:
        if not root.is_dir():
            print(f"[warn] missing config root: {root}")
            continue
        for config_path in sorted(root.rglob("*.json")):
            try:
                cfg = load_config(config_path)
            except Exception as exc:
                print(f"[warn] {config_path}: cannot read JSON: {exc}")
                continue
            if not include_no_write_particles and not bool(cfg.get("write_particles", False)):
                continue
            method = config_method(cfg)
            if method not in ("pic", "flip", "apic"):
                print(f"[skip] {config_path}: method={method or '<missing>'}")
                continue
            selected.append((config_path, cfg))
    return selected


def write_run_config(
    source_config: Path,
    base_cfg: Dict,
    config_path: Path,
    raw_dir: Path,
    keep_config_outputs: bool,
    overrides: argparse.Namespace,
) -> Dict:
    cfg = dict(base_cfg)
    cfg["folder"] = str(raw_dir)
    cfg["filename"] = "simulation"
    cfg["write_particles"] = True
    if not keep_config_outputs:
        for key in HEAVY_OUTPUT_FLAGS:
            cfg[key] = False
    if overrides.nt is not None:
        cfg["nt"] = overrides.nt
    if overrides.nx is not None:
        cfg["nx"] = overrides.nx
    if overrides.ny is not None:
        cfg["ny"] = overrides.ny
    if overrides.samples is not None:
        nt = int(cfg.get("nt", 1))
        cfg["sampling_rate"] = max(1, nt // max(1, overrides.samples))

    config_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as handle:
        json.dump(cfg, handle, indent=2)
        handle.write("\n")
    print(f"[config] {source_config} -> {config_path}")
    return cfg


def run_simulation(binary: Path, config_path: Path, run_dir: Path, threads: int) -> Tuple[int, float]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PROC_BIND", "spread")
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_DYNAMIC", "false")
    command = [str(binary), str(config_path)]
    print(f"[run] OMP_NUM_THREADS={threads} {' '.join(command)}")
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=PICM_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    elapsed = time.perf_counter() - start
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.log").write_text(result.stdout)
    (run_dir / "stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        print(f"[fail] {config_path.name}: simulation exited with {result.returncode}")
    return result.returncode, elapsed


def encode_video(raw_dir: Path, out_path: Path, args: argparse.Namespace, title: str) -> bool:
    pvd_path = raw_dir / "particles.pvd"
    if not pvd_path.is_file():
        print(f"[warn] missing particle sequence: {pvd_path}")
        return False
    paths = [path for path in particles_to_mp4.parse_pvd(pvd_path) if path.exists()]
    if args.sample > 1:
        paths = paths[:: args.sample]
    if not paths:
        print(f"[warn] no VTP frames listed by {pvd_path}")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    particles_to_mp4.build_mp4(
        paths,
        out_path,
        fps=args.fps,
        cmap_name=args.cmap,
        width=args.width,
        height=args.height,
        title=title,
        mode=args.mode,
        n_workers=args.workers,
        background=args.background,
        encoder=args.encoder,
        crf=args.crf,
        preset=args.preset,
    )
    return True


def write_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "status",
        "method",
        "config",
        "generated_config",
        "raw_dir",
        "video",
        "returncode",
        "wall_time_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config-roots", default=None)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--misc-dir", type=Path, default=MISC_DIR / "video")
    parser.add_argument("--threads", type=int, default=scheduler_threads())
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--mode", choices=("speed", "density"), default="speed")
    parser.add_argument(
        "--encoder",
        default="auto",
        help="ffmpeg encoder policy: auto, hardware, software, libx265, or libx264",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=24,
        help="constant-quality value; lower is higher quality and larger files",
    )
    parser.add_argument("--preset", default="veryslow")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--keep-config-outputs", action="store_true")
    parser.add_argument("--include-no-write-particles", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()

    args.binary = args.binary.resolve()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    args.video_dir = args.video_dir.resolve()
    args.misc_dir = args.misc_dir.resolve()
    args.threads = max(1, args.threads)
    args.workers = max(1, args.workers)
    args.sample = max(1, args.sample)

    configs = discover_configs(parse_roots(args.config_roots), args.include_no_write_particles)
    if args.limit is not None:
        configs = configs[: max(0, args.limit)]
    if not configs:
        print("[video] no JSON configs selected")
        return 0

    print(f"[video] selected {len(configs)} particle config(s)")
    rows = []
    failures = 0
    for source_config, base_cfg in configs:
        relative = relative_to_test(source_config)
        run_stem = slug_for(relative)
        generated_config = args.misc_dir / "configs" / relative
        run_dir = args.misc_dir / "runs" / run_stem
        raw_dir = run_dir / "raw"
        out_path = args.video_dir / relative.with_suffix(".mp4")
        method = config_method(base_cfg)

        row = {
            "status": "planned",
            "method": method,
            "config": str(source_config),
            "generated_config": str(generated_config),
            "raw_dir": str(raw_dir),
            "video": str(out_path),
            "returncode": "",
            "wall_time_s": "",
        }
        rows.append(row)

        if out_path.exists() and not args.force:
            print(f"[skip] {out_path}: already exists")
            row["status"] = "skipped"
            continue
        if args.dry_run:
            print(f"[dry-run] {source_config} -> {out_path}")
            row["status"] = "dry-run"
            continue

        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        cfg = write_run_config(
            source_config,
            base_cfg,
            generated_config,
            raw_dir,
            args.keep_config_outputs,
            args,
        )
        returncode, elapsed = run_simulation(args.binary, generated_config, run_dir, args.threads)
        row["returncode"] = returncode
        row["wall_time_s"] = elapsed
        if returncode != 0:
            row["status"] = "failed"
            failures += 1
            continue

        try:
            title = f"{relative.as_posix()} ({cfg.get('method', method)})"
            ok = encode_video(raw_dir, out_path, args, title)
        except Exception as exc:
            print(f"[fail] {source_config}: video encoding failed: {exc}")
            ok = False
        row["status"] = "ok" if ok else "failed"
        if not ok:
            failures += 1
        if not args.keep_raw and raw_dir.exists():
            shutil.rmtree(raw_dir)

    manifest = args.misc_dir / "manifest.csv"
    write_manifest(manifest, rows)
    print(f"[video] manifest: {manifest}")
    print(f"[video] output:   {args.video_dir}")
    if failures:
        print(f"[video] {failures} config(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
