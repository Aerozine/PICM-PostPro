#!/usr/bin/env python3
from typing import List
"""Discover test configs, run simulations with particles, generate MP4s."""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from picm_postpro.paths import PICM_ROOT, POSTPRO_ROOT, VIDEO_DIR
from picm_postpro.core import build_binary, run_binary, scheduler_threads, write_csv, read_csv

try:
    from particles_to_mp4 import parse_pvd, build_mp4
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from particles_to_mp4 import parse_pvd, build_mp4


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
            # Drop 'test' and method dirs; keep section and stem
            skip = {"test", "PIC", "FLIP", "APIC", "GPIC", "FLAPIC", "SL", "extra"}
            name_parts = [p for p in parts[:-1] if p not in skip]
            name_parts.append(config_path.stem)
            return "_".join(name_parts) if name_parts else config_path.stem
        except ValueError:
            continue
    return config_path.stem


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
                        help="shorthand: run a single simulation config JSON "
                             "(equivalent to --tests path/to/config.json)")
    parser.add_argument("--out", type=Path, default=VIDEO_DIR)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--encoder", default="av1",
                        help="encoder: av1 (default), auto, libx265, libx264, hardware")
    parser.add_argument("--crf", type=int, default=28,
                        help="quality (lower=better); AV1: 28=good, 18=near-lossless")
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--force", action="store_true")
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

    # Resolve binary
    binary = args.binary
    if binary is None or not binary.exists():
        print("[video] binary not found, building...")
        binary = build_binary(args.build_dir, build_jobs, skip=False, build_type="Release")
    binary = binary.resolve()

    # Discover configs
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

    # Filter by method if requested
    method_filter = None
    if args.methods:
        method_filter = {m.strip().lower() for m in args.methods.split(",") if m.strip()}

    manifest_csv = out_dir / "manifest.csv"
    manifest_rows = read_csv(manifest_csv)
    completed = {r["name"] for r in manifest_rows if r.get("status") == "ok"}

    runs_base = out_dir / ".runs"
    runs_base.mkdir(parents=True, exist_ok=True)

    new_manifest_rows = list(manifest_rows)

    for config_path in sim_configs:
        name = _config_name(config_path)

        # Apply method filter
        if method_filter:
            try:
                with open(config_path) as fh:
                    cfg_data = json.load(fh)
                cfg_method = cfg_data.get("method", "pic").lower()
                if cfg_method not in method_filter:
                    continue
            except (json.JSONDecodeError, OSError):
                continue

        if not args.force and name in completed:
            print(f"[video] skip {name} (already completed)")
            continue

        mp4_path = out_dir / f"{name}.mp4"

        if args.dry_run:
            print(f"[video] dry-run: would process {config_path} -> {mp4_path}")
            continue

        # Build modified config
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

        cfg = copy.deepcopy(base_cfg)
        cfg["write_particles"] = True
        cfg["write_u"] = False
        cfg["write_v"] = False
        cfg["write_p"] = False
        cfg["write_div"] = False
        cfg["write_smoke"] = False
        cfg["write_norm_velocity"] = False
        cfg["write_vorticity"] = False
        cfg["folder"] = str(raw_dir)
        cfg["filename"] = "simulation"

        if args.nt is not None:
            cfg["nt"] = args.nt

        mod_config_path = run_dir / f"{name}.json"
        with open(mod_config_path, "w") as fh:
            json.dump(cfg, fh, indent=2)

        # Run simulation
        env = {"OMP_NUM_THREADS": str(threads)}
        cmd = [str(binary), str(mod_config_path)]
        print(f"[video] running {name}")
        t0 = time.perf_counter()
        result = run_binary(cmd, env)
        wall = time.perf_counter() - t0

        (run_dir / "stdout.log").write_bytes(result.stdout)
        (run_dir / "stderr.log").write_bytes(result.stderr)

        if result.returncode != 0:
            print(f"[video] FAILED {name} (exit {result.returncode})")
            new_manifest_rows = [r for r in new_manifest_rows if r.get("name") != name]
            new_manifest_rows.append({
                "name": name, "config": str(config_path),
                "status": "failed", "mp4_path": "",
            })
            write_csv(manifest_csv, new_manifest_rows)
            continue

        print(f"[video] simulation done ({wall:.1f}s), encoding...")

        # Find particles.pvd
        pvd_path = raw_dir / "particles.pvd"
        if not pvd_path.exists():
            # Search subdirs
            found_pvds = list(raw_dir.rglob("particles.pvd"))
            if found_pvds:
                pvd_path = found_pvds[0]
            else:
                print(f"[video] no particles.pvd found for {name}")
                new_manifest_rows = [r for r in new_manifest_rows if r.get("name") != name]
                new_manifest_rows.append({
                    "name": name, "config": str(config_path),
                    "status": "no_pvd", "mp4_path": "",
                })
                write_csv(manifest_csv, new_manifest_rows)
                continue

        vtp_paths = [p for p in parse_pvd(pvd_path) if p.exists()]
        if not vtp_paths:
            print(f"[video] no VTP frames for {name}")
            continue

        try:
            build_mp4(
                vtp_paths,
                mp4_path,
                fps=args.fps,
                cmap_name=args.cmap,
                width=args.width,
                height=args.height,
                title=name,
                n_workers=args.workers,
                encoder=args.encoder,
                crf=args.crf,
            )
            status = "ok"
        except Exception as exc:
            print(f"[video] encoding failed for {name}: {exc}")
            status = "encode_failed"

        # Cleanup raw
        if run_dir.exists():
            shutil.rmtree(run_dir)

        new_manifest_rows = [r for r in new_manifest_rows if r.get("name") != name]
        new_manifest_rows.append({
            "name": name, "config": str(config_path),
            "status": status, "mp4_path": str(mp4_path) if status == "ok" else "",
        })
        write_csv(manifest_csv, new_manifest_rows)

    print(f"[video] done, manifest: {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
