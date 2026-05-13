#!/usr/bin/env python3
from typing import List, Optional, Tuple
"""Run Rankine-vortex simulations and compare to analytical solution.

Analytical Rankine vortex (inviscid, steady):
  u_θ = ω r          for r ≤ R_c   (solid-body core, ω_z = 2ω)
  u_θ = ω R_c² / r   for r > R_c   (potential outer,  ω_z = 0)

Numerical diffusion is quantified via the second moment of vorticity:
  σ²(t) = ∫ r² ω_z dA / ∫ ω_z dA
For 2D inviscid flow σ² = const.  Growth rate → effective numerical viscosity:
  ν_eff = (dσ²/dt) / 4

Produces:
  data/rankine/kinetic_energy.csv  — E_k(t)/E_k(0) time series
  data/rankine/sigma.csv           — σ²(t) time series + ν_eff estimate
  data/rankine/radial_profile.csv  — u_θ(r) at t_final vs analytical
  data/rankine/per_run/<run>_*.csv — per-run copies
"""

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from picm_postpro.paths import DATA_DIR, PICM_ROOT
from picm_postpro.core import (
    build_binary,
    drop_run,
    read_csv,
    read_vti_field,
    run_binary,
    scheduler_threads,
    write_csv,
)

RANKINE_CONFIG = PICM_ROOT / "test" / "PIC" / "extra" / "rankine-vortex.json"
N_RADIAL_BINS = 150  # bins for u_θ(r) profile


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _run_name(method: str, ppc: int, flip_coef: Optional[float], threads: int) -> str:
    name = f"rankine_{method}_ppc{ppc}"
    if method == "flip" and flip_coef is not None:
        name += f"_coefpic{flip_coef:g}"
    name += f"_t{threads}"
    return name


def _build_config(
    base: dict,
    method: str,
    ppc: int,
    flip_coef: Optional[float],
    raw_dir: Path,
    nt: int,
    sampling_rate: int,
    overrides: dict,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["method"] = method
    cfg["ppcx"] = ppc
    cfg["ppcy"] = ppc
    cfg["folder"] = str(raw_dir.resolve())
    cfg["filename"] = "simulation"
    cfg["nt"] = nt
    cfg["sampling_rate"] = sampling_rate

    if method == "flip" and flip_coef is not None:
        cfg["coefpic"] = flip_coef

    # normVelocity → KE;  u + v → curl for σ² and azimuthal profile
    cfg["write_norm_velocity"] = True
    cfg["write_u"] = True
    cfg["write_v"] = True
    cfg["write_vorticity"] = False
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_smoke"] = False
    cfg["write_particles"] = False

    for key, val in overrides.items():
        cfg[key] = val
    return cfg


def _run_sim(
    binary: Path,
    name: str,
    config: dict,
    run_dir: Path,
    threads: int,
    dry_run: bool,
) -> Tuple[str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)

    config_path = (run_dir / f"{name}.json").resolve()
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    if dry_run:
        print(f"[dry-run] would run: {binary} {config_path}")
        return "dry-run", 0.0

    env = {"OMP_NUM_THREADS": str(threads)}
    cmd = [str(binary), str(config_path)]
    print(f"[run] {name} (threads={threads})")
    t0 = time.perf_counter()
    result = run_binary(cmd, env)
    wall = time.perf_counter() - t0

    (run_dir / "stdout.log").write_bytes(result.stdout)
    (run_dir / "stderr.log").write_bytes(result.stderr)

    if result.returncode != 0:
        print(f"[run] FAILED {name} (exit {result.returncode})")
        return "failed", wall

    print(f"[run] OK {name} ({wall:.1f}s)")
    return "ok", wall


# ---------------------------------------------------------------------------
# PVD iterator
# ---------------------------------------------------------------------------

def _pvd_datasets(pvd_path: Path, dt: float, sampling_rate: int):
    """Yield (timestep_float, vti_path) from a PVD file."""
    import xml.etree.ElementTree as ET
    if not pvd_path.exists():
        return
    for sample_i, ds in enumerate(ET.parse(pvd_path).getroot().iter("DataSet")):
        t = float(sample_i) * float(dt) * float(sampling_rate)
        fpath = pvd_path.parent / ds.get("file", "")
        if fpath.exists():
            yield t, fpath


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _staggered_to_cellcentre(u_field, v_field, nx: int, ny: int):
    """Interpolate staggered u,v to cell-centred arrays of shape (ny, nx)."""
    if u_field.shape[1] == nx + 1:
        u_cc = 0.5 * (u_field[:, :-1] + u_field[:, 1:])
    else:
        u_cc = u_field[:ny, :nx]

    if v_field.shape[0] == ny + 1:
        v_cc = 0.5 * (v_field[:-1, :] + v_field[1:, :])
    else:
        v_cc = v_field[:ny, :nx]

    return u_cc[:ny, :nx], v_cc[:ny, :nx]


def _extract_all(
    raw_dir: Path,
    dx: float, dy: float,
    nx: int, ny: int,
    cx_phys: float, cy_phys: float,
    r_max_phys: float,
    dt: float,
    sampling_rate: int,
):
    """
    Iterate all u+v frames.  For each compute:
      - E_k from normVelocity
      - σ²  from second vorticity moment
      - u_θ(r) profile only at the LAST frame

    Returns:
      ke_rows    list of {sample, time, kinetic_energy, ke_norm}
      sigma_rows list of {sample, time, sigma_sq, sigma_sq_excess}
      rp_rows    list of {r, v_theta_sim}          (last frame only)
    """
    norm_frames = list(_pvd_datasets(raw_dir / "normVelocity.pvd", dt, sampling_rate))
    u_frames    = list(_pvd_datasets(raw_dir / "u.pvd", dt, sampling_rate))
    v_frames    = list(_pvd_datasets(raw_dir / "v.pvd", dt, sampling_rate))

    if not u_frames or not v_frames:
        return [], [], []

    # cell-centre grid (ny×nx)
    jj, ii = np.mgrid[0:ny, 0:nx]
    x_phys = (ii + 0.5) * dx
    y_phys = (jj + 0.5) * dy
    dx_rel = x_phys - cx_phys
    dy_rel = y_phys - cy_phys
    r_grid = np.sqrt(dx_rel ** 2 + dy_rel ** 2)

    ke_rows: List[dict] = []
    sigma_rows: List[dict] = []
    sigma_sq_0: Optional[float] = None

    # --- KE from normVelocity ---
    ke_by_sample = {}
    ke0 = None
    for i, (t, fpath) in enumerate(norm_frames):
        try:
            field = read_vti_field(fpath, "normVelocity")
            ke = float(np.sum(field ** 2) * dx * dy)
        except Exception:
            continue
        if ke0 is None:
            ke0 = ke if ke > 0 else None
        ke_by_sample[i] = (t, ke)

    if ke0 and ke0 > 0:
        for i, (t, ke) in sorted(ke_by_sample.items()):
            ke_rows.append({"sample": i, "time": t,
                            "kinetic_energy": ke, "ke_norm": ke / ke0})

    # --- σ² from curl(u, v) at each u+v frame ---
    last_u_path = last_v_path = None
    for i, ((tu, u_path), (tv, v_path)) in enumerate(zip(u_frames, v_frames)):
        try:
            u_f = read_vti_field(u_path, "u")
            v_f = read_vti_field(v_path, "v")
        except Exception:
            continue

        u_cc, v_cc = _staggered_to_cellcentre(u_f, v_f, nx, ny)

        # Vorticity via numpy gradient on cell-centred fields
        dv_dx = np.gradient(v_cc, dx, axis=1)
        du_dy = np.gradient(u_cc, dy, axis=0)
        omega_z = dv_dx - du_dy

        # Second moment only over cells with positive vorticity (core)
        pos = omega_z > 0
        sum_omega = float(np.sum(omega_z[pos]) * dx * dy)
        if sum_omega > 1e-12:
            sigma_sq = float(np.sum(r_grid[pos] ** 2 * omega_z[pos]) * dx * dy) / sum_omega
        else:
            sigma_sq = float("nan")

        if sigma_sq_0 is None and not math.isnan(sigma_sq):
            sigma_sq_0 = sigma_sq

        sigma_rows.append({
            "sample": i, "time": tu,
            "sigma_sq": sigma_sq,
            "sigma_sq_excess": sigma_sq - sigma_sq_0 if sigma_sq_0 is not None else float("nan"),
        })

        last_u_path = u_path
        last_v_path = v_path

    # --- Azimuthal profile at last frame ---
    rp_rows: List[dict] = []
    if last_u_path and last_v_path:
        try:
            u_f = read_vti_field(last_u_path, "u")
            v_f = read_vti_field(last_v_path, "v")
            u_cc, v_cc = _staggered_to_cellcentre(u_f, v_f, nx, ny)

            with np.errstate(divide="ignore", invalid="ignore"):
                v_theta = np.where(
                    r_grid > 1e-12,
                    (-u_cc * dy_rel + v_cc * dx_rel) / r_grid,
                    0.0,
                )

            r_bins = np.linspace(0, r_max_phys, N_RADIAL_BINS + 1)
            r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
            for k in range(N_RADIAL_BINS):
                mask = (r_grid >= r_bins[k]) & (r_grid < r_bins[k + 1])
                if mask.any():
                    rp_rows.append({
                        "r": float(r_centres[k]),
                        "v_theta_sim": float(np.mean(v_theta[mask])),
                    })
        except Exception:
            pass

    return ke_rows, sigma_rows, rp_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="pic,flip,apic")
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--flip-coef", default="0,0.01,0.05,0.1")
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=PICM_ROOT / "build-release")
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-raw", "--raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    threads  = args.threads    if args.threads    is not None else scheduler_threads()
    build_jobs = args.build_jobs if args.build_jobs is not None else scheduler_threads()
    out_dir  = args.out        if args.out        is not None else DATA_DIR / "rankine"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = out_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    binary = args.binary
    if binary is None or not binary.exists():
        print("[build] binary not found, building...")
        binary = build_binary(args.build_dir, build_jobs, skip=False, build_type="Release")
    binary = binary.resolve()

    if not RANKINE_CONFIG.exists():
        print(f"[error] config not found: {RANKINE_CONFIG}", file=sys.stderr)
        return 1
    with open(RANKINE_CONFIG) as fh:
        base_config = json.load(fh)

    dx = float(base_config.get("dx", 0.02))
    dy = float(base_config.get("dy", 0.02))
    nx = int(base_config.get("nx", 400))
    ny = int(base_config.get("ny", 400))
    nt = args.nt if args.nt is not None else int(base_config.get("nt", 2000))
    dt = float(base_config.get("dt", 0.002))
    sampling_rate = max(1, nt // args.samples)

    rk   = base_config.get("rankine_vortex", {})
    omega        = float(rk.get("omega", 2.5))
    core_r_cells = int(str(rk.get("core_r", 60)))
    core_r_phys  = core_r_cells * dx
    # outer confinement radius (strings like "nx/3" → evaluate)
    try:
        r_cells = int(eval(str(rk.get("r", nx // 3)),
                           {"nx": nx, "ny": ny}))
    except Exception:
        r_cells = nx // 3
    r_max_phys = r_cells * dx
    cx_phys = (nx // 2) * dx
    cy_phys = (ny // 2) * dy

    print(f"[info] omega={omega}, core_r={core_r_phys:.3f} m, "
          f"peak_u_theta={omega*core_r_phys:.3f} m/s, t_final={nt*dt:.2f} s")

    methods    = [m.strip() for m in args.methods.split(",")   if m.strip()]
    flip_coefs = [float(c.strip()) for c in args.flip_coef.split(",") if c.strip()]

    run_specs = []
    for method in methods:
        for coef in (flip_coefs if method == "flip" else [None]):
            run_specs.append((_run_name(method, args.ppc, coef, threads), method, coef))

    ke_csv    = out_dir / "kinetic_energy.csv"
    sigma_csv = out_dir / "sigma.csv"
    rp_csv    = out_dir / "radial_profile.csv"
    ke_rows   = read_csv(ke_csv)
    sigma_rows = read_csv(sigma_csv)
    rp_rows   = read_csv(rp_csv)
    completed = {r["run"] for r in ke_rows}

    runs_dir = out_dir / "runs"

    for name, method, coef in run_specs:
        run_dir = runs_dir / name
        raw_dir = run_dir / "raw"

        if not args.force and name in completed:
            if not args.keep_raw or raw_dir.exists():
                print(f"[skip] {name}")
                continue
            print(f"[rerun] {name} (raw output missing)")

        config = _build_config(
            base_config, method, args.ppc, coef,
            raw_dir, nt, sampling_rate, {},
        )

        status, _ = _run_sim(binary, name, config, run_dir, threads, args.dry_run)
        if args.dry_run or status != "ok":
            continue
        if np is None:
            print(f"[warn] numpy not available — raw kept in {raw_dir}")
            continue

        ke_data, sig_data, rp_data = _extract_all(
            raw_dir, dx, dy, nx, ny, cx_phys, cy_phys, r_max_phys,
            dt, sampling_rate
        )

        if not ke_data:
            print(f"[warn] no normVelocity data for {name} — binary may need rebuild")
            continue

        meta = {"run": name, "method": method, "ppc": args.ppc,
                "flip_coef": coef if coef is not None else ""}

        # KE
        write_csv(per_run_dir / f"{name}_ke.csv", ke_data)
        ke_rows = drop_run(ke_rows, name)
        ke_rows.extend({**meta, **r} for r in ke_data)
        write_csv(ke_csv, ke_rows)

        # σ²
        if sig_data:
            write_csv(per_run_dir / f"{name}_sigma.csv", sig_data)
            sigma_rows = drop_run(sigma_rows, name)
            sigma_rows.extend({**meta, **r} for r in sig_data)
            write_csv(sigma_csv, sigma_rows)

        # radial profile
        if rp_data:
            write_csv(per_run_dir / f"{name}_rp.csv", rp_data)
            rp_rows = drop_run(rp_rows, name)
            rp_rows.extend({**meta, **r} for r in rp_data)
            write_csv(rp_csv, rp_rows)

        print(f"[extract] {name}: {len(ke_data)} KE frames, "
              f"{len(sig_data)} σ² frames, {len(rp_data)} radial bins")

        if not args.keep_raw and raw_dir.exists():
            shutil.rmtree(raw_dir)

    print(f"[done] results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
