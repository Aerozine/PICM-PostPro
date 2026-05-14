#!/usr/bin/env python3
from typing import List, Optional, Tuple
"""Run Lamb-Oseen vortex simulations and estimate numerical viscosity.

The Lamb-Oseen vortex has the exact analytical solution:
  u_θ(r,t) = (Γ/2πr) · (1 − exp(−r²/σ²(t)))
  σ²(t)    = 4ν·(t₀ + t_sim)      with  σ₀² = 4ν·t₀

where t₀ = physical_time (simulation starts on an already-aged vortex).
The solver is inviscid, so any growth of σ² beyond the analytical curve is
pure numerical diffusion:
  Δσ²(t_sim) = σ²_num(t_sim) − σ²_analytical(t_sim) = 4·ν_num·t_sim

Produces:
  data/lamb/sigma.csv           — σ², Δσ², instantaneous ν_num per snapshot
  data/lamb/radial_profile.csv  — u_θ(r) at t_final + analytical curve
  data/lamb/per_run/<run>_*.csv — per-run copies
"""

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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

LAMB_CONFIG = PICM_ROOT / "test" / "PIC" / "extra" / "lamb-oseen-vortex.json"
N_RADIAL_BINS = 150


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------

def lamb_oseen_sigma_sq(nu: float, t0: float, t_sim: float) -> float:
    return 4.0 * nu * (t0 + t_sim)


def lamb_oseen_v_theta(r, Gamma: float, sigma_sq: float):
    """Analytical azimuthal velocity for Lamb-Oseen vortex."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            r > 1e-12,
            (Gamma / (2.0 * math.pi * r)) * (1.0 - np.exp(-r ** 2 / sigma_sq)),
            0.0,
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _run_name(method: str, ppc: int, flip_coef: Optional[float], threads: int) -> str:
    name = f"lamb_{method}_ppc{ppc}"
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
    cfg.setdefault("solver", {}).update({"type": "cg", "max_iterations": 10_000, "tolerance": 1e-2})

    if method == "flip" and flip_coef is not None:
        cfg["coefPic"] = flip_coef
        cfg["coefpic"] = flip_coef

    # vorticity → σ²;  u+v → azimuthal profile at t_final
    cfg["write_vorticity"] = True
    cfg["write_u"] = True
    cfg["write_v"] = True
    cfg["write_norm_velocity"] = False
    cfg["write_p"] = False
    cfg["write_div"] = False
    cfg["write_smoke"] = False
    cfg["write_particles"] = False

    for key, val in overrides.items():
        cfg[key] = val
    return cfg


def _run_sim(
    binary: Path, name: str, config: dict,
    run_dir: Path, threads: int, dry_run: bool,
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
    t0 = time.perf_counter()
    result = run_binary([str(binary), str(config_path)], env)
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

def _pvd_datasets(pvd_path: Path):
    import xml.etree.ElementTree as ET
    if not pvd_path.exists():
        return
    for ds in ET.parse(pvd_path).getroot().iter("DataSet"):
        t = float(ds.get("timestep", 0))
        fpath = pvd_path.parent / ds.get("file", "")
        if fpath.exists():
            yield t, fpath


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _staggered_to_cc(u_field, v_field, nx, ny):
    u_cc = (0.5 * (u_field[:, :-1] + u_field[:, 1:])
            if u_field.shape[1] == nx + 1 else u_field[:ny, :nx])
    v_cc = (0.5 * (v_field[:-1, :] + v_field[1:, :])
            if v_field.shape[0] == ny + 1 else v_field[:ny, :nx])
    return u_cc[:ny, :nx], v_cc[:ny, :nx]


def _extract_sigma_series(
    raw_dir: Path,
    dx: float, dy: float, nx: int, ny: int,
    cx_phys: float, cy_phys: float,
    nu: float, t0: float,
) -> List[dict]:
    """
    Compute σ²(t) from the vorticity field second moment at every snapshot.
    Also return Δσ² = σ²_num − σ²_analytical and instantaneous ν_num.
    """
    jj, ii = np.mgrid[0:ny, 0:nx]
    x_phys = (ii + 0.5) * dx
    y_phys = (jj + 0.5) * dy
    r_sq   = (x_phys - cx_phys) ** 2 + (y_phys - cy_phys) ** 2

    rows: List[dict] = []
    for i, (t_sim, fpath) in enumerate(_pvd_datasets(raw_dir / "vorticity.pvd")):
        try:
            omega_z = read_vti_field(fpath, "vorticity")
        except Exception:
            continue

        pos      = omega_z > 0.0
        sum_w    = float(np.sum(omega_z[pos]) * dx * dy)
        if sum_w < 1e-12:
            continue

        sigma_sq  = float(np.sum(r_sq[pos] * omega_z[pos]) * dx * dy) / sum_w
        sigma_sq_analytical = lamb_oseen_sigma_sq(nu, t0, t_sim)
        delta_sigma_sq = sigma_sq - sigma_sq_analytical
        # instantaneous ν_num = Δσ² / (4·t_sim) for t_sim > 0
        nu_num = delta_sigma_sq / (4.0 * t_sim) if t_sim > 1e-9 else float("nan")

        rows.append({
            "sample": i,
            "time": t_sim,
            "sigma_sq": sigma_sq,
            "sigma_sq_analytical": sigma_sq_analytical,
            "delta_sigma_sq": delta_sigma_sq,
            "nu_num": nu_num,
        })
    return rows


def _extract_radial_profile(
    raw_dir: Path,
    dx: float, dy: float, nx: int, ny: int,
    cx_phys: float, cy_phys: float,
    r_max_phys: float,
    Gamma: float, nu: float, t0: float,
) -> List[dict]:
    """u_θ(r) at the last u+v frame, with analytical Lamb-Oseen at same time."""
    u_frames = list(_pvd_datasets(raw_dir / "u.pvd"))
    v_frames = list(_pvd_datasets(raw_dir / "v.pvd"))
    if not u_frames or not v_frames:
        return []

    t_sim, u_path = u_frames[-1]
    _,     v_path = v_frames[-1]
    try:
        u_cc, v_cc = _staggered_to_cc(
            read_vti_field(u_path, "u"),
            read_vti_field(v_path, "v"),
            nx, ny,
        )
    except Exception:
        return []

    jj, ii = np.mgrid[0:ny, 0:nx]
    x_p = (ii + 0.5) * dx
    y_p = (jj + 0.5) * dy
    dx_rel = x_p - cx_phys
    dy_rel = y_p - cy_phys
    r_grid = np.sqrt(dx_rel ** 2 + dy_rel ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        v_theta_num = np.where(
            r_grid > 1e-12,
            (-u_cc * dy_rel + v_cc * dx_rel) / r_grid,
            0.0,
        )

    r_bins    = np.linspace(0, r_max_phys, N_RADIAL_BINS + 1)
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
    sigma_sq_t0    = lamb_oseen_sigma_sq(nu, t0, 0.0)
    sigma_sq_tfin  = lamb_oseen_sigma_sq(nu, t0, t_sim)

    rows = []
    for k in range(N_RADIAL_BINS):
        mask = (r_grid >= r_bins[k]) & (r_grid < r_bins[k + 1])
        if not mask.any():
            continue
        r = float(r_centres[k])
        rows.append({
            "r": r,
            "time": t_sim,
            "v_theta_sim": float(np.mean(v_theta_num[mask])),
            "v_theta_analytical_t0":   float(lamb_oseen_v_theta(r, Gamma, sigma_sq_t0)),
            "v_theta_analytical_tfin": float(lamb_oseen_v_theta(r, Gamma, sigma_sq_tfin)),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods",   default="pic,flip,apic")
    parser.add_argument("--ppc",       type=int, default=3)
    parser.add_argument("--flip-coef", default="0,0.01,0.05,0.1")
    parser.add_argument("--nt",        type=int, default=None)
    parser.add_argument("--samples",   type=int, default=30,
                        help="number of vorticity output frames")
    parser.add_argument("--threads",   type=int, default=None)
    parser.add_argument("--out",       type=Path, default=None)
    parser.add_argument("--binary",    type=Path, default=None)
    parser.add_argument("--build-dir", type=Path,
                        default=PICM_ROOT / "build-release")
    parser.add_argument("--build-jobs",type=int,  default=None)
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--keep-raw",  action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    threads    = args.threads    or scheduler_threads()
    build_jobs = args.build_jobs or scheduler_threads()
    out_dir    = args.out        or DATA_DIR / "lamb"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = out_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    binary = args.binary
    if binary is None or not binary.exists():
        print("[build] binary not found, building...")
        binary = build_binary(args.build_dir, build_jobs,
                              skip=False, build_type="Release")
    binary = binary.resolve()

    if not LAMB_CONFIG.exists():
        print(f"[error] config not found: {LAMB_CONFIG}", file=sys.stderr)
        return 1
    with open(LAMB_CONFIG) as fh:
        base_config = json.load(fh)

    dx = float(base_config.get("dx", 0.02))
    dy = float(base_config.get("dy", 0.02))
    nx = int(base_config.get("nx", 256))
    ny = int(base_config.get("ny", 256))
    nt = args.nt or int(base_config.get("nt", 600))
    sampling_rate = max(1, nt // args.samples)

    lo  = base_config.get("lamb_oseen_vortex", {})
    nu  = float(lo.get("viscosity",     0.002))
    t0  = float(lo.get("physical_time", 20.0))
    omega = float(lo.get("omega", 2.0))

    # core_r can be given as int cells or physical; parse same way as C++
    core_r_raw = lo.get("core_r", None)
    if core_r_raw is not None:
        core_r_cells = int(core_r_raw)
        core_r_phys  = core_r_cells * 0.5 * (dx + dy)
    else:
        core_r_phys = math.sqrt(4.0 * nu * t0)   # fallback from viscosity+t0

    Gamma   = 2.0 * math.pi * omega * core_r_phys ** 2
    cx_phys = (nx // 2) * dx
    cy_phys = (ny // 2) * dy

    # confinement radius for profile (r = "nx/2-12" → eval)
    try:
        r_cells = int(eval(str(lo.get("r", nx // 2 - 12)),
                           {"nx": nx, "ny": ny}))
    except Exception:
        r_cells = nx // 2 - 12
    r_max_phys = r_cells * dx

    t_final = nt * float(base_config.get("dt", 0.002))
    print(f"[info] Lamb-Oseen: Γ={Gamma:.4f} m²/s, ν={nu}, t₀={t0} s, "
          f"σ₀={core_r_phys:.4f} m, t_final={t_final:.2f} s")

    methods    = [m.strip() for m in args.methods.split(",")    if m.strip()]
    flip_coefs = [float(c.strip()) for c in args.flip_coef.split(",") if c.strip()]

    run_specs = []
    for method in methods:
        for coef in (flip_coefs if method == "flip" else [None]):
            run_specs.append(
                (_run_name(method, args.ppc, coef, threads), method, coef))

    sig_csv = out_dir / "sigma.csv"
    rp_csv  = out_dir / "radial_profile.csv"
    sig_rows = read_csv(sig_csv)
    rp_rows  = read_csv(rp_csv)
    completed = {r["run"] for r in sig_rows}

    runs_dir = out_dir / "runs"

    for name, method, coef in run_specs:
        if not args.force and name in completed:
            print(f"[skip] {name}")
            continue

        run_dir = runs_dir / name
        raw_dir = run_dir / "raw"

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

        meta = {"run": name, "method": method, "ppc": args.ppc,
                "flip_coef": coef if coef is not None else ""}

        # σ² time series
        sig_data = _extract_sigma_series(
            raw_dir, dx, dy, nx, ny, cx_phys, cy_phys, nu, t0)
        if not sig_data:
            print(f"[warn] no vorticity data for {name} — rebuild binary?")
            continue
        write_csv(per_run_dir / f"{name}_sigma.csv", sig_data)
        sig_rows = drop_run(sig_rows, name)
        sig_rows.extend({**meta, **r} for r in sig_data)
        write_csv(sig_csv, sig_rows)

        # azimuthal profile at t_final
        rp_data = _extract_radial_profile(
            raw_dir, dx, dy, nx, ny, cx_phys, cy_phys,
            r_max_phys, Gamma, nu, t0)
        if rp_data:
            write_csv(per_run_dir / f"{name}_rp.csv", rp_data)
            rp_rows = drop_run(rp_rows, name)
            rp_rows.extend({**meta, **r} for r in rp_data)
            write_csv(rp_csv, rp_rows)

        print(f"[extract] {name}: {len(sig_data)} σ² frames, "
              f"{len(rp_data)} radial bins")

        if not args.keep_raw and raw_dir.exists():
            shutil.rmtree(raw_dir)

    print(f"[done] results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
