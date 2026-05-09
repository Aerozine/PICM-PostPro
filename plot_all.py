#!/usr/bin/env python3
"""Regenerate derived CSV files and plots from PICM PostPro data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pic_scaling
import report_compare
import solver_iterations
from picm_postpro.paths import DATA_DIR, IMG_DIR
from picm_postpro.plots import parse_formats


REPORT_DIRS = (
    "report_comparisons",
    "postpro_run",
    "study_energy",
    "study_vorticity",
    "study_ppc_impact",
)


def run_report_dirs(data_root: Path, img_root: Path, image_formats: tuple[str, ...], make_plots: bool) -> None:
    for name in REPORT_DIRS:
        out_dir = data_root / name
        if not (out_dir / "summary.csv").is_file():
            continue
        report_compare.postprocess_csv(out_dir)
        if make_plots:
            report_compare.plot_all(out_dir, img_root / name, image_formats)


def run_scaling_dirs(data_root: Path, img_root: Path, image_formats: tuple[str, ...], make_plots: bool) -> None:
    for name in ("pic_scaling", "study_pic_scaling"):
        out_dir = data_root / name
        if not (out_dir / "summary.csv").is_file():
            continue
        pic_scaling.postprocess_csv(out_dir)
        if make_plots:
            pic_scaling.plot_scaling(out_dir, img_root / name, image_formats)


def run_solver_dirs(data_root: Path, img_root: Path, image_formats: tuple[str, ...], make_plots: bool) -> None:
    out_dir = data_root / "study_iterative_solvers"
    if not (out_dir / "summary.csv").is_file():
        return
    solver_iterations.postprocess_csv(out_dir)
    if make_plots:
        solver_iterations.plot_iterations(out_dir, img_root / "study_iterative_solvers", image_formats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--img", type=Path, default=IMG_DIR)
    parser.add_argument("--image-formats", default="png,svg,pdf,jpg")
    parser.add_argument("--postpro-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data.resolve()
    img_root = args.img.resolve()
    image_formats = parse_formats(args.image_formats)
    make_plots = not args.postpro_only

    run_report_dirs(data_root, img_root, image_formats, make_plots)
    run_scaling_dirs(data_root, img_root, image_formats, make_plots)
    run_solver_dirs(data_root, img_root, image_formats, make_plots)

    if make_plots:
        print(f"[plot] regenerated plots under {img_root}")
    print(f"[postpro] CSV data is under {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
