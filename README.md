# PICM PostPro

This repository is meant to be checked out as the PICM post-processing
submodule:

```text
PICM/
  PostPro/
```

It also works while developed beside PICM:

```text
pi/
  PICM/
  PICM-PostPro/
```

PostPro auto-detects `PICM_ROOT`; override it only when needed:

```bash
PICM_ROOT=/path/to/PICM make -C . build
```

## Folder Layout

Generated artifacts stay inside this repository:

```text
data/                 CSV files used by plotting
data/misc/            generated configs, logs, and temporary postpro inputs
img/                  generated figures: png, svg, pdf, jpg
```

The Slurm jobs delete raw simulation fields after CSV extraction by default.
Set `PICM_KEEP_RAW=1` only when you explicitly need VTI/VTP/PVD files.
Run `make build` before `make sbatch`; the `.slurm` files always run with
`--skip-build`.

## Make Targets

Run these from `PICM/PostPro`:

```bash
make build
make sbatch
make postpro
make plot
make clean
```

From the PICM root, use the same targets through `make -C PostPro`:

```bash
make -C PostPro build
make -C PostPro sbatch
make -C PostPro postpro
make -C PostPro plot
make -C PostPro clean
```

- `make build` builds the PICM CPU OpenMP release binary and the debug binary
  used by the solver-iteration study.
- `make sbatch` checks that both binaries already exist, then submits each
  study with a separate `sbatch` command. Slurm jobs never compile PICM.
- `make postpro` regenerates derived CSV files from `data/`.
- `make plot` writes energy/vorticity comparison figures into `img/` in
  `png`, `svg`, `pdf`, and `jpg` formats.
- `make clean` removes PICM build folders, raw simulation fields, image output,
  and Python caches while keeping CSV files under `data/`.

## Server Jobs

Non-scaling Slurm jobs request 32 CPU cores and 16 GB RAM:

- `slurm/study_energy.slurm`
- `slurm/study_vorticity.slurm`
- `slurm/study_ppc_impact.slurm`
- `slurm/study_iterative_solvers.slurm`

The scaling job keeps the dedicated high-memory/exclusive node request so it can
measure `1,2,4,8,16,32,64` threads:

- `slurm/study_pic_scaling.slurm`

Common overrides:

```bash
PICM_REPORT_TEST=dambreak make sbatch
PICM_PPC_VALUES=2,3,4 sbatch slurm/study_ppc_impact.slurm
PICM_FLIP_COEF_PIC=0,0.02,0.05,0.1 sbatch slurm/study_vorticity.slurm
PICM_SCALING_THREADS=1,2,4,8,16,32,64 sbatch slurm/study_pic_scaling.slurm
PICM_SOLVER_TOLERANCES=1e-1,1e-2,1e-3 sbatch slurm/study_iterative_solvers.slurm
```

## Data and Image Overrides

Defaults:

```bash
PICM_POSTPRO_DATA=PICM/PostPro/data
PICM_POSTPRO_MISC=PICM/PostPro/data/misc
PICM_POSTPRO_IMG=PICM/PostPro/img
```

You normally do not need to set these.

## Plot Inputs

`make plot` uses `kinetic_energy.csv` and `vorticity.csv` for the main report
figures. `max_div.csv` is still kept as diagnostic CSV data, but it is not used
as the main PPC plot. If those two CSV files are empty, rerun the Slurm studies
with the current PostPro scripts so energy and vorticity are extracted before
raw VTK files are deleted.
