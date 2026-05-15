# PICM PostPro

This repository is meant to be checked out as the PICM post-processing
submodule:

```text
PICM/
  PostPro/
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
video/                generated MP4 particle clips
```

The Slurm jobs delete raw simulation fields after CSV extraction by default.
Set `PICM_KEEP_RAW=1` only when you explicitly need VTI/VTP/PVD files.
Run `make build` before `make sbatch`; the `.slurm` files always run with
`--skip-build`.

## Make Targets


From the PICM root, use the same targets through `make -C PostPro`:

- `make build` builds the PICM CPU OpenMP release binary and the debug binary
  used by the solver-iteration study.
- `make sbatch` checks that both binaries already exist, then submits each
  study with a separate `sbatch` command. Slurm jobs never compile PICM.
- `make postpro` regenerates derived CSV files from `data/`.
- `make plot` writes energy/vorticity comparison figures into `img/` in
  `png`, `svg`, `pdf`, and `jpg` formats.
- `make postpro-extract` rebuilds report CSV metrics and plots from raw
  simulation output already saved under `data/misc/`. This is the laptop-side
  step to run after a cluster job deferred extraction because `numpy` was not
  available there.
- `make postpro-run` builds PICM, runs the selected report simulations, and
  extracts CSV data/plots. By default it is a low-CPU local smoke run: PIC and
  pure FLIP only, one OpenMP thread.
- `make free-fall-particles` runs the no-water falling-block particle study
  and compares particle vertical velocity percentiles against `v_th = v0 + g t`.
- `make video` recursively scans `test/PIC/extra`, `test/FLIP`, and `test/APIC`,
  runs every JSON with `write_particles: true`, and writes one white-background
  `viridis` MP4 per config under `video/` using the same directory structure.
  Existing MP4 files are regenerated. Encoding uses automatic FFmpeg encoder
  selection, preferring working hardware HEVC/H.264 encoders and falling back
  to slow high-compression software HEVC/H.264.
- `make clean` removes PICM build folders, raw simulation fields, image output,
  video output, and Python caches while keeping CSV files under `data/`.
