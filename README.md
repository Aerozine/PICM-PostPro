# PICM PostPro

This is a submodule of PICM. After cloning PICM, initialise it with `git submodule update --init` and run targets from `PostPro/` (or via `make -C PostPro` from the PICM root).

## Targets

**`make video`** — discovers every simulation config under `test/`, runs any that have no cached data, and encodes one MP4 per config into `video/` using GPU AV1 (falls back to CPU). Re-running always re-encodes from cache without re-simulating; use `--force` to redo the simulation.

**`make archive`** — packages all MP4s in `video/` into both `video/videos.zip` and `video/videos.tar.xz` using maximum compression. Run after `make video`.

**`make plot`** — regenerates analysis figures from CSV data into `img/`. Run after simulation data exists under `data/`.
