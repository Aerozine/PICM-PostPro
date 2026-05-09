from __future__ import annotations

import os
from pathlib import Path


POSTPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = POSTPRO_ROOT
DATA_DIR = Path(os.environ.get("PICM_POSTPRO_DATA", POSTPRO_ROOT / "data")).expanduser()
MISC_DIR = Path(os.environ.get("PICM_POSTPRO_MISC", DATA_DIR / "misc")).expanduser()
IMG_DIR = Path(os.environ.get("PICM_POSTPRO_IMG", POSTPRO_ROOT / "img")).expanduser()


def _is_picm_root(path: Path) -> bool:
    return (path / "CMakeLists.txt").is_file() and (path / "src").is_dir()


def find_picm_root() -> Path:
    """Locate the PICM source tree from standalone or submodule layouts."""
    candidates: list[Path] = []
    env_root = os.environ.get("PICM_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        (
            POSTPRO_ROOT.parent,
            POSTPRO_ROOT.parent / "PICM",
            POSTPRO_ROOT.parent.parent,
            Path.cwd(),
            Path.cwd() / "PICM",
            Path.cwd().parent,
        )
    )

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_picm_root(resolved):
            return resolved

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not locate PICM root; tried: {tried}")


PICM_ROOT = find_picm_root()


def default_misc_dir(data_dir: Path) -> Path:
    return MISC_DIR / data_dir.resolve().name


def default_img_dir(data_dir: Path) -> Path:
    return IMG_DIR / data_dir.resolve().name
