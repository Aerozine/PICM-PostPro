import os
from pathlib import Path

POSTPRO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PICM_POSTPRO_DATA", POSTPRO_ROOT / "data")).expanduser()
IMG_DIR = Path(os.environ.get("PICM_POSTPRO_IMG", POSTPRO_ROOT / "img")).expanduser()
VIDEO_DIR = Path(os.environ.get("PICM_POSTPRO_VIDEO", POSTPRO_ROOT / "video")).expanduser()


def _is_picm_root(p: Path) -> bool:
    return (p / "CMakeLists.txt").is_file() and (p / "src").is_dir()


def find_picm_root() -> Path:
    candidates = []
    if os.environ.get("PICM_ROOT"):
        candidates.append(Path(os.environ["PICM_ROOT"]).expanduser())
    candidates += [
        POSTPRO_ROOT.parent,
        POSTPRO_ROOT.parent / "PICM",
        Path.cwd(),
        Path.cwd() / "PICM",
    ]
    seen: set[Path] = set()
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r in seen:
            continue
        seen.add(r)
        if _is_picm_root(r):
            return r
    raise FileNotFoundError(f"cannot locate PICM root; tried {candidates}")


PICM_ROOT = find_picm_root()
