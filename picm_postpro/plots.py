from pathlib import Path
from typing import Iterable, Optional, Tuple, Union


DEFAULT_FORMATS = ("png", "svg", "pdf", "jpg")


def parse_formats(value: Optional[Union[str, Iterable[str]]]) -> Tuple[str, ...]:
    if value is None:
        return DEFAULT_FORMATS
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    formats = tuple(dict.fromkeys(item.strip().lstrip(".") for item in items if item.strip()))
    if not formats:
        raise ValueError("empty image format list")
    return formats


def save_figure(fig, output_stem: Path, *, formats: Iterable[str] = DEFAULT_FORMATS, dpi: int = 180) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for image_format in formats:
        fig.savefig(output_stem.with_suffix(f".{image_format}"), dpi=dpi)
