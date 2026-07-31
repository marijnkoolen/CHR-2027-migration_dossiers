"""
Shared image-discovery and output-path helpers for run_got_ocr2.py and
run_qwen_vl.py - lets either script take a flat --images list or a
recursive --input-dir (mirrored into --output-dir the same way
make_thumbnails.py mirrors data/images/<dossier>/ into data/thumbs/<dossier>/),
and skip pages whose output is already up to date, so an interrupted
multi-hour batch run can be resumed without redoing finished pages.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect_images(images: list[Path] | None, input_dir: Path | None) -> list[tuple[Path, Path]]:
    """Returns [(source_path, relative_path)], sorted. With --input-dir,
    relative_path is relative to input_dir (preserving subdirectory
    structure for mirroring into --output-dir); with --images, it's just
    the bare filename."""
    if input_dir is not None:
        paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        return [(p, p.relative_to(input_dir)) for p in paths]
    return [(p, Path(p.name)) for p in images]


def output_path(output_dir: Path, relative_path: Path, suffix: str) -> Path:
    """suffix e.g. '.txt', '.md', '.bbox.json' - replaces relative_path's
    own extension (with_suffix treats the whole given string as the new
    final suffix, so a multi-dot suffix like '.bbox.json' works as-is)."""
    return output_dir / relative_path.with_suffix(suffix)


def is_up_to_date(src: Path, dst: Path) -> bool:
    return dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime
