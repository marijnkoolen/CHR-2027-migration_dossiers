"""
Generates smaller thumbnail copies of the per-page PNG images in
data/images/<dossier>/*.png, mirroring the same subdirectory-per-dossier
structure under data/thumbs/ - so a thumbnail's path can always be derived
from its full-size image's path by swapping the root directory, for
building a grid view of pages by classification without loading full-size
scans.

Skips a thumbnail if it already exists and is newer than its source image,
so re-running after adding a few new dossiers only processes what's new.

Usage:
    python scripts/vision/make_thumbnails.py
    python scripts/vision/make_thumbnails.py --size 192 --images-dir data/images --thumbs-dir data/thumbs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def make_thumbnails(images_dir: Path, thumbs_dir: Path, size: int, force: bool) -> None:
    paths = sorted(images_dir.rglob("*.png"))
    print(f"{len(paths)} PNGs found under {images_dir}")

    made, skipped = 0, 0
    for src in paths:
        dst = thumbs_dir / src.relative_to(images_dir)
        if not force and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail((size, size), Image.LANCZOS)
            img.save(dst)
        made += 1
        if made % 200 == 0:
            print(f"  {made} thumbnails made so far...")

    print(f"\nMade {made} thumbnails, skipped {skipped} already up to date, wrote to {thumbs_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--thumbs-dir", type=Path, default=Path("data/thumbs"))
    parser.add_argument("--size", type=int, default=256, help="max width/height in pixels, aspect ratio preserved")
    parser.add_argument("--force", action="store_true", help="regenerate every thumbnail, even if already up to date")
    args = parser.parse_args()

    make_thumbnails(args.images_dir, args.thumbs_dir, args.size, args.force)


if __name__ == "__main__":
    main()
