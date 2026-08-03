"""
Visualizes form_registration.py's detect_text_boxes() output on the image
itself, to make picking anchor strings and target field boxes for a new
FormTemplate faster than reading through a bare list of (text, box) pairs
blind. Draws every detected box with a small reference number; the full
detected text isn't drawn on the image itself (would be unreadable clutter
on a dense page) but is written to a companion table alongside it, so you
can look up what a given number says, then copy its box straight into a
FormTemplate.

Usage:
    python scripts/ocr/visualize_detected_text.py --image page.png --out page_annotated.png
    # then open page_annotated.png and page_annotated.txt side by side

    python scripts/ocr/visualize_detected_text.py --image page.png --out page_annotated.png \\
        --min-confidence 20   # lower to see more/noisier detections if an anchor is missing
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from form_registration import detect_text_boxes

BOX_COLOR = (255, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_BG_COLOR = (255, 0, 0)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # common on Linux
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def visualize(image_path: Path, out_path: Path, min_confidence: int) -> None:
    detected = detect_text_boxes(image_path, min_confidence=min_confidence)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(14)

    for i, d in enumerate(detected):
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=1)

        label = str(i)
        label_box = draw.textbbox((0, 0), label, font=font)
        label_w, label_h = label_box[2] - label_box[0], label_box[3] - label_box[1]
        # Prefer just above the box; if that would go off the top edge,
        # draw inside the box's top instead so the label stays visible.
        label_y = y1 - label_h - 3 if y1 - label_h - 3 >= 0 else y1 + 1
        draw.rectangle([x1, label_y, x1 + label_w + 3, label_y + label_h + 2], fill=LABEL_BG_COLOR)
        draw.text((x1 + 1, label_y), label, fill=LABEL_TEXT_COLOR, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    table_path = out_path.with_suffix(".txt")
    lines = [f"{i}\t{d.text!r}\t{tuple(round(v) for v in d.box)}" for i, d in enumerate(detected)]
    table_path.write_text("\n".join(lines) + "\n")

    print(f"{len(detected)} box(es) detected")
    print(f"wrote annotated image to {out_path}")
    print(f"wrote index -> (text, box) table to {table_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                         help="annotated image path - a same-named .txt index table is written alongside it")
    parser.add_argument("--min-confidence", type=int, default=40,
                         help="passed to detect_text_boxes - lower to see more/noisier detections")
    args = parser.parse_args()
    visualize(args.image, args.out, args.min_confidence)


if __name__ == "__main__":
    main()
