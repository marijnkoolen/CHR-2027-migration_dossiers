"""
Generate a synthetic stand-in dataset for the document-type classification
pipeline, since the real scanned pages cannot leave this machine.

Each synthetic image is built from one of a handful of layout archetypes
(form / letter / photo / card / cover / mixed) and then degraded to mimic the
real material: varying paper colour and background, rotation, stains, torn
edges, noise, blur and uneven lighting. Class names and sample counts are a
made-up but realistically imbalanced taxonomy (a few common classes, a long
tail of rare ones) - not the project's real label list.

Images are written in torchvision.datasets.ImageFolder layout:
    <out-dir>/<split>/<class_name>/<uuid>.jpg
alongside a same-named synthetic PageXML transcription (<uuid>.xml, English/
Dutch/mixed filler text - see train_multimodal.py), plus a flat manifest.tsv
with (image, pagexml, split, label, archetype).

Usage:
    python scripts/vision/make_dummy_dataset.py --out-dir data/dummy_images --scale 0.3
"""

from __future__ import annotations

import argparse
import math
import random
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# (class name, layout archetype, sample count at scale=1.0)
# Counts mirror the shape of a real merged multi-annotator label distribution:
# a handful of dominant classes, a mid tail, and several near-singleton classes.
CLASSES: list[tuple[str, str, int]] = [
    ("testimonial_labour", "letter", 309),
    ("form_d1", "form", 270),
    ("form_dm1", "form", 249),
    ("portrait_photo", "photo", 205),
    ("report_selection", "form", 187),
    ("testimonial_medical", "letter", 176),
    ("form_d2", "form", 155),
    ("letter_procedure", "letter", 139),
    ("passage_agreement", "form", 133),
    ("background_check", "form", 104),
    ("registration_card", "card", 102),
    ("approval_notice", "letter", 80),
    ("cover_page", "cover", 65),
    ("testimonial_medical_form", "letter", 36),
    ("other_misc", "mixed", 34),
    ("form_xs1", "form", 28),
    ("testimonial_status", "letter", 22),
    ("emigration_report", "letter", 10),
    ("form_47a", "form", 10),
    ("trade_proforma", "form", 9),
    ("checksheet_xs1", "form", 8),
    ("form_xs2", "form", 8),
    ("medical_declaration", "letter", 8),
    ("form_40", "form", 8),
    ("consent_parental", "letter", 5),
    ("checksheet_d1", "form", 4),
    ("form_k", "form", 4),
    ("checksheet_47", "form", 4),
    ("rare_letter_variant", "letter", 2),
    ("rare_form_variant", "form", 1),
]

PAPER_COLORS = [
    (250, 248, 244),  # white-ish
    (238, 228, 200),  # aged cream
    (210, 180, 130),  # tan cardboard
    (120, 95, 65),  # dark cardboard folder
    (25, 25, 25),  # black background/folder
    (170, 178, 185),  # grey-blue card
]
STAIN_COLORS = [(150, 110, 40), (90, 60, 20), (40, 40, 40), (180, 150, 90)]

# Made-up English/Dutch filler for the synthetic PageXML transcriptions -
# not translations of each other, just two pools mimicking bureaucratic
# document phrasing in each language, mixed per page to mirror the real
# material (some pages entirely English, others entirely Dutch, some mixed).
EN_PHRASES = [
    "Application for Assisted Passage", "This is to certify that", "Date of birth",
    "Signed at Rotterdam", "Medical examination report", "Approved by the selection officer",
    "Reference number", "Please find enclosed", "Yours faithfully", "Registration card",
    "Photograph attached", "Place of employment", "Next of kin", "Country of destination",
]
NL_PHRASES = [
    "Aanvraag tot geassisteerde overtocht", "Hierbij verklaren wij dat", "Geboortedatum",
    "Getekend te Rotterdam", "Medisch onderzoeksrapport", "Goedgekeurd door de selectieambtenaar",
    "Referentienummer", "Hierbij ingesloten", "Hoogachtend", "Registratiekaart",
    "Foto bijgevoegd", "Plaats van tewerkstelling", "Naaste familie", "Land van bestemming",
]


def synthetic_text_for(label: str, archetype: str) -> list[str]:
    """A few lines of fake transcribed text for the PageXML companion file -
    a heading derived from the label (real form/letter headers often do
    state their own type) plus filler lines in English, Dutch, or a mix."""
    language = random.choice(["en", "nl", "mixed"])
    pool = EN_PHRASES if language == "en" else NL_PHRASES if language == "nl" else EN_PHRASES + NL_PHRASES
    n_lines = {"photo": 0, "card": 3, "cover": 2}.get(archetype, random.randint(4, 10))

    lines = []
    if random.random() < 0.7:
        lines.append(label.replace("_", " ").title())
    lines.extend(random.choices(pool, k=n_lines))
    return lines


def make_pagexml(lines: list[str], width: int, height: int) -> str:
    """A minimal but schema-valid PAGE XML document with one TextRegion
    containing one TextLine per line of text. Coordinates are dummy
    placeholders (0,0) - only the transcribed text is used downstream."""
    text_lines_xml = "\n".join(
        f'      <TextLine id="l{i}">\n'
        f'        <Coords points="0,0 0,0 0,0 0,0"/>\n'
        f'        <TextEquiv><Unicode>{escape(line)}</Unicode></TextEquiv>\n'
        f'      </TextLine>'
        for i, line in enumerate(lines)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">\n'
        f'  <Page imageWidth="{width}" imageHeight="{height}">\n'
        '    <TextRegion id="r1">\n'
        '      <Coords points="0,0 0,0 0,0 0,0"/>\n'
        f'{text_lines_xml}\n'
        '    </TextRegion>\n'
        '  </Page>\n'
        '</PcGts>\n'
    )


def rand_paper_color() -> tuple[int, int, int]:
    base = random.choice(PAPER_COLORS)
    jitter = lambda c: max(0, min(255, c + random.randint(-12, 12)))
    return tuple(jitter(c) for c in base)


def draw_scribble(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, color, n_points=8, width=2):
    """A wavy polyline standing in for a line of handwriting or a signature."""
    pts = []
    for i in range(n_points):
        x = x0 + (x1 - x0) * i / (n_points - 1)
        y = y0 + random.uniform(-1, 1) * (y1 - y0) / 2
        pts.append((x, y))
    draw.line(pts, fill=color, width=width, joint="curve")


def draw_typed_text(draw: ImageDraw.ImageDraw, x0, y0, x1, color, font, width_frac=1.0):
    text = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ")
                    for _ in range(random.randint(10, 40)))
    max_w = int((x1 - x0) * width_frac)
    draw.text((x0, y0), text[: max_w // 7], fill=color, font=font)


def render_content(img: Image.Image, archetype: str, ink_color: tuple[int, int, int]):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Courier New.ttf", size=max(10, h // 55))
    except OSError:
        font = ImageFont.load_default()

    handwritten = random.random() < 0.5
    margin = int(0.08 * w)

    if archetype == "form":
        n_rows = random.randint(5, 14)
        n_cols = random.randint(2, 4)
        row_h = (h - 2 * margin) / n_rows
        col_w = (w - 2 * margin) / n_cols
        for r in range(n_rows + 1):
            y = margin + r * row_h
            draw.line([(margin, y), (w - margin, y)], fill=ink_color, width=1)
        for c in range(n_cols + 1):
            x = margin + c * col_w
            draw.line([(x, margin), (x, h - margin)], fill=ink_color, width=1)
        for r in range(n_rows):
            for c in range(n_cols):
                if random.random() < 0.6:
                    cx0 = margin + c * col_w + 4
                    cy = margin + r * row_h + row_h * 0.6
                    cx1 = margin + (c + 1) * col_w - 4
                    if handwritten or random.random() < 0.4:
                        draw_scribble(draw, cx0, cy, cx1, cy, ink_color, width=1)
                    else:
                        draw_typed_text(draw, cx0, margin + r * row_h + 2, cx1, ink_color, font)

    elif archetype == "letter":
        n_lines = random.randint(10, 22)
        line_h = (h - 2 * margin) / (n_lines + 2)
        for i in range(n_lines):
            y = margin + i * line_h
            x1 = w - margin - random.uniform(0, 0.3) * (w - 2 * margin)
            if handwritten:
                draw_scribble(draw, margin, y, x1, y, ink_color, n_points=6, width=1)
            else:
                draw_typed_text(draw, margin, y - 4, x1, ink_color, font)
        if random.random() < 0.5:
            draw_scribble(draw, w * 0.55, h - margin, w * 0.85, h - margin * 1.3, ink_color, width=2)

    elif archetype == "photo":
        cx, cy = w * random.uniform(0.4, 0.6), h * random.uniform(0.35, 0.55)
        rx, ry = w * random.uniform(0.15, 0.25), h * random.uniform(0.25, 0.4)
        skin = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in (200, 170, 140))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=skin)
        draw.rectangle([cx - rx * 1.3, cy + ry * 0.7, cx + rx * 1.3, h], fill=tuple(c // 2 for c in skin))

    elif archetype == "card":
        n_lines = random.randint(3, 6)
        line_h = (h - 2 * margin) / (n_lines + 1)
        for i in range(n_lines):
            y = margin + i * line_h
            x1 = w - margin - random.uniform(0, 0.2) * (w - 2 * margin)
            if handwritten:
                draw_scribble(draw, margin, y, x1, y, ink_color, n_points=5, width=1)
            else:
                draw_typed_text(draw, margin, y - 4, x1, ink_color, font)

    elif archetype == "cover":
        cx, cy = w / 2, h * 0.4
        r = min(w, h) * random.uniform(0.15, 0.22)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink_color, width=3)
        for i in range(3):
            y = h * 0.7 + i * h * 0.06
            draw_typed_text(draw, w * 0.25, y, w * 0.75, ink_color, font, width_frac=0.8)

    else:  # mixed
        render_content(img, random.choice(["form", "letter", "photo", "card"]), ink_color)


def apply_degradations(img: Image.Image) -> Image.Image:
    w, h = img.size

    # rotation to mimic a slightly skewed photograph of the page
    angle = random.uniform(-6, 6)
    bg = img.getpixel((2, 2))
    img = img.rotate(angle, expand=True, fillcolor=bg, resample=Image.BICUBIC)

    # torn/missing corners: paint irregular background-coloured polygons over edges
    draw = ImageDraw.Draw(img)
    w2, h2 = img.size
    for _ in range(random.randint(0, 3)):
        corner = random.choice([(0, 0), (w2, 0), (0, h2), (w2, h2)])
        size = random.uniform(0.03, 0.1) * min(w2, h2)
        pts = [corner]
        for _ in range(4):
            pts.append((corner[0] + random.uniform(-size, size), corner[1] + random.uniform(-size, size)))
        draw.polygon(pts, fill=bg)

    # stains: blurred translucent blobs
    if random.random() < 0.6:
        stain_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(stain_layer)
        for _ in range(random.randint(1, 4)):
            color = random.choice(STAIN_COLORS) + (random.randint(40, 100),)
            cx, cy = random.uniform(0, w2), random.uniform(0, h2)
            r = random.uniform(0.05, 0.18) * min(w2, h2)
            sdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        stain_layer = stain_layer.filter(ImageFilter.GaussianBlur(radius=8))
        img = Image.alpha_composite(img.convert("RGBA"), stain_layer).convert("RGB")

    # lighting / contrast jitter
    arr = np.asarray(img).astype(np.float32)
    brightness = random.uniform(0.7, 1.2)
    contrast = random.uniform(0.8, 1.2)
    arr = (arr - 128) * contrast + 128 * brightness
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # optional vignette for uneven lighting
    if random.random() < 0.3:
        yy, xx = np.mgrid[0:h2, 0:w2]
        cx, cy = w2 / 2, h2 / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(cx, cy)
        vignette = np.clip(1 - 0.5 * dist, 0.4, 1.0)[..., None]
        arr = np.asarray(img).astype(np.float32) * vignette
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # sensor noise
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.normal(0, random.uniform(2, 10), arr.shape).astype(np.int16)
    img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    # occasional soft focus
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return img


def make_image(archetype: str) -> Image.Image:
    if archetype == "photo":
        w, h = random.randint(500, 700), random.randint(650, 900)
    elif archetype == "card":
        w, h = random.randint(450, 650), random.randint(300, 450)
    else:
        w, h = random.randint(600, 850), random.randint(850, 1150)

    paper = rand_paper_color()
    ink = (20, 20, 20) if sum(paper) > 380 else (230, 225, 215)

    # background surrounding the paper (table/folder visible around the page)
    bg_color = rand_paper_color()
    canvas = Image.new("RGB", (int(w * 1.15), int(h * 1.15)), bg_color)
    page = Image.new("RGB", (w, h), paper)
    render_content(page, archetype, ink)
    offset = (random.randint(0, canvas.width - w), random.randint(0, canvas.height - h))
    canvas.paste(page, offset)

    return apply_degradations(canvas)


def split_indices(n: int, ratios=(0.7, 0.15, 0.15)) -> list[str]:
    if n < 3:
        return ["train"] * n
    n_train = max(1, round(n * ratios[0]))
    n_val = max(1, round(n * ratios[1])) if n - n_train >= 2 else 0
    n_test = n - n_train - n_val
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    random.shuffle(labels)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/dummy_images"))
    parser.add_argument("--scale", type=float, default=1.0, help="multiply all class counts by this factor")
    parser.add_argument("--min-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest = []
    too_small_for_split = []

    for label, archetype, count in CLASSES:
        n = max(args.min_per_class, round(count * args.scale))
        splits = split_indices(n)
        if len(set(splits)) < 2:
            too_small_for_split.append((label, n))

        for split in splits:
            img = make_image(archetype)
            stem = uuid.uuid4().hex[:12]
            out_path = args.out_dir / split / label / f"{stem}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, quality=random.randint(70, 95))

            pagexml_path = out_path.with_suffix(".xml")
            lines = synthetic_text_for(label, archetype)
            pagexml_path.write_text(make_pagexml(lines, *img.size), encoding="utf-8")

            manifest.append({
                "image": str(out_path), "pagexml": str(pagexml_path), "split": split,
                "label": label, "archetype": archetype,
            })

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(args.out_dir / "manifest.tsv", sep="\t", index=False)

    print(f"Wrote {len(manifest_df)} images across {len(CLASSES)} classes to {args.out_dir}")
    print(manifest_df.groupby("split").size().to_string())
    if too_small_for_split:
        print("\nClasses too small to appear in val/test (train-only, like real singleton labels):")
        for label, n in too_small_for_split:
            print(f"  - {label}: {n} sample(s)")


if __name__ == "__main__":
    main()
