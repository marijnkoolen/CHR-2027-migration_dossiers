"""
Generate a synthetic stand-in dataset of multi-page PDFs for the
sequence-context model (sequence_model.py / train_sequence.py).

Unlike make_dummy_dataset.py (one image per class, no page order), this
generates whole synthetic "PDFs": each is a run of several document segments
back-to-back, where every page of a segment shares the same Document type /
Layout Type Classification / Functional Categories (mirroring how a real
multi-page form or letter repeats its label across pages) and only the first
page of each segment is a Start page. Reuses the archetype rendering code
from make_dummy_dataset.py so the pages look like the same material.

Output: data/dummy_sequences/<split>/<pdf_id>/<page_number>.jpg (plus a
same-named .xml PageXML transcription), and a manifest.tsv with columns
matching this project's real annotation schema (image path, page number,
Document type, Layout Type Classification, Functional Categories, Start
page) plus `image` (actual image file), `pagexml` (its transcription - for
train_sequence.py's multimodal mode, --pagexml-col pagexml) and `split`
(assigned per PDF, not per page).

Usage:
    python scripts/vision/make_dummy_sequence_dataset.py --out-dir data/dummy_sequences
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from make_dummy_dataset import CLASSES, make_image, make_pagexml, synthetic_text_for

ARCHETYPE_TO_LAYOUT = {
    "form": "Structured Form",
    "letter": "Letter",
    "photo": "Photo",
    "card": "Card (Index Card)",
    "cover": "Cover",
    "mixed": "Other",
}

FUNCTIONAL_CATEGORIES = [
    "Application Documents",
    "Decision Documents",
    "Medical & Health Documents",
    "Qualification & Employment Proof",
    "Security & Political Screening Documents",
    "Administrative & Internal Processing Documents",
    "Other",
]


def build_functional_map(seed: int = 0) -> dict[str, str]:
    """A made-up but fixed label -> functional-category assignment, just so
    the demo data has a stable (if arbitrary) relationship to learn."""
    rng = random.Random(seed)
    return {label: rng.choice(FUNCTIONAL_CATEGORIES) for label, _, _ in CLASSES}


def make_pdf(pdf_id: str, functional_map: dict[str, str], out_dir: Path,
             n_segments_range=(3, 7), segment_len_weights=(0.5, 0.3, 0.15, 0.05)) -> list[dict]:
    labels, _, counts = zip(*CLASSES)
    weights = np.array(counts, dtype=float)
    weights /= weights.sum()

    n_segments = random.randint(*n_segments_range)
    rows = []
    page_number = 1
    for _ in range(n_segments):
        label = np.random.choice(labels, p=weights)
        archetype = next(a for l, a, _ in CLASSES if l == label)
        seg_len = np.random.choice(range(1, len(segment_len_weights) + 1), p=segment_len_weights)

        for i in range(seg_len):
            img = make_image(archetype)
            out_path = out_dir / pdf_id / f"{page_number}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, quality=random.randint(70, 95))

            pagexml_path = out_path.with_suffix(".xml")
            lines = synthetic_text_for(label, archetype)
            pagexml_path.write_text(make_pagexml(lines, *img.size), encoding="utf-8")

            rows.append(
                {
                    "image path": pdf_id,
                    "page number": page_number,
                    "image": str(out_path),
                    "pagexml": str(pagexml_path),
                    "Document type": label,
                    "Layout Type Classification": ARCHETYPE_TO_LAYOUT[archetype],
                    "Functional Categories": functional_map[label],
                    "Start page": "yes" if i == 0 else "no",
                }
            )
            page_number += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/dummy_sequences"))
    parser.add_argument("--n-train-pdfs", type=int, default=24)
    parser.add_argument("--n-val-pdfs", type=int, default=5)
    parser.add_argument("--n-test-pdfs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    functional_map = build_functional_map(args.seed)

    manifest_rows = []
    pdf_counter = 0
    for split, n_pdfs in [("train", args.n_train_pdfs), ("val", args.n_val_pdfs), ("test", args.n_test_pdfs)]:
        for _ in range(n_pdfs):
            pdf_id = f"dummy_pdf_{pdf_counter:04d}.pdf"
            pdf_counter += 1
            rows = make_pdf(pdf_id, functional_map, args.out_dir / split)
            for row in rows:
                row["split"] = split
            manifest_rows.extend(rows)

    manifest = pd.DataFrame(manifest_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_dir / "manifest.tsv", sep="\t", index=False)

    print(f"Wrote {len(manifest)} pages across {pdf_counter} PDFs to {args.out_dir}")
    print(manifest.groupby("split")["image path"].nunique().rename("n_pdfs").to_string())
    print(f"\nStart page rate: {(manifest['Start page'] == 'yes').mean():.2f}")
    print(f"Document type classes: {manifest['Document type'].nunique()}")


if __name__ == "__main__":
    main()
