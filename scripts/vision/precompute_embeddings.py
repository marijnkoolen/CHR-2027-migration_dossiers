"""
Pre-computes per-page embeddings for troubleshooting the sequence-context
model, so they can be shared for debugging without sharing the underlying
images/text at all - only page embeddings (opaque vectors from a frozen
pretrained model, not reconstructable back into the original image or its
text), page labels, and an anonymized per-PDF identifier (real ids in this
project are archival filenames that embed real people's surnames - never
written to either output file).

Run this once on the machine with access to the real images/PageXML, then
copy the two output files into this project - nothing else is needed to
reproduce/debug the sequence-context model's behaviour on the real label
distribution and document-length structure.

The backbone is always used frozen (no gradient, eval mode): this captures
exactly the input the sequence-context model receives at the very start of
training, which is what matters for debugging its architecture and data
pipeline in isolation from backbone fine-tuning.

Usage:
    python scripts/vision/precompute_embeddings.py \\
        --manifest <your real page-level manifest> --image-root <root> \\
        --out-dir data/embeddings --image-backbone facebook/dinov2-small

    # or, to reproduce the multimodal run that showed the collapse:
    python scripts/vision/precompute_embeddings.py \\
        --manifest <manifest> --image-root <root> --out-dir data/embeddings_multimodal \\
        --image-backbone facebook/dinov2-small --modality multimodal \\
        --text-backbone xlm-roberta-base --pagexml-col <col>

Writes:
    <out-dir>/embeddings.npy            (N_pages, D) float32
    <out-dir>/embeddings_manifest.tsv   row_id, pdf_id (anonymized), page_number,
                                        split, start_page, document_type,
                                        layout_type, functional_category
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader

from common import build_transforms, pick_device
from models import MultimodalPageEmbedder, PageEmbedder
from pagexml import extract_text


def anonymize_pdf_ids(pdf_values: pd.Series) -> pd.Series:
    """Deterministic, order-preserving pdf-id -> pdf_00001-style id."""
    unique_ids = pdf_values.drop_duplicates().tolist()
    mapping = {real: f"pdf_{i:05d}" for i, real in enumerate(unique_ids)}
    return pdf_values.map(mapping)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path(""))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=["vision", "multimodal"], default="vision")
    parser.add_argument("--image-backbone", default="facebook/dinov2-small")
    parser.add_argument("--text-backbone", default="xlm-roberta-base")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pdf-col", default="pdf_name")
    parser.add_argument("--page-col", default="page_num")
    parser.add_argument("--image-col", default="img_path")
    parser.add_argument("--pagexml-col", default="text_path")
    parser.add_argument("--doctype-col", default="document_type")
    parser.add_argument("--layout-col", default="layout_type")
    parser.add_argument("--functional-col", default="functional_category")
    parser.add_argument("--start-col", default="page_start")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    sep = "\t" if str(args.manifest).endswith(".tsv") else ","
    manifest = pd.read_csv(args.manifest, sep=sep)
    n = len(manifest)
    print(f"{n} pages, {manifest[args.pdf_col].nunique()} PDFs")

    if args.modality == "vision":
        embedder = PageEmbedder(args.image_backbone, unfreeze_last_n_blocks=0, device=device).to(device)
    else:
        embedder = MultimodalPageEmbedder(
            args.image_backbone, args.text_backbone, unfreeze_image_blocks=0, unfreeze_text_layers=0,
            max_text_length=args.max_text_length, device=device,
        ).to(device)
    embedder.eval()

    transform = build_transforms(args.image_size, train=False)
    image_root = Path(args.image_root)
    all_embeddings = np.zeros((n, embedder.embed_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            rows = manifest.iloc[start : start + args.batch_size]
            images = torch.stack(
                [transform(default_loader(str(image_root / p))) for p in rows[args.image_col]]
            ).to(device)

            if args.modality == "vision":
                out = embedder(images)
            else:
                texts = [
                    extract_text(str(image_root / p)) if pd.notna(p) else "" for p in rows[args.pagexml_col]
                ]
                encoded = embedder.text_embedder.tokenizer(
                    texts, padding=True, truncation=True, max_length=args.max_text_length, return_tensors="pt"
                )
                out = embedder(images, encoded["input_ids"].to(device), encoded["attention_mask"].to(device))

            all_embeddings[start : start + len(rows)] = out.cpu().numpy()
            if (start // args.batch_size) % 20 == 0:
                print(f"  {start + len(rows)}/{n}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "embeddings.npy", all_embeddings)

    export = pd.DataFrame(
        {
            "row_id": range(n),
            "pdf_id": anonymize_pdf_ids(manifest[args.pdf_col]),
            "page_number": manifest[args.page_col],
            "split": manifest[args.split_col] if args.split_col in manifest.columns else "unassigned",
            "start_page": manifest[args.start_col],
            "document_type": manifest[args.doctype_col],
            "layout_type": manifest[args.layout_col],
            "functional_category": manifest[args.functional_col],
        }
    )
    export.to_csv(args.out_dir / "embeddings_manifest.tsv", sep="\t", index=False)

    print(f"\nWrote {args.out_dir / 'embeddings.npy'} ({all_embeddings.shape}, {all_embeddings.nbytes / 1e6:.1f} MB)")
    print(f"Wrote {args.out_dir / 'embeddings_manifest.tsv'}")
    print(
        "\nNeither file contains image paths, PageXML text, or original document "
        "identifiers - safe to share."
    )


if __name__ == "__main__":
    main()
