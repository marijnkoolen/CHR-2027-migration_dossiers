"""
Pre-computes per-page embeddings for troubleshooting the sequence-context
model, so they can be shared for debugging without sharing the underlying
images/text at all - only page embeddings (opaque vectors from a frozen
pretrained model, not reconstructable back into the original image or its
text), page labels, and an anonymized per-PDF identifier (real ids in this
project are archival filenames that embed real people's surnames - never
written to either output file).

Run this once (with access to the real images/PageXML), then train.py
--mode sequence --cached-embeddings <out-dir> trains directly from the
result - the backbone forward pass (the dominant per-epoch cost when
training on raw images) never needs to run again, since sequence mode
always freezes the backbone anyway (see SEQUENCE_MODE_OVERRIDES in
train.py).

The backbone is always used frozen (no gradient, eval mode): this captures
exactly the input the sequence-context model receives at the very start of
training, which is what matters for debugging its architecture and data
pipeline in isolation from backbone fine-tuning.

--augment-passes N additionally embeds N randomly-augmented copies of each
TRAINING-split PDF (val/test always stay on the single deterministic pass,
since they need to be reproducible for evaluation). Each pass becomes its
own virtual PDF (pdf_id suffixed _augK, same pages/labels), so a later
--cached-embeddings run sees more distinct document sequences without ever
touching a raw image again. These copies are fixed once, then reused
identically across every epoch of that later run - unlike live per-epoch
augmentation, N is not capped by --epochs (a run still cycles through all
N+1 copies every epoch, regardless of N). The real tradeoffs are elsewhere:
higher N costs more disk and more compute per epoch (~linearly), for
diminishing marginal benefit once N starts covering most of what the
transform's rotation/crop/color-jitter ranges can actually produce - beyond
that, extra passes increasingly resemble ones already in the cache. A
handful (3-8) is a reasonable starting range.

Usage:
    python scripts/classification/precompute_embeddings.py \\
        --manifest <your real page-level manifest> --image-root <root> \\
        --out-dir data/embeddings --image-backbone facebook/dinov2-small

    # or, to reproduce the multimodal run that showed the collapse:
    python scripts/classification/precompute_embeddings.py \\
        --manifest <manifest> --image-root <root> --out-dir data/embeddings_multimodal \\
        --image-backbone facebook/dinov2-small --modality multimodal \\
        --text-backbone bert-base-uncased --pagexml-col <col>

    # or, text-only (no --image-backbone/--image-size involved at all):
    python scripts/classification/precompute_embeddings.py \\
        --manifest <manifest> --image-root <root> --out-dir data/embeddings_text \\
        --modality text --text-backbone bert-base-uncased --pagexml-col <col>

Writes:
    <out-dir>/embeddings.npy            (N_pages, D) float32
    <out-dir>/embeddings_manifest.tsv   row_id, pdf_id (anonymized), page_number,
                                        split, start_page, document_type,
                                        layout_type, functional_category
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from common import build_transforms, pick_device
from models import MultimodalPageEmbedder, PageEmbedder, TextEmbedder
from pagexml import extract_text


def build_pdf_id_mapping(pdf_values: pd.Series) -> dict:
    """Deterministic, order-preserving real-pdf-id -> pdf_00001-style id.
    Exposed as a dict (not applied directly) so augmented passes can reuse
    the exact same mapping as the base pass - each PDF's augmented copies
    then share its anonymized base id (just _augK-suffixed), rather than
    each pass computing its own independent, inconsistent numbering."""
    unique_ids = pdf_values.drop_duplicates().tolist()
    return {real: f"pdf_{i:05d}" for i, real in enumerate(unique_ids)}


def embed_rows(rows: pd.DataFrame, transform, embedder, args, device: torch.device,
                image_root: Path, label: str) -> np.ndarray:
    n = len(rows)
    embeddings = np.zeros((n, embedder.embed_dim), dtype=np.float32)
    n_empty_text, n_total_text = 0, 0
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            batch = rows.iloc[start : start + args.batch_size]

            if args.modality == "text":
                # no image ever touched - the whole point of this modality
                texts = [
                    extract_text(str(image_root / p)) if pd.notna(p) else "" for p in batch[args.pagexml_col]
                ]
                n_empty_text += sum(1 for t in texts if not t.strip())
                n_total_text += len(texts)
                encoded = embedder.tokenizer(
                    texts, padding=True, truncation=True, max_length=args.max_text_length, return_tensors="pt"
                )
                out = embedder(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            else:
                images = torch.stack(
                    [transform(default_loader(str(image_root / p))) for p in batch[args.image_col]]
                ).to(device)
                if args.modality == "vision":
                    out = embedder(images)
                else:
                    texts = [
                        extract_text(str(image_root / p)) if pd.notna(p) else "" for p in batch[args.pagexml_col]
                    ]
                    n_empty_text += sum(1 for t in texts if not t.strip())
                    n_total_text += len(texts)
                    encoded = embedder.text_embedder.tokenizer(
                        texts, padding=True, truncation=True, max_length=args.max_text_length, return_tensors="pt"
                    )
                    out = embedder(images, encoded["input_ids"].to(device), encoded["attention_mask"].to(device))

            embeddings[start : start + len(batch)] = out.cpu().numpy()
            if (start // args.batch_size) % 20 == 0:
                print(f"  [{label}] {start + len(batch)}/{n}")

    # An empty string tokenizes to the same [CLS][SEP]-only input every
    # time, so if PageXML paths are silently failing to resolve/parse
    # (wrong --pagexml-col, wrong --image-root, or a tag-name mismatch in
    # pagexml.py's assumptions about the PAGE-XML schema in use),
    # extract_text()'s "return '' rather than raise" fallback (by design,
    # so pages that legitimately have no transcription - e.g. photos -
    # don't break the run) quietly produces near-identical embeddings for
    # every page instead of an error. A handful of legitimately blank pages
    # is normal; most/all of them being blank is not.
    if n_total_text > 0:
        empty_frac = n_empty_text / n_total_text
        if empty_frac > 0.5:
            print(f"  WARNING [{label}]: {n_empty_text}/{n_total_text} ({empty_frac:.0%}) pages produced empty "
                  f"text - check --pagexml-col/--image-root are correct and that pagexml.py's tag-name "
                  f"assumptions (TextLine/TextEquiv/Unicode) match your PAGE-XML schema. Embeddings from mostly-"
                  f"empty text will be near-identical regardless of each page's real content.")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path(""))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=["vision", "multimodal", "text"], default="vision")
    parser.add_argument("--image-backbone", default="facebook/dinov2-small")
    parser.add_argument("--text-backbone", default="bert-base-uncased")
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
    parser.add_argument("--start-col", default="start_page")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--augment-passes", type=int, default=0,
                         help="also embed N randomly-augmented copies of each split=='train' PDF, each as its "
                              "own virtual PDF (pdf_id suffixed _augK) - see module docstring. 0 = off (default).")
    parser.add_argument("--augment-strength", choices=["moderate", "strong"], default="moderate")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    if args.augment_passes > 0 and args.modality == "text":
        # A frozen text backbone in eval() mode (no dropout) is fully
        # deterministic - re-embedding the same text N times produces
        # bit-identical vectors, so augmented passes would just be wasted
        # disk/compute here, unlike for images (where the transform draws a
        # fresh random crop/rotation/jitter every call).
        print("--augment-passes has no effect for --modality text (a frozen text encoder is deterministic - "
              "there's no text-side equivalent of image augmentation here) - ignoring.")
        args.augment_passes = 0

    sep = "\t" if str(args.manifest).endswith(".tsv") else ","
    manifest = pd.read_csv(args.manifest, sep=sep)
    n = len(manifest)
    print(f"{n} pages, {manifest[args.pdf_col].nunique()} PDFs")

    if args.modality == "vision":
        embedder = PageEmbedder(args.image_backbone, unfreeze_last_n_blocks=0, device=device).to(device)
    elif args.modality == "text":
        embedder = TextEmbedder(args.text_backbone, unfreeze_last_n_layers=0, max_length=args.max_text_length,
                                 device=device).to(device)
    else:
        embedder = MultimodalPageEmbedder(
            args.image_backbone, args.text_backbone, unfreeze_image_blocks=0, unfreeze_text_layers=0,
            max_text_length=args.max_text_length, device=device,
        ).to(device)
    embedder.eval()

    eval_transform = build_transforms(args.image_size, train=False)
    image_root = Path(args.image_root)
    pdf_id_mapping = build_pdf_id_mapping(manifest[args.pdf_col])

    def export_rows(rows: pd.DataFrame, pdf_ids: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({
            "pdf_id": pdf_ids,
            "page_number": rows[args.page_col],
            "split": rows[args.split_col] if args.split_col in rows.columns else "unassigned",
            "start_page": rows[args.start_col],
            "document_type": rows[args.doctype_col],
            "layout_type": rows[args.layout_col],
            "functional_category": rows[args.functional_col],
        })

    all_embeddings = [embed_rows(manifest, eval_transform, embedder, args, device, image_root, "base")]
    all_export = [export_rows(manifest, manifest[args.pdf_col].map(pdf_id_mapping))]

    if args.augment_passes > 0:
        train_rows = manifest[manifest[args.split_col] == "train"] if args.split_col in manifest.columns else manifest.iloc[0:0]
        if train_rows.empty:
            print(f"--augment-passes {args.augment_passes} requested but no rows have "
                  f"{args.split_col}=='train' - skipping.")
        else:
            train_transform = build_transforms(args.image_size, train=True, augment_strength=args.augment_strength)
            train_pdf_ids = train_rows[args.pdf_col].map(pdf_id_mapping)
            for pass_idx in range(1, args.augment_passes + 1):
                label = f"aug {pass_idx}/{args.augment_passes}"
                all_embeddings.append(embed_rows(train_rows, train_transform, embedder, args, device, image_root, label))
                all_export.append(export_rows(train_rows, train_pdf_ids + f"_aug{pass_idx}"))
            print(f"added {args.augment_passes} augmented pass(es) over {train_rows[args.pdf_col].nunique()} "
                  f"training PDFs ({len(train_rows)} pages/pass)")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    export = pd.concat(all_export, ignore_index=True)
    export.insert(0, "row_id", range(len(export)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "embeddings.npy", all_embeddings)
    export.to_csv(args.out_dir / "embeddings_manifest.tsv", sep="\t", index=False)

    print(f"\nWrote {args.out_dir / 'embeddings.npy'} ({all_embeddings.shape}, {all_embeddings.nbytes / 1e6:.1f} MB)")
    print(f"Wrote {args.out_dir / 'embeddings_manifest.tsv'} ({len(export)} rows, {export['pdf_id'].nunique()} PDFs)")
    print(
        "\nNeither file contains image paths, PageXML text, or original document "
        "identifiers - safe to share."
    )


if __name__ == "__main__":
    main()
