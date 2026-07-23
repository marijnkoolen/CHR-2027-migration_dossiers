"""
Runs a trained sequence-context model (see train.py --mode sequence) over a
new, unlabeled corpus of PDFs (images + PageXML) and writes one row per page
with predicted Start page / Document type / Layout Type / Functional
Category, plus a confidence score and a per-document segment id for each.

Reads `model_config.json` + `classes.json` + `sequence_model.pt` from a
training run's --out-dir to reconstruct the exact trained architecture - no
need to re-specify --image-backbone/--n-heads/etc. by hand.

Segmentation uses the model's own start-page predictions (there's no ground
truth here), and predicted_segment_id groups pages into predicted documents
(`<pdf_id>::seg<n>`) - the unit flag_prediction_errors.py operates on.

Usage:
    python scripts/vision/predict.py \\
        --run-dir runs/quality_sequence_multimodal_start_page+document_type+layout_type+functional_category \\
        --manifest new_corpus_manifest.tsv --image-root <root> --out predictions.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from transformers import AutoTokenizer

from common import build_transforms, pick_device
from models import MultimodalPageEmbedder, PageEmbedder
from pagexml import extract_text
from sequence_data import PageBudgetBatchSampler
from sequence_model import SequenceContextModel


class InferencePageSequenceDataset(Dataset):
    """Like sequence_data.PageSequenceDataset, but for unlabeled data: groups
    pages by PDF (ordered by page number), with no label columns at all -
    __getitem__ also hands back the manifest rows themselves so predictions
    can be written out alongside every original column."""

    def __init__(self, manifest: pd.DataFrame, image_root: Path, transform,
                 pdf_col: str, page_col: str, image_col: str, pagexml_col: str | None = None):
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_col = image_col
        self.pagexml_col = pagexml_col
        self._text_cache: dict[str, str] = {}
        self.pdfs: list[pd.DataFrame] = [
            group.sort_values(page_col) for _, group in manifest.groupby(pdf_col, sort=False)
        ]

    def __len__(self) -> int:
        return len(self.pdfs)

    def page_counts(self) -> list[int]:
        return [len(g) for g in self.pdfs]

    def _text_for(self, pagexml_path: str) -> str:
        if not pagexml_path:
            return ""
        if pagexml_path not in self._text_cache:
            self._text_cache[pagexml_path] = extract_text(pagexml_path)
        return self._text_cache[pagexml_path]

    def __getitem__(self, idx: int):
        group = self.pdfs[idx]
        paths = [str(self.image_root / p) for p in group[self.image_col]]
        images = torch.stack([self.transform(default_loader(p)) for p in paths])
        if self.pagexml_col:
            texts = [
                self._text_for(str(self.image_root / p)) if pd.notna(p) else "" for p in group[self.pagexml_col]
            ]
        else:
            texts = [""] * len(paths)
        return images, texts, group


def make_inference_collate_fn(tokenizer=None, max_text_length: int = 256):
    def collate(batch: list[tuple]):
        lengths = [item[0].shape[0] for item in batch]
        B, T_max = len(batch), max(lengths)

        images_flat = torch.cat([item[0] for item in batch], dim=0)
        batch_index = torch.cat([torch.full((n,), b, dtype=torch.long) for b, n in enumerate(lengths)])
        time_index = torch.cat([torch.arange(n) for n in lengths])
        padding_mask = torch.ones(B, T_max, dtype=torch.bool)
        for b, n in enumerate(lengths):
            padding_mask[b, :n] = False

        result = {
            "images_flat": images_flat, "batch_index": batch_index, "time_index": time_index,
            "padding_mask": padding_mask, "groups": [item[2] for item in batch],
        }
        if tokenizer is not None:
            texts_flat = [text for _, texts, _ in batch for text in texts]
            encoded = tokenizer(
                texts_flat, padding=True, truncation=True, max_length=max_text_length, return_tensors="pt"
            )
            result["input_ids_flat"] = encoded["input_ids"]
            result["attention_mask_flat"] = encoded["attention_mask"]
        return result

    return collate


def embed_pages(embedder, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["images_flat"].to(device)
    if "input_ids_flat" in batch:
        embeds_flat = embedder(images, batch["input_ids_flat"].to(device), batch["attention_mask_flat"].to(device))
    else:
        embeds_flat = embedder(images)
    B, T = batch["padding_mask"].shape
    embeddings = torch.zeros(B, T, embedder.embed_dim, device=device, dtype=embeds_flat.dtype)
    embeddings[batch["batch_index"], batch["time_index"]] = embeds_flat
    return embeddings, batch["padding_mask"].to(device)


@torch.no_grad()
def predict_all(embedder, seq_model, loader: DataLoader, classes: dict, device: torch.device, amp: bool) -> pd.DataFrame:
    embedder.eval()
    seq_model.eval()
    doctype_classes = classes["document_type"]
    layout_classes = classes["layout_type"]
    functional_classes = classes["functional_category"]

    all_rows = []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            embeddings, padding_mask = embed_pages(embedder, batch, device)
            out = seq_model(embeddings, padding_mask, true_start_page=None)

        start_prob = torch.sigmoid(out["start_logits"].float()).cpu()
        doctype_prob = F.softmax(out["doctype_logits"].float(), dim=-1).cpu()
        layout_prob = F.softmax(out["layout_logits"].float(), dim=-1).cpu()
        functional_prob = F.softmax(out["functional_logits"].float(), dim=-1).cpu()
        segment_ids = out["segment_ids"].cpu()

        for b, group in enumerate(batch["groups"]):
            for t in range(len(group)):
                row = group.iloc[t].to_dict()
                doc_idx = int(doctype_prob[b, t].argmax())
                lay_idx = int(layout_prob[b, t].argmax())
                func_idx = int(functional_prob[b, t].argmax())
                start_p = float(start_prob[b, t])
                row.update({
                    "predicted_start_page": "yes" if start_p > 0.5 else "no",
                    "start_page_confidence": max(start_p, 1 - start_p),
                    "predicted_document_type": doctype_classes[doc_idx],
                    "document_type_confidence": float(doctype_prob[b, t, doc_idx]),
                    "predicted_layout_type": layout_classes[lay_idx],
                    "layout_type_confidence": float(layout_prob[b, t, lay_idx]),
                    "predicted_functional_category": functional_classes[func_idx],
                    "functional_category_confidence": float(functional_prob[b, t, func_idx]),
                    "predicted_segment_local_id": int(segment_ids[b, t]),
                })
                all_rows.append(row)

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                         help="a train.py --mode sequence --out-dir, containing model_config.json/"
                              "classes.json/sequence_model.pt")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path(""))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf-col", default="pdf_name")
    parser.add_argument("--page-col", default="page_num")
    parser.add_argument("--image-col", default="img_path")
    parser.add_argument("--pagexml-col", default="text_path")
    parser.add_argument("--batch-size", type=int, default=8, help="PDFs per batch; ignored if --max-pages-per-batch is set")
    parser.add_argument("--max-pages-per-batch", type=int, default=None)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    amp = args.amp == "on" or (args.amp == "auto" and device.type == "cuda")
    print(f"device: {device}  amp(bf16): {amp}")

    config = json.loads((args.run_dir / "model_config.json").read_text())
    classes = json.loads((args.run_dir / "classes.json").read_text())
    print(f"model config: {config}")

    multimodal = config["modality"] == "multimodal"
    tokenizer = AutoTokenizer.from_pretrained(config["text_backbone"]) if multimodal else None

    sep = "\t" if str(args.manifest).endswith(".tsv") else ","
    manifest = pd.read_csv(args.manifest, sep=sep)
    print(f"{len(manifest)} pages, {manifest[args.pdf_col].nunique()} PDFs to predict")

    dataset = InferencePageSequenceDataset(
        manifest, args.image_root, build_transforms(config["image_size"], train=False),
        pdf_col=args.pdf_col, page_col=args.page_col, image_col=args.image_col,
        pagexml_col=args.pagexml_col if multimodal else None,
    )
    collate = make_inference_collate_fn(tokenizer, config["max_text_length"])
    if args.max_pages_per_batch:
        sampler = PageBudgetBatchSampler(dataset.page_counts(), args.max_pages_per_batch, shuffle=False)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    if multimodal:
        embedder = MultimodalPageEmbedder(
            config["image_backbone"], config["text_backbone"], max_text_length=config["max_text_length"],
            project_to=config["project_to"], device=device,
        ).to(device)
    else:
        embedder = PageEmbedder(config["image_backbone"], project_to=config["project_to"], device=device).to(device)

    seq_model = SequenceContextModel(
        embed_dim=config["embed_dim"], num_doctype=len(classes["document_type"]),
        num_layout=len(classes["layout_type"]), num_functional=len(classes["functional_category"]),
        n_heads=config["n_heads"], n_layers=config["n_layers"],
    ).to(device)

    state = torch.load(args.run_dir / "sequence_model.pt", map_location=device)
    embedder.load_state_dict(state["embedder"])
    seq_model.load_state_dict(state["seq_model"])
    print("loaded checkpoint")

    predictions = predict_all(embedder, seq_model, loader, classes, device, amp)
    predictions["predicted_segment_id"] = (
        predictions[args.pdf_col].astype(str) + "::seg" + predictions["predicted_segment_local_id"].astype(str)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, sep="\t", index=False)

    n_docs = predictions["predicted_segment_id"].nunique()
    print(f"\n{len(predictions)} pages, {n_docs} predicted documents")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
