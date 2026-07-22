"""
Trains/evaluates the sequence-context model (sequence_model.py) directly on
pre-computed page embeddings (see precompute_embeddings.py) instead of raw
images - for reproducing and debugging the majority-class collapse without
needing the real images/PageXML at all.

Since the embeddings are frozen, fixed inputs (there is no backbone in this
script at all), the only thing being trained is SequenceContextModel itself:
the positional encoding, Transformer encoder, and the four heads. If the
collapse reproduces here, it's conclusively in this architecture or its data
pipeline (segmentation, batching, loss/eval) - not the backbone, not bf16,
and not anything specific to loading real images/text.

Usage:
    python scripts/vision/train_from_embeddings.py \\
        --embeddings data/embeddings/embeddings.npy \\
        --manifest data/embeddings/embeddings_manifest.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from common import pick_device
from sequence_model import SequenceContextModel

IGNORE_INDEX = -100


def build_label_vocab(manifest: pd.DataFrame, split_col: str, column: str, train_split: str = "train") -> list[str]:
    train_rows = manifest[manifest[split_col] == train_split]
    return sorted(train_rows[column].dropna().unique())


class EmbeddingSequenceDataset(Dataset):
    """One item = one PDF's ordered page embeddings, read directly from the
    precomputed (N_pages, D) array via each row's `row_id` - no image
    loading, no backbone."""

    def __init__(
        self, embeddings: np.ndarray, manifest: pd.DataFrame, split: str,
        doctype_classes: list[str], layout_classes: list[str], functional_classes: list[str],
        pdf_col: str = "pdf_id", page_col: str = "page_number", start_col: str = "start_page",
        doctype_col: str = "document_type", layout_col: str = "layout_type",
        functional_col: str = "functional_category", split_col: str = "split",
    ):
        self.embeddings = embeddings
        self.doctype_to_idx = {c: i for i, c in enumerate(doctype_classes)}
        self.layout_to_idx = {c: i for i, c in enumerate(layout_classes)}
        self.functional_to_idx = {c: i for i, c in enumerate(functional_classes)}
        self.start_col, self.doctype_col, self.layout_col, self.functional_col = (
            start_col, doctype_col, layout_col, functional_col
        )
        rows = manifest[manifest[split_col] == split]
        self.pdfs = [group.sort_values(page_col) for _, group in rows.groupby(pdf_col, sort=False)]

    def __len__(self) -> int:
        return len(self.pdfs)

    def __getitem__(self, idx: int):
        group = self.pdfs[idx]
        embeddings = torch.from_numpy(self.embeddings[group["row_id"].to_numpy()]).float()
        start = torch.tensor([1.0 if str(v).strip().lower() == "yes" else 0.0 for v in group[self.start_col]])
        doctype = torch.tensor([self.doctype_to_idx.get(v, IGNORE_INDEX) for v in group[self.doctype_col]])
        layout = torch.tensor([self.layout_to_idx.get(v, IGNORE_INDEX) for v in group[self.layout_col]])
        functional = torch.tensor([self.functional_to_idx.get(v, IGNORE_INDEX) for v in group[self.functional_col]])
        return embeddings, start, doctype, layout, functional


def collate(batch: list[tuple]) -> dict:
    lengths = [item[0].shape[0] for item in batch]
    B, T_max = len(batch), max(lengths)
    D = batch[0][0].shape[1]

    embeddings = torch.zeros(B, T_max, D)
    padding_mask = torch.ones(B, T_max, dtype=torch.bool)
    start = torch.zeros(B, T_max)
    doctype = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long)
    layout = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long)
    functional = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long)

    for b, (emb, s, d, l, f) in enumerate(batch):
        n = emb.shape[0]
        embeddings[b, :n] = emb
        padding_mask[b, :n] = False
        start[b, :n] = s
        doctype[b, :n] = d
        layout[b, :n] = l
        functional[b, :n] = f

    return {
        "embeddings": embeddings, "padding_mask": padding_mask,
        "start": start, "doctype": doctype, "layout": layout, "functional": functional,
    }


def compute_losses(out: dict, batch: dict, device, start_pos_weight: float) -> dict:
    padding_mask = batch["padding_mask"].to(device)
    valid = ~padding_mask

    start_target = batch["start"].to(device)
    start_loss_per_page = F.binary_cross_entropy_with_logits(
        out["start_logits"], start_target, reduction="none",
        pos_weight=torch.tensor(start_pos_weight, device=device),
    )
    start_loss = (start_loss_per_page * valid).sum() / valid.sum().clamp(min=1)

    def ce_loss(logits, key):
        target = batch[key].to(device)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=IGNORE_INDEX)

    doctype_loss = ce_loss(out["doctype_logits"], "doctype")
    layout_loss = ce_loss(out["layout_logits"], "layout")
    functional_loss = ce_loss(out["functional_logits"], "functional")
    total = start_loss + doctype_loss + layout_loss + functional_loss
    return {"total": total, "start": start_loss, "doctype": doctype_loss, "layout": layout_loss, "functional": functional_loss}


@torch.no_grad()
def evaluate(seq_model, loader: DataLoader, device, classes: dict[str, list[str]] | None = None) -> dict:
    seq_model.eval()
    start_true, start_pred = [], []
    task_true = {"doctype": [], "layout": [], "functional": []}
    task_pred = {"doctype": [], "layout": [], "functional": []}

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        out = seq_model(embeddings, padding_mask, true_start_page=None)
        valid = (~padding_mask).cpu()

        start_true.append(batch["start"][valid])
        start_pred.append((torch.sigmoid(out["start_logits"]).cpu() > 0.5).float()[valid])
        for key in task_true:
            target = batch[key][valid]
            preds = out[f"{key}_logits"].argmax(dim=-1).cpu()[valid]
            keep = target != IGNORE_INDEX
            task_true[key].append(target[keep])
            task_pred[key].append(preds[keep])

    start_true_cat = torch.cat(start_true).numpy()
    start_pred_cat = torch.cat(start_pred).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        start_true_cat, start_pred_cat, average="binary", zero_division=0
    )
    metrics = {"start_precision": precision, "start_recall": recall, "start_f1": f1}
    if classes is not None:
        metrics["start_report"] = classification_report(
            start_true_cat, start_pred_cat, target_names=["not_start_page", "start_page"], zero_division=0
        )

    for key in task_true:
        true_cat = torch.cat(task_true[key]) if task_true[key] else torch.tensor([])
        pred_cat = torch.cat(task_pred[key]) if task_pred[key] else torch.tensor([])
        if len(true_cat) == 0:
            metrics[f"{key}_accuracy"] = float("nan")
            metrics[f"{key}_macro_f1"] = float("nan")
            continue
        metrics[f"{key}_accuracy"] = (true_cat == pred_cat).float().mean().item()
        metrics[f"{key}_macro_f1"] = f1_score(true_cat.numpy(), pred_cat.numpy(), average="macro", zero_division=0)
        if classes is not None and key in classes:
            metrics[f"{key}_report"] = classification_report(
                true_cat.numpy(), pred_cat.numpy(), labels=list(range(len(classes[key]))),
                target_names=classes[key], zero_division=0,
            )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/from_embeddings"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.0,
                         help="stddev of Gaussian noise added independently to each page's embedding on every "
                              "training forward pass (not eval) - simulates the effect of the real pipeline's "
                              "per-page random image augmentation, without needing raw images. 0 = off.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    embeddings = np.load(args.embeddings)
    manifest = pd.read_csv(args.manifest, sep="\t")
    print(f"{embeddings.shape[0]} pages, {manifest['pdf_id'].nunique()} PDFs, embed_dim={embeddings.shape[1]}")

    doctype_classes = build_label_vocab(manifest, "split", "document_type")
    layout_classes = build_label_vocab(manifest, "split", "layout_type")
    functional_classes = build_label_vocab(manifest, "split", "functional_category")
    print(f"{len(doctype_classes)} doctype, {len(layout_classes)} layout, {len(functional_classes)} functional classes")

    def make_ds(split: str) -> EmbeddingSequenceDataset:
        return EmbeddingSequenceDataset(embeddings, manifest, split, doctype_classes, layout_classes, functional_classes)

    train_ds, val_ds, test_ds = make_ds("train"), make_ds("val"), make_ds("test")
    print(f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test PDFs")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    train_rows = manifest[manifest["split"] == "train"]
    n_pos = (train_rows["start_page"].astype(str).str.strip().str.lower() == "yes").sum()
    n_total = len(train_rows)
    start_pos_weight = max(1.0, (n_total - n_pos) / max(1, n_pos))
    print(f"start-page positive rate: {n_pos / max(1, n_total):.2f} (pos_weight={start_pos_weight:.2f})")

    seq_model = SequenceContextModel(
        embed_dim=embeddings.shape[1], num_doctype=len(doctype_classes), num_layout=len(layout_classes),
        num_functional=len(functional_classes), n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(seq_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        seq_model.train()
        running = {"total": 0.0, "start": 0.0, "doctype": 0.0, "layout": 0.0, "functional": 0.0}
        n_batches = 0
        for batch in train_loader:
            embeddings_batch = batch["embeddings"].to(device)
            if args.noise_std > 0:
                embeddings_batch = embeddings_batch + torch.randn_like(embeddings_batch) * args.noise_std
            padding_mask = batch["padding_mask"].to(device)
            out = seq_model(embeddings_batch, padding_mask, true_start_page=batch["start"].to(device))
            losses = compute_losses(out, batch, device, start_pos_weight)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(seq_model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1
        scheduler.step()
        avg = {k: v / max(1, n_batches) for k, v in running.items()}

        val_metrics = evaluate(seq_model, val_loader, device)
        print(
            f"epoch {epoch:>3}/{args.epochs}  loss={avg['total']:.3f} "
            f"(start={avg['start']:.3f} doctype={avg['doctype']:.3f} layout={avg['layout']:.3f} "
            f"functional={avg['functional']:.3f})  "
            f"val: start_f1={val_metrics['start_f1']:.3f} doctype_f1={val_metrics['doctype_macro_f1']:.3f} "
            f"layout_f1={val_metrics['layout_macro_f1']:.3f} functional_f1={val_metrics['functional_macro_f1']:.3f}"
        )

    class_lists = {"doctype": doctype_classes, "layout": layout_classes, "functional": functional_classes}
    test_metrics = evaluate(seq_model, test_loader, device, classes=class_lists)
    print("\ntest metrics:")
    report_lines = []
    for k, v in test_metrics.items():
        if k.endswith("_report"):
            continue
        print(f"  {k}: {v:.3f}")
        report_lines.append(f"{k}: {v:.3f}")
    for k, v in test_metrics.items():
        if k.endswith("_report"):
            report_lines.append(f"\n--- {k} ---\n{v}")
    (args.out_dir / "test_report.txt").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
