"""
Trains SequenceContextModel end-to-end on precomputed page embeddings (see
precompute_embeddings.py), with a trainable projection layer in front of it -
the fix for the majority-class collapse (see models.py's PageEmbedder
docstring): SequenceContextModel's heads scale with embed_dim, so a raw
concatenated multimodal embedding badly overparameterizes them against a
training set of a few dozen PDFs:

    facebook/dinov2-small (384) + xlm-roberta-base  (768) = 1152-dim (efficient)
    dit-large-finetuned-rvlcdip (1024) + xlm-roberta-large (768) = 1792-dim (quality)

both well past the 1024-dim DiT-large-alone case that was already shown to
collapse and get fixed by projecting to 384 (matching DINOv2-small, the one
config that trains cleanly without any projection at all). This script
applies the same fix to the multimodal embeddings.

Since precompute_embeddings.py always freezes the backbone, training here
needs no images/text at all: only a from-scratch projection (LayerNorm ->
Linear -> GELU, exactly what MultimodalPageEmbedder/PageEmbedder build
internally for `project_to`) and SequenceContextModel are trained. Once
training finishes, the trained projection is folded into a freshly
constructed real embedder (its backbone weights are unchanged from the
pretrained checkpoint, since they were frozen the whole time) so the saved
sequence_model.pt is a complete, self-contained checkpoint - loadable by
predict.py exactly as if train.py --mode sequence had produced it.

Usage:
    # efficient (DINOv2-small + XLM-R-base)
    python scripts/classification/train_sequence_from_embeddings.py \\
        --embeddings ../../data/embeddings_multimodal/embeddings.npy \\
        --manifest ../../data/embeddings_multimodal/embeddings_manifest.tsv \\
        --image-backbone facebook/dinov2-small --text-backbone xlm-roberta-base \\
        --n-heads 4 --n-layers 2 \\
        --out-dir ../../runs/sequence_multimodal_efficient

    # quality (DiT-large + XLM-R-large)
    python scripts/classification/train_sequence_from_embeddings.py \\
        --embeddings ../../data/embeddings_multimodal_dit/embeddings.npy \\
        --manifest ../../data/embeddings_multimodal_dit/embeddings_manifest.tsv \\
        --image-backbone microsoft/dit-large-finetuned-rvlcdip --text-backbone xlm-roberta-large \\
        --n-heads 8 --n-layers 4 \\
        --out-dir ../../runs/sequence_multimodal_quality
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from common import format_confusion_matrix, pick_device
from models import MultimodalPageEmbedder, PageEmbedder
from sequence_model import SequenceContextModel
from train_from_embeddings import EmbeddingSequenceDataset, build_label_vocab, collate, compute_losses

TARGET_METRIC_KEY = {
    "start_page": "start_f1", "document_type": "doctype_macro_f1",
    "layout_type": "layout_macro_f1", "functional_category": "functional_macro_f1",
}


class ProjectedSequenceModel(nn.Module):
    """A trainable projection (matching what PageEmbedder/MultimodalPageEmbedder
    build for `project_to`) in front of SequenceContextModel, so it can be
    trained directly on raw precomputed embeddings and later folded into a
    real embedder's `.projection` submodule."""

    def __init__(self, raw_dim: int, project_to: int, num_doctype: int, num_layout: int, num_functional: int,
                 n_heads: int, n_layers: int, doc_classification: str = "late"):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(raw_dim), nn.Linear(raw_dim, project_to), nn.GELU())
        self.seq_model = SequenceContextModel(
            embed_dim=project_to, num_doctype=num_doctype, num_layout=num_layout,
            num_functional=num_functional, n_heads=n_heads, n_layers=n_layers,
            doc_classification=doc_classification,
        )

    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor, true_start_page=None) -> dict:
        return self.seq_model(self.projection(embeddings), padding_mask, true_start_page)


@torch.no_grad()
def evaluate(model: ProjectedSequenceModel, loader: DataLoader, device, classes: dict[str, list[str]] | None = None) -> dict:
    model.eval()
    start_true, start_pred = [], []
    task_true = {"doctype": [], "layout": [], "functional": []}
    task_pred = {"doctype": [], "layout": [], "functional": []}

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        out = model(embeddings, padding_mask, true_start_page=None)
        valid = (~padding_mask).cpu()

        start_true.append(batch["start"][valid])
        start_pred.append((torch.sigmoid(out["start_logits"]).cpu() > 0.5).float()[valid])
        for key in task_true:
            target = batch[key][valid]
            preds = out[f"{key}_logits"].argmax(dim=-1).cpu()[valid]
            keep = target != -100
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
        metrics["start_confusion_matrix"] = confusion_matrix(start_true_cat, start_pred_cat, labels=[0, 1]).tolist()

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
            labels = list(range(len(classes[key])))
            metrics[f"{key}_report"] = classification_report(
                true_cat.numpy(), pred_cat.numpy(), labels=labels, target_names=classes[key], zero_division=0,
            )
            metrics[f"{key}_confusion_matrix"] = confusion_matrix(true_cat.numpy(), pred_cat.numpy(), labels=labels).tolist()
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=["vision", "multimodal"], default="multimodal")
    parser.add_argument("--image-backbone", default="facebook/dinov2-small")
    parser.add_argument("--text-backbone", default="xlm-roberta-base")
    parser.add_argument("--image-size", type=int, default=224, help="stored in model_config.json only - the "
                         "backbone here is frozen/precomputed, so this doesn't affect training")
    parser.add_argument("--max-text-length", type=int, default=256, help="stored in model_config.json only")
    parser.add_argument("--project-to", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target", nargs="+", default=["start_page", "document_type", "layout_type", "functional_category"],
                         choices=list(TARGET_METRIC_KEY), help="which head(s) determine the saved 'best' checkpoint")
    parser.add_argument("--doc-classification", choices=["early", "late"], default="late",
                         help="how doctype/layout/functional use document segments - see sequence_model.py's "
                              "module docstring. 'late' (default): each page classified from its own "
                              "contextualized state, not pooled with document-mates. 'early': segment's "
                              "contextualized states mean-pooled first, one shared input per document.")
    parser.add_argument("--loss-weight-start", type=float, default=1.0)
    parser.add_argument("--loss-weight-doctype", type=float, default=1.0,
                         help="the four task losses are summed unweighted by default - document_type's raw loss "
                              "tends to dominate (many classes, several with only a handful of training "
                              "examples), so downweighting it (e.g. 0.3) tests whether that dominance in the "
                              "shared gradient is holding the other heads back.")
    parser.add_argument("--loss-weight-layout", type=float, default=1.0)
    parser.add_argument("--loss-weight-functional", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}")

    embeddings = np.load(args.embeddings)
    manifest = pd.read_csv(args.manifest, sep="\t")
    raw_dim = embeddings.shape[1]
    print(f"{embeddings.shape[0]} pages, {manifest['pdf_id'].nunique()} PDFs, raw_dim={raw_dim} -> project_to={args.project_to}")

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

    loss_weights = {
        "start": args.loss_weight_start, "doctype": args.loss_weight_doctype,
        "layout": args.loss_weight_layout, "functional": args.loss_weight_functional,
    }
    if any(w != 1.0 for w in loss_weights.values()):
        print(f"loss weights: {loss_weights}")

    model = ProjectedSequenceModel(
        raw_dim, args.project_to, len(doctype_classes), len(layout_classes), len(functional_classes),
        args.n_heads, args.n_layers, doc_classification=args.doc_classification,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_metric, best_state = -1.0, None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "start": 0.0, "doctype": 0.0, "layout": 0.0, "functional": 0.0}
        n_batches = 0
        for batch in train_loader:
            embeddings_batch = batch["embeddings"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            out = model(embeddings_batch, padding_mask, true_start_page=batch["start"].to(device))
            losses = compute_losses(out, batch, device, start_pos_weight, loss_weights=loss_weights)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1
        scheduler.step()
        avg = {k: v / max(1, n_batches) for k, v in running.items()}

        val_metrics = evaluate(model, val_loader, device)
        tracked = sum(val_metrics[TARGET_METRIC_KEY[t]] for t in args.target)
        print(
            f"epoch {epoch:>3}/{args.epochs}  loss={avg['total']:.3f} "
            f"(start={avg['start']:.3f} doctype={avg['doctype']:.3f} layout={avg['layout']:.3f} "
            f"functional={avg['functional']:.3f})  "
            f"val: start_f1={val_metrics['start_f1']:.3f} doctype_f1={val_metrics['doctype_macro_f1']:.3f} "
            f"layout_f1={val_metrics['layout_macro_f1']:.3f} functional_f1={val_metrics['functional_macro_f1']:.3f}"
            f"  [tracked ({'+'.join(args.target)})={tracked:.3f}]"
        )
        history.append({
            "epoch": epoch,
            "train_loss": avg["total"], "train_loss_start": avg["start"], "train_loss_doctype": avg["doctype"],
            "train_loss_layout": avg["layout"], "train_loss_functional": avg["functional"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "tracked": tracked,
        })
        if tracked > best_metric:
            best_metric = tracked
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (args.out_dir / "classes.json").write_text(json.dumps({
        "document_type": doctype_classes, "layout_type": layout_classes, "functional_category": functional_classes,
    }, indent=2))

    # Fold the trained projection into a real, freshly-loaded embedder (its
    # backbone weights are exactly the pretrained checkpoint - unchanged,
    # since precompute_embeddings.py always froze it) so sequence_model.pt is
    # a self-contained checkpoint predict.py can load directly.
    multimodal = args.modality == "multimodal"
    if multimodal:
        embedder = MultimodalPageEmbedder(
            args.image_backbone, args.text_backbone, max_text_length=args.max_text_length,
            project_to=args.project_to, device=device,
        ).to(device)
    else:
        embedder = PageEmbedder(args.image_backbone, project_to=args.project_to, device=device).to(device)
    embedder.projection.load_state_dict(model.projection.state_dict())

    torch.save({"embedder": embedder.state_dict(), "seq_model": model.seq_model.state_dict()},
               args.out_dir / "sequence_model.pt")
    (args.out_dir / "model_config.json").write_text(json.dumps({
        "modality": args.modality,
        "image_backbone": args.image_backbone,
        "text_backbone": args.text_backbone if multimodal else None,
        "image_size": args.image_size,
        "max_text_length": args.max_text_length,
        "project_to": args.project_to,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "doc_classification": args.doc_classification,
        "embed_dim": embedder.embed_dim,
    }, indent=2))

    class_lists = {"doctype": doctype_classes, "layout": layout_classes, "functional": functional_classes}

    # Training-set evaluation (self-predicted segmentation): if the model
    # can't even fit the PDFs it trained on, that's an optimization/capacity
    # problem, not a generalization one - same diagnostic train.py runs.
    train_metrics = evaluate(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
                              device, classes=class_lists)
    print("\ntrain-set metrics:")
    train_report_lines = ["train-set (self-predicted segmentation) metrics:", ""]
    for key in ["start_precision", "start_recall", "start_f1", "doctype_accuracy", "doctype_macro_f1",
                "layout_accuracy", "layout_macro_f1", "functional_accuracy", "functional_macro_f1"]:
        print(f"  {key}: {train_metrics[key]:.3f}")
        train_report_lines.append(f"{key}: {train_metrics[key]:.3f}")
    for key in ("start_report", "doctype_report", "layout_report", "functional_report"):
        if key in train_metrics:
            train_report_lines.append(f"\n--- {key} ---\n{train_metrics[key]}")
    (args.out_dir / "train_set_report.txt").write_text("\n".join(train_report_lines))

    test_metrics = evaluate(model, test_loader, device, classes=class_lists)
    print("\ntest metrics:")
    report_lines = [f"tracked targets: {', '.join(args.target)}", ""]
    for k, v in test_metrics.items():
        if k.endswith("_report") or k.endswith("_confusion_matrix"):
            continue
        print(f"  {k}: {v:.3f}")
        report_lines.append(f"{k}: {v:.3f}")
    for k, v in test_metrics.items():
        if k.endswith("_report"):
            report_lines.append(f"\n--- {k} ---\n{v}")
    (args.out_dir / "test_report.txt").write_text("\n".join(report_lines))

    cm_json = {"start_page": {"labels": ["not_start_page", "start_page"], "matrix": test_metrics["start_confusion_matrix"]}}
    cm_text = [f"--- start_page ---\n{format_confusion_matrix(test_metrics['start_confusion_matrix'], ['not_start_page', 'start_page'])}"]
    target_names = {"doctype": "document_type", "layout": "layout_type", "functional": "functional_category"}
    for key, labels in class_lists.items():
        matrix = test_metrics.get(f"{key}_confusion_matrix")
        if matrix is None:
            continue
        cm_json[target_names[key]] = {"labels": labels, "matrix": matrix}
        cm_text.append(f"--- {target_names[key]} ---\n{format_confusion_matrix(matrix, labels)}")
    (args.out_dir / "confusion_matrices.json").write_text(json.dumps(cm_json, indent=2))
    (args.out_dir / "confusion_matrices.txt").write_text("\n\n".join(cm_text))

    print(f"\nWrote sequence_model.pt, model_config.json, classes.json, history.json, train_set_report.txt, "
          f"test_report.txt, confusion_matrices.{{json,txt}} to {args.out_dir}")


if __name__ == "__main__":
    main()
