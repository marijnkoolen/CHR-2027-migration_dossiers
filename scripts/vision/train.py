"""
Single entry point for training any combination of:

    --scenario  {efficient, quality}     backbone size / hyperparameter preset
    --mode      {page, sequence}         per-page classification vs whole-PDF sequence context
    --modality  {vision, multimodal}     image-only vs image + PageXML text
    --target    document_type | layout_type | functional_category | start_page
                    page mode: exactly one - the column to classify.
                    sequence mode: one or more (default: all four) - which
                    head(s) determine the saved "best" checkpoint and get
                    top billing in the printed summary. All four heads are
                    always trained together in sequence mode regardless of
                    --target, because start-page detection is what document
                    segmentation (and therefore the type/layout/functional
                    heads) is built on - dropping it would break them, not
                    just skip reporting it.

Replaces train_efficient.py, train_quality.py, train_multimodal.py and
train_sequence.py, which each covered one corner of this space. The
underlying building blocks are unchanged - this only wires them together
behind one CLI: common.py and multimodal_data.py for page-mode data/loaders,
sequence_data.py/sequence_model.py for sequence mode, models.py for the
backbones (PageEmbedder/MultimodalPageEmbedder/BackboneClassifier/
MultimodalBackboneClassifier).

Examples:
    # efficient, single page, vision-only, document type
    python scripts/vision/train.py --manifest data/dummy_sequences/manifest.tsv \\
        --scenario efficient --mode page --modality vision --target document_type

    # quality, whole-PDF sequence context, image+text, best checkpoint
    # tracked on start-page + document-type
    python scripts/vision/train.py --manifest data/dummy_sequences/manifest.tsv \\
        --scenario quality --mode sequence --modality multimodal \\
        --target start_page document_type

Any hyperparameter flag (--image-backbone, --batch-size, --epochs, ...) can
still be set explicitly to override the --scenario preset for just that one
value.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from transformers import AutoTokenizer

from common import build_dataloaders_from_manifest, build_transforms, pick_device
from models import (
    BackboneClassifier,
    MultimodalBackboneClassifier,
    MultimodalPageEmbedder,
    PageEmbedder,
    trainable_parameter_summary,
)
from multimodal_data import build_multimodal_dataloaders
from sequence_data import IGNORE_INDEX, PageSequenceDataset, build_label_vocab, make_pdf_collate_fn
from sequence_model import SequenceContextModel

TARGET_COLUMN_ARG = {
    "document_type": "doctype_col",
    "layout_type": "layout_col",
    "functional_category": "functional_col",
    "start_page": "start_col",
}
TARGET_METRIC_KEY = {
    "start_page": "start_f1",
    "document_type": "doctype_macro_f1",
    "layout_type": "layout_macro_f1",
    "functional_category": "functional_macro_f1",
}

PRESETS = {
    "efficient": dict(
        image_backbone="facebook/dinov2-small", text_backbone="xlm-roberta-base",
        unfreeze_image_blocks=2, unfreeze_text_layers=2, image_size=224, batch_size=32,
        epochs=15, lr=1e-3, lr_backbone=1e-4, lr_head=1e-3, augment_strength="moderate",
        max_text_length=256, tta_views=0, n_heads=4, n_layers=2,
    ),
    "quality": dict(
        image_backbone="microsoft/dit-large-finetuned-rvlcdip", text_backbone="xlm-roberta-large",
        unfreeze_image_blocks=1000, unfreeze_text_layers=1000, image_size=336, batch_size=8,
        epochs=30, lr=2e-5, lr_backbone=2e-5, lr_head=1e-3, augment_strength="strong",
        max_text_length=256, tta_views=5, n_heads=8, n_layers=4,
    ),
}


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def differential_param_groups(embedder_params, other_params, args) -> list[dict]:
    """embedder (backbone) params get a lower lr than the head/other params
    for the quality scenario (full fine-tune needs a gentler backbone lr);
    efficient uses one flat lr for both, expressed as the same value twice
    so this code path doesn't need a separate branch."""
    lr_backbone = args.lr_backbone if args.scenario == "quality" else args.lr
    lr_head = args.lr_head if args.scenario == "quality" else args.lr
    groups = [
        {"params": [p for p in embedder_params if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in other_params if p.requires_grad], "lr": lr_head},
    ]
    return [g for g in groups if g["params"]]


def class_weights_from_samples(samples, num_classes: int) -> torch.Tensor:
    counts = Counter(label for *_, label in samples)
    freq = torch.tensor([counts.get(i, 1) for i in range(num_classes)], dtype=torch.float)
    return freq.sum() / (num_classes * freq)


def resolve_amp(args, device: torch.device) -> bool:
    """bf16 autocast, not fp16: on Ampere+ (A10 included) bf16 has the same
    exponent range as fp32, so there's no overflow/GradScaler machinery
    needed - it's a straightforward memory/speed win. --amp auto only
    enables it on CUDA, where this is well-supported and tested; MPS/CPU
    autocast coverage is patchier, so those stay off unless forced with
    --amp on."""
    if args.amp == "on":
        return True
    if args.amp == "off":
        return False
    return device.type == "cuda"


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def format_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> str:
    """A readable aligned text grid (true label = row, predicted = column).
    Can get wide for many classes, but that's inherent to confusion
    matrices - fine once redirected to a file."""
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "true \\ pred"
    return df.to_string()


def _classification_metrics(preds: list[int], targets: list[int], classes: list[str]) -> dict:
    accuracy = sum(p == t for p, t in zip(preds, targets)) / max(1, len(targets))
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    report = classification_report(
        targets, preds, labels=list(range(len(classes))), target_names=classes, zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes)))).tolist()
    return {"accuracy": accuracy, "macro_f1": macro_f1, "report": report, "confusion_matrix": cm}


# --------------------------------------------------------------------------
# Page mode (train_efficient.py / train_quality.py / train_multimodal.py)
# --------------------------------------------------------------------------

def make_vision_forward(model, device):
    def forward(batch):
        images, targets = batch
        return model(images.to(device)), targets.to(device)
    return forward


def make_multimodal_forward(model, device):
    def forward(batch):
        images, input_ids, attention_mask, targets = batch
        logits = model(images.to(device), input_ids.to(device), attention_mask.to(device))
        return logits, targets.to(device)
    return forward


@torch.no_grad()
def evaluate_page_model(model, loader: DataLoader, device, classes: list[str], forward_fn, amp: bool = False) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        with autocast_context(device, amp):
            logits, targets = forward_fn(batch)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
    return _classification_metrics(all_preds, all_targets, classes)


@torch.no_grad()
def evaluate_page_model_tta(model, dataset, classes: list[str], device, image_size: int,
                             n_views: int, batch_size: int, amp: bool = False) -> dict:
    """Vision-only test-time augmentation: averages softmax over the plain
    view plus a few augmented ones. `dataset` must expose `.samples` as
    (path, ..., label_idx) tuples (both ImageFolder and MultimodalManifestDataset do)."""
    model.eval()
    tta_transform = build_transforms(image_size, train=True, augment_strength="moderate")
    plain_transform = build_transforms(image_size, train=False)
    samples = dataset.samples

    all_preds, all_targets = [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        targets = [s[-1] for s in batch]
        probs_sum = None
        for transform in [plain_transform] + [tta_transform] * (n_views - 1):
            images = torch.stack([transform(default_loader(s[0])) for s in batch]).to(device)
            with autocast_context(device, amp):
                logits = model(images)
            probs = F.softmax(logits, dim=1).cpu()
            probs_sum = probs if probs_sum is None else probs_sum + probs
        all_preds.extend((probs_sum / n_views).argmax(dim=1).tolist())
        all_targets.extend(targets)
    return _classification_metrics(all_preds, all_targets, classes)


def train_page_model(model, train_loader, val_loader, test_loader, classes, device, epochs,
                      optimizer, scheduler, forward_fn, out_dir: Path,
                      class_weights: torch.Tensor | None = None, tta_views: int = 0, image_size: int = 224,
                      amp: bool = False):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    weight_tensor = class_weights.to(device) if class_weights is not None else None
    best_metric, best_state = -1.0, None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            with autocast_context(device, amp):
                logits, targets = forward_fn(batch)
                loss = F.cross_entropy(logits, targets, weight=weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()

        entry = {"epoch": epoch, "train_loss": running_loss / max(1, n_batches)}
        msg = f"epoch {epoch:>3}/{epochs}  loss={entry['train_loss']:.4f}"
        tracked = -entry["train_loss"]
        if val_loader is not None:
            metrics = evaluate_page_model(model, val_loader, device, classes, forward_fn, amp=amp)
            entry["val_accuracy"] = metrics["accuracy"]
            entry["val_macro_f1"] = metrics["macro_f1"]
            msg += f"  val_acc={metrics['accuracy']:.3f} val_macro_f1={metrics['macro_f1']:.3f}"
            tracked = metrics["macro_f1"]
        print(msg)
        history.append(entry)

        if tracked > best_metric:
            best_metric = tracked
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), out_dir / "model.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2))

    if test_loader is not None:
        if tta_views and tta_views > 1:
            metrics = evaluate_page_model_tta(
                model, test_loader.dataset, classes, device, image_size, tta_views, test_loader.batch_size, amp=amp
            )
            print(f"\ntest (TTA x{tta_views}) accuracy={metrics['accuracy']:.3f}  macro-F1={metrics['macro_f1']:.3f}")
        else:
            metrics = evaluate_page_model(model, test_loader, device, classes, forward_fn, amp=amp)
            print(f"\ntest accuracy={metrics['accuracy']:.3f}  macro-F1={metrics['macro_f1']:.3f}")
        print(metrics["report"])
        (out_dir / "test_report.txt").write_text(metrics["report"])
        (out_dir / "confusion_matrix.json").write_text(
            json.dumps({"labels": classes, "matrix": metrics["confusion_matrix"]}, indent=2)
        )
        (out_dir / "confusion_matrix.txt").write_text(
            format_confusion_matrix(metrics["confusion_matrix"], classes)
        )


def run_page(args, target_column: str):
    device = pick_device(args.device)
    amp = resolve_amp(args, device)
    print(f"device: {device}  amp(bf16): {amp}")

    if args.modality == "vision":
        train_loader, val_loader, test_loader, classes = build_dataloaders_from_manifest(
            args.manifest, args.image_root, args.image_size, args.batch_size,
            image_col=args.image_col, label_col=target_column, split_col=args.split_col,
            augment_strength=args.augment_strength, seed=args.seed,
        )
        model = BackboneClassifier(
            args.image_backbone, len(classes), unfreeze_last_n_blocks=args.unfreeze_image_blocks
        ).to(device)
        forward_fn = make_vision_forward(model, device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.text_backbone)
        train_loader, val_loader, test_loader, classes = build_multimodal_dataloaders(
            args.manifest, args.image_root, target_column, tokenizer,
            image_col=args.image_col, pagexml_col=args.pagexml_col, split_col=args.split_col,
            image_size=args.image_size, batch_size=args.batch_size, max_text_length=args.max_text_length,
            augment_strength=args.augment_strength, seed=args.seed,
        )
        model = MultimodalBackboneClassifier(
            args.image_backbone, len(classes), text_backbone=args.text_backbone,
            unfreeze_image_blocks=args.unfreeze_image_blocks, unfreeze_text_layers=args.unfreeze_text_layers,
            max_text_length=args.max_text_length, project_to=args.project_to,
        ).to(device)
        forward_fn = make_multimodal_forward(model, device)

    print(f"{len(classes)} classes, {len(train_loader.dataset)} train pages, target={target_column!r}")
    print(trainable_parameter_summary(model))

    groups = differential_param_groups(model.embedder.parameters(), model.head.parameters(), args)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = (
        class_weights_from_samples(train_loader.dataset.samples, len(classes))
        if args.scenario == "quality" else None
    )

    use_tta = args.modality == "vision" and args.tta_views and args.tta_views > 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_page_model(
        model, train_loader, val_loader, test_loader, classes, device, args.epochs,
        optimizer, scheduler, forward_fn, args.out_dir,
        class_weights=class_weights, tta_views=(args.tta_views if use_tta else 0), image_size=args.image_size,
        amp=amp,
    )


# --------------------------------------------------------------------------
# Sequence mode (train_sequence.py)
# --------------------------------------------------------------------------

def embed_pages(embedder, batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["images_flat"].to(device)
    if "input_ids_flat" in batch:
        embeds_flat = embedder(images, batch["input_ids_flat"].to(device), batch["attention_mask_flat"].to(device))
    else:
        embeds_flat = embedder(images)
    B, T = batch["padding_mask"].shape
    embeddings = torch.zeros(B, T, embedder.embed_dim, device=device, dtype=embeds_flat.dtype)
    embeddings[batch["batch_index"], batch["time_index"]] = embeds_flat
    return embeddings, batch["padding_mask"].to(device)


def compute_sequence_losses(out: dict, batch: dict, device, start_pos_weight: float) -> dict:
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
def evaluate_sequence(embedder, seq_model, loader: DataLoader, device,
                       classes: dict[str, list[str]] | None = None, amp: bool = False,
                       teacher_forced: bool = False) -> dict:
    """classes (optional): {"doctype": [...], "layout": [...], "functional": [...]}
    - when given, a full sklearn classification_report is also computed per
    task (and for start-page). Only pass this for the final test evaluation,
    not the per-epoch val one - it's needlessly expensive/verbose otherwise.

    teacher_forced: segment using ground-truth start-page labels instead of
    the model's own predictions - not achievable at real inference time, but
    useful as a diagnostic to check whether low doctype/layout/functional
    scores are caused by imperfect self-predicted segmentation (see
    --eval-teacher-forced). start_* metrics are identical either way, since
    the start-page head's own predictions never depend on this flag."""
    embedder.eval()
    seq_model.eval()

    start_true, start_pred = [], []
    task_true = {"doctype": [], "layout": [], "functional": []}
    task_pred = {"doctype": [], "layout": [], "functional": []}

    for batch in loader:
        with autocast_context(device, amp):
            embeddings, padding_mask = embed_pages(embedder, batch, device)
            true_start = batch["start"].to(device) if teacher_forced else None
            out = seq_model(embeddings, padding_mask, true_start_page=true_start)
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
            metrics[f"{key}_report"] = classification_report(
                true_cat.numpy(), pred_cat.numpy(), labels=list(range(len(classes[key]))),
                target_names=classes[key], zero_division=0,
            )
            metrics[f"{key}_confusion_matrix"] = confusion_matrix(
                true_cat.numpy(), pred_cat.numpy(), labels=list(range(len(classes[key])))
            ).tolist()

    return metrics


def run_sequence(args, targets: list[str]):
    device = pick_device(args.device)
    amp = resolve_amp(args, device)
    print(f"device: {device}  amp(bf16): {amp}")

    manifest = pd.read_csv(args.manifest, sep="\t" if str(args.manifest).endswith(".tsv") else ",")

    doctype_classes = build_label_vocab(manifest, args.split_col, args.doctype_col)
    layout_classes = build_label_vocab(manifest, args.split_col, args.layout_col)
    functional_classes = build_label_vocab(manifest, args.split_col, args.functional_col)
    print(f"{len(doctype_classes)} doctype classes, {len(layout_classes)} layout classes, "
          f"{len(functional_classes)} functional classes")

    multimodal = args.modality == "multimodal"
    tokenizer = AutoTokenizer.from_pretrained(args.text_backbone) if multimodal else None

    def make_dataset(split: str, train: bool) -> PageSequenceDataset:
        return PageSequenceDataset(
            manifest, split, args.image_root, build_transforms(args.image_size, train=train),
            doctype_classes, layout_classes, functional_classes,
            pdf_col=args.pdf_col, page_col=args.page_col, image_col=args.image_col,
            doctype_col=args.doctype_col, layout_col=args.layout_col, functional_col=args.functional_col,
            start_col=args.start_col, split_col=args.split_col,
            pagexml_col=args.pagexml_col if multimodal else None,
        )

    train_ds = make_dataset("train", train=True)
    val_ds = make_dataset("val", train=False)
    test_ds = make_dataset("test", train=False)
    print(f"{len(train_ds)} train PDFs, {len(val_ds)} val PDFs, {len(test_ds)} test PDFs")

    collate = make_pdf_collate_fn(tokenizer, args.max_text_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    n_pos = sum(row[args.start_col].strip().lower() == "yes" for _, row in
                manifest[manifest[args.split_col] == "train"].iterrows())
    n_total = (manifest[args.split_col] == "train").sum()
    start_pos_weight = max(1.0, (n_total - n_pos) / max(1, n_pos))
    print(f"start-page positive rate: {n_pos / max(1, n_total):.2f} (pos_weight={start_pos_weight:.2f})")

    if multimodal:
        embedder = MultimodalPageEmbedder(
            image_backbone=args.image_backbone, text_backbone=args.text_backbone,
            unfreeze_image_blocks=args.unfreeze_image_blocks, unfreeze_text_layers=args.unfreeze_text_layers,
            max_text_length=args.max_text_length, project_to=args.project_to,
        ).to(device)
    else:
        embedder = PageEmbedder(args.image_backbone, unfreeze_last_n_blocks=args.unfreeze_image_blocks).to(device)

    seq_model = SequenceContextModel(
        embed_dim=embedder.embed_dim, num_doctype=len(doctype_classes), num_layout=len(layout_classes),
        num_functional=len(functional_classes), n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device)
    print(trainable_parameter_summary(embedder))

    embedder_params = [p for p in embedder.parameters() if p.requires_grad]
    seq_params = list(seq_model.parameters())
    groups = differential_param_groups(embedder_params, seq_params, args)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    trainable_params = [p for p in embedder_params if p.requires_grad] + seq_params

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_metric, best_state = -1.0, None
    history = []

    for epoch in range(1, args.epochs + 1):
        embedder.train()
        seq_model.train()
        running = {"total": 0.0, "start": 0.0, "doctype": 0.0, "layout": 0.0, "functional": 0.0}
        n_batches = 0

        for batch in train_loader:
            with autocast_context(device, amp):
                embeddings, padding_mask = embed_pages(embedder, batch, device)
                out = seq_model(embeddings, padding_mask, true_start_page=batch["start"].to(device))
                losses = compute_sequence_losses(out, batch, device, start_pos_weight)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()
        avg = {k: v / max(1, n_batches) for k, v in running.items()}

        val_metrics = evaluate_sequence(embedder, seq_model, val_loader, device, amp=amp)
        tracked = sum(val_metrics[TARGET_METRIC_KEY[t]] for t in targets)
        print(
            f"epoch {epoch:>3}/{args.epochs}  loss={avg['total']:.3f} "
            f"(start={avg['start']:.3f} doctype={avg['doctype']:.3f} layout={avg['layout']:.3f} "
            f"functional={avg['functional']:.3f})  "
            f"val: start_f1={val_metrics['start_f1']:.3f} doctype_f1={val_metrics['doctype_macro_f1']:.3f} "
            f"layout_f1={val_metrics['layout_macro_f1']:.3f} functional_f1={val_metrics['functional_macro_f1']:.3f}"
            f"  [tracked ({'+'.join(targets)})={tracked:.3f}]"
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
            best_state = {
                "embedder": {k: v.detach().cpu().clone() for k, v in embedder.state_dict().items()},
                "seq_model": {k: v.detach().cpu().clone() for k, v in seq_model.state_dict().items()},
            }

    if best_state is not None:
        embedder.load_state_dict(best_state["embedder"])
        seq_model.load_state_dict(best_state["seq_model"])
        torch.save(best_state, args.out_dir / "sequence_model.pt")

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (args.out_dir / "classes.json").write_text(json.dumps({
        "document_type": doctype_classes, "layout_type": layout_classes, "functional_category": functional_classes,
    }, indent=2))

    class_lists = {"doctype": doctype_classes, "layout": layout_classes, "functional": functional_classes}
    test_metrics = evaluate_sequence(embedder, seq_model, test_loader, device, classes=class_lists, amp=amp)

    print(f"\ntest metrics (tracked targets: {', '.join(targets)}):")
    report_lines = [f"tracked targets: {', '.join(targets)}", ""]
    for k, v in test_metrics.items():
        if k.endswith("_report") or k.endswith("_confusion_matrix"):
            continue
        print(f"  {k}: {v:.3f}")
        report_lines.append(f"{k}: {v:.3f}")
    for k, v in test_metrics.items():
        if k.endswith("_report"):
            report_lines.append(f"\n--- {k} ---\n{v}")
    (args.out_dir / "test_report.txt").write_text("\n".join(report_lines))

    cm_json = {"start_page": {"labels": ["not_start_page", "start_page"],
                               "matrix": test_metrics["start_confusion_matrix"]}}
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

    if args.eval_teacher_forced:
        oracle_metrics = evaluate_sequence(
            embedder, seq_model, test_loader, device, classes=class_lists, amp=amp, teacher_forced=True
        )
        compare_keys = [
            "start_f1", "doctype_accuracy", "doctype_macro_f1",
            "layout_accuracy", "layout_macro_f1", "functional_accuracy", "functional_macro_f1",
        ]
        print("\ndiagnostic: self-predicted vs ORACLE (ground-truth) segmentation "
              "- a big gap means low doctype/layout/functional scores are mostly "
              "caused by imperfect start-page segmentation, not those heads themselves:")
        print(f"  {'metric':<22} {'self-predicted':>15} {'oracle':>10}")
        diag_lines = [
            "ORACLE (ground-truth start-page) segmentation diagnostic.",
            "Not achievable at real inference time - compares against the self-predicted",
            "test metrics in test_report.txt. A big gap here means the doctype/layout/",
            "functional heads are being hurt mainly by imperfect self-predicted",
            "segmentation, not by those heads' own learned representations; a small gap",
            "means segmentation isn't the bottleneck. start_f1 is identical in both",
            "columns by construction - the start-page head's own predictions never",
            "depend on this flag, only the *other* heads' segment pooling does.",
            "",
            f"{'metric':<22} {'self-predicted':>15} {'oracle':>10}",
        ]
        for key in compare_keys:
            print(f"  {key:<22} {test_metrics[key]:>15.3f} {oracle_metrics[key]:>10.3f}")
            diag_lines.append(f"{key:<22} {test_metrics[key]:>15.3f} {oracle_metrics[key]:>10.3f}")
        for key in ("doctype_report", "layout_report", "functional_report"):
            if key in oracle_metrics:
                diag_lines.append(f"\n--- oracle {key} ---\n{oracle_metrics[key]}")
        (args.out_dir / "diagnostic_teacher_forced.txt").write_text("\n".join(diag_lines))
        print(f"\nWrote {args.out_dir / 'diagnostic_teacher_forced.txt'}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path(""))
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="default: runs/<scenario>_<mode>_<modality>_<target>")

    parser.add_argument("--scenario", choices=list(PRESETS), required=True)
    parser.add_argument("--mode", choices=["page", "sequence"], required=True)
    parser.add_argument("--modality", choices=["vision", "multimodal"], required=True)
    parser.add_argument(
        "--target", nargs="+", choices=list(TARGET_COLUMN_ARG), default=None,
        help="page mode: exactly one. sequence mode: default is all four (see module docstring).",
    )

    # Hyperparameters default to None so --scenario fills them in; pass any
    # of these explicitly to override just that one value.
    parser.add_argument("--image-backbone", default=None)
    parser.add_argument("--text-backbone", default=None)
    parser.add_argument("--unfreeze-image-blocks", type=int, default=None)
    parser.add_argument("--unfreeze-text-layers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="pages per batch (page mode) or PDFs per batch (sequence mode)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="flat lr, used when --scenario efficient")
    parser.add_argument("--lr-backbone", type=float, default=None, help="used when --scenario quality")
    parser.add_argument("--lr-head", type=float, default=None, help="used when --scenario quality")
    parser.add_argument("--augment-strength", choices=["moderate", "strong"], default=None)
    parser.add_argument("--max-text-length", type=int, default=None)
    parser.add_argument("--tta-views", type=int, default=None, help="page+vision mode only")
    parser.add_argument("--n-heads", type=int, default=None, help="sequence mode only")
    parser.add_argument("--n-layers", type=int, default=None, help="sequence mode only")
    parser.add_argument("--project-to", type=int, default=None,
                         help="multimodal only: project the fused embedding to this size (default: no projection)")

    # Manifest column names (defaults match this project's real annotation schema)
    parser.add_argument("--pdf-col", default="dossier_name")
    parser.add_argument("--page-col", default="page_num")
    parser.add_argument("--image-col", default="img_path")
    parser.add_argument("--pagexml-col", default="text_page")
    parser.add_argument("--doctype-col", default="doc_type")
    parser.add_argument("--layout-col", default="Layout Type Classification")
    parser.add_argument("--functional-col", default="func_label")
    parser.add_argument("--start-col", default="is_start")
    parser.add_argument("--split-col", default="split")

    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                         help="mixed precision (bf16). auto = on for CUDA, off otherwise")
    parser.add_argument("--eval-teacher-forced", action="store_true",
                         help="sequence mode only: also evaluate the test set with ground-truth "
                              "start-page segmentation (oracle), to check how much self-predicted "
                              "segmentation is hurting doctype/layout/functional - see module docstring")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser


def apply_scenario_preset(args: argparse.Namespace) -> None:
    for key, value in PRESETS[args.scenario].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.target is None:
        if args.mode == "page":
            raise SystemExit("--target is required in --mode page (choose exactly one).")
        args.target = list(TARGET_COLUMN_ARG)
    if args.mode == "page" and len(args.target) != 1:
        raise SystemExit(f"--mode page needs exactly one --target, got {args.target}")
    return args.target


def main():
    args = build_arg_parser().parse_args()
    apply_scenario_preset(args)
    targets = resolve_targets(args)

    if args.out_dir is None:
        args.out_dir = Path("runs") / f"{args.scenario}_{args.mode}_{args.modality}_{'+'.join(targets)}"

    print(f"scenario={args.scenario}  mode={args.mode}  modality={args.modality}  target={targets}")

    if args.mode == "page":
        target_column = getattr(args, TARGET_COLUMN_ARG[targets[0]])
        run_page(args, target_column)
    else:
        run_sequence(args, targets)


if __name__ == "__main__":
    main()
