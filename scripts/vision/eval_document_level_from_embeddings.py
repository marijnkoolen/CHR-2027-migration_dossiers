"""
Two experiments plus a proper end-to-end evaluation for representing a
whole document by just its start page, rather than every individual page
(plain page mode) or a whole-segment context pool (the sequence model).

Experiment 1 - "oracle upper bound": train the usual all-pages document-
type/layout/functional classifiers, but evaluate them only on the test
set's *true* start pages. No retraining - just a filtered evaluation.

Experiment 2 - "train on start pages only": retrain those same classifiers
using only the train set's start pages, so train and eval distributions
match, then evaluate on the same true-start-page test subset as
Experiment 1 for a direct comparison.

The harder part - evaluating on *predicted* (not ground-truth) start pages,
using a separately-trained start-page detector - reports two complementary
numbers rather than one, since scoring only "pages flagged as start" can
hide either failure mode:

  (A) detection-conditional accuracy: score document-type/layout/functional
      only on pages that are BOTH true start pages AND correctly predicted
      as start (true positives) - isolates classification quality from
      detection quality.
  (B) end-to-end document-level accuracy: for every TRUE document (a
      document = one true start page, since it's represented by just that
      page here), credit it only if its start page was BOTH correctly
      detected as a start AND correctly classified - a missed detection
      counts as a failure for that document, same as a misclassification
      would, since in a real pipeline a missed boundary means that document
      never gets classified at all.

Neither (A) nor (B) penalizes false positives (non-start pages wrongly
flagged as starts, which would split one real document into two) - the
detector's own precision is reported alongside instead of folding that into
either composite metric.

Usage:
    python scripts/vision/eval_document_level_from_embeddings.py \\
        --embeddings data/embeddings_vision/embeddings.npy \\
        --manifest data/embeddings_vision/embeddings_manifest.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from common import pick_device

DOC_TARGETS = ["document_type", "layout_type", "functional_category"]


def train_classifier(
    embeddings: np.ndarray, manifest: pd.DataFrame, target: str, device: torch.device,
    train_only_start_pages: bool = False, epochs: int = 15, batch_size: int = 64, lr: float = 1e-3,
) -> tuple[nn.Module, list[str]]:
    """LayerNorm+Linear classifier (matching BackboneClassifier's head) for
    `target`, optionally restricting training rows to true start pages.
    Model selection uses the (always full) val split."""
    train_rows = manifest[manifest["split"] == "train"]
    if train_only_start_pages:
        train_rows = train_rows[train_rows["_is_start"]]
    classes = sorted(train_rows[target].dropna().unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def make_xy(rows: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        rows = rows[rows[target].isin(class_to_idx)]
        X = torch.from_numpy(embeddings[rows["row_id"].to_numpy()]).float()
        y = torch.tensor([class_to_idx[v] for v in rows[target]])
        return X, y

    X_train, y_train = make_xy(train_rows)
    X_val, y_val = make_xy(manifest[manifest["split"] == "val"])

    embed_dim = embeddings.shape[1]
    model = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, len(classes))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    counts = torch.bincount(y_train, minlength=len(classes)).float()
    weights = (1.0 / counts.clamp(min=1))[y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, sampler=sampler)

    best_val_f1, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device)).argmax(dim=1).cpu()
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, classes


@torch.no_grad()
def predict(model: nn.Module, embeddings: np.ndarray, row_ids: np.ndarray, device: torch.device) -> np.ndarray:
    X = torch.from_numpy(embeddings[row_ids]).float().to(device)
    return model(X).argmax(dim=1).cpu().numpy()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    if len(y_true) == 0:
        return {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan")}
    accuracy = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(n_classes)))
    return {"n": len(y_true), "accuracy": float(accuracy), "macro_f1": float(macro_f1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    embeddings = np.load(args.embeddings)
    manifest = pd.read_csv(args.manifest, sep="\t")
    manifest["_is_start"] = manifest["start_page"].astype(str).str.strip().str.lower() == "yes"
    print(f"{len(manifest)} pages, embed_dim={embeddings.shape[1]}")

    test_rows = manifest[manifest["split"] == "test"]
    n_start_train = manifest[(manifest["split"] == "train") & manifest["_is_start"]].shape[0]
    n_train = (manifest["split"] == "train").sum()
    print(f"train: {n_train} pages, {n_start_train} start pages ({n_start_train / n_train:.1%})")

    # --- Start-page detector, needed for the predicted-start evaluations ---
    print("\nTraining start-page detector...")
    start_model, start_classes = train_classifier(
        embeddings, manifest, "start_page", device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    start_pred_idx = predict(start_model, embeddings, test_rows["row_id"].to_numpy(), device)
    start_pred = np.array(start_classes)[start_pred_idx] == "yes"
    start_true = test_rows["_is_start"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        start_true, start_pred, average="binary", zero_division=0
    )
    print(f"start-page detector on test set: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

    # --- Document-type/layout/functional classifiers: "all-pages" vs "start-only" training ---
    variants: dict[str, dict[str, tuple[nn.Module, list[str]]]] = {}
    for train_only_start in (False, True):
        label = "start-only" if train_only_start else "all-pages"
        variants[label] = {}
        for target in DOC_TARGETS:
            print(f"Training {target} ({label} training data)...")
            model, classes = train_classifier(
                embeddings, manifest, target, device, train_only_start_pages=train_only_start,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            )
            variants[label][target] = (model, classes)

    true_start_rows = test_rows[test_rows["_is_start"]]
    true_positive_mask = test_rows["_is_start"].to_numpy() & start_pred
    true_positive_rows = test_rows[true_positive_mask]

    def eval_rows(model, classes, rows, target) -> dict:
        rows = rows[rows[target].isin(classes)]
        y_true = np.array([classes.index(v) for v in rows[target]])
        y_pred = predict(model, embeddings, rows["row_id"].to_numpy(), device)
        return classification_metrics(y_true, y_pred, len(classes))

    print("\n" + "=" * 70)
    print("Experiment 1 & 2: oracle upper bound (true start pages, no detector)")
    print("=" * 70)
    for label in variants:
        print(f"\n--- {label} training data ---")
        for target in DOC_TARGETS:
            model, classes = variants[label][target]
            m = eval_rows(model, classes, true_start_rows, target)
            print(f"  {target:<22} n={m['n']:<4} accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")

    print("\n" + "=" * 70)
    print("(A) Detection-conditional: true start pages correctly predicted as start")
    print(f"    ({true_positive_mask.sum()} of {len(true_start_rows)} true start pages detected)")
    print("=" * 70)
    for label in variants:
        print(f"\n--- {label} training data ---")
        for target in DOC_TARGETS:
            model, classes = variants[label][target]
            m = eval_rows(model, classes, true_positive_rows, target)
            print(f"  {target:<22} n={m['n']:<4} accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")

    print("\n" + "=" * 70)
    print("(B) End-to-end document-level: every true document, credited only if")
    print("    its start page was both detected AND correctly classified")
    print("    (false positives - spurious detected splits - not penalized here;")
    print("     see detector precision above)")
    print("=" * 70)
    detected_row_ids = set(true_positive_rows["row_id"])
    for label in variants:
        print(f"\n--- {label} training data ---")
        for target in DOC_TARGETS:
            model, classes = variants[label][target]
            rows = true_start_rows[true_start_rows[target].isin(classes)]
            detected = rows["row_id"].isin(detected_row_ids).to_numpy()
            y_true = np.array([classes.index(v) for v in rows[target]])
            y_pred = predict(model, embeddings, rows["row_id"].to_numpy(), device)
            end_to_end_correct = detected & (y_true == y_pred)
            print(
                f"  {target:<22} n={len(rows):<4} detected={detected.mean():.3f}  "
                f"end_to_end_accuracy={end_to_end_correct.mean():.3f}"
            )


if __name__ == "__main__":
    main()
