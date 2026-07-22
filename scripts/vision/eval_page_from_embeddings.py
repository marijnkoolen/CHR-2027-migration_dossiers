"""
Trains a simple frozen-backbone page classifier directly on precomputed
per-page embeddings (see precompute_embeddings.py) - the page-mode
equivalent of train_from_embeddings.py's sequence-mode test. Each row is
treated as an independent example (no PDF/page-order grouping at all),
matching what train.py's `--mode page --unfreeze-image-blocks 0` does,
just without needing live images or a live backbone.

Important scope limit: since the embeddings were precomputed once with a
frozen backbone, this can only measure the frozen-backbone case for page
mode - it cannot test whether *unfreezing* the backbone helps or hurts
there, the way the sequence-mode comparison did. That comparison needs a
live run with real images (there's no backbone in the loop here to
fine-tune at all - the classifier head is the only thing being trained).

Usage:
    python scripts/vision/eval_page_from_embeddings.py \\
        --embeddings data/embeddings_vision/embeddings.npy \\
        --manifest data/embeddings_vision/embeddings_manifest.tsv

    # single target instead of all four:
    python scripts/vision/eval_page_from_embeddings.py \\
        --embeddings data/embeddings_vision/embeddings.npy \\
        --manifest data/embeddings_vision/embeddings_manifest.tsv \\
        --target document_type
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from common import pick_device

TARGETS = ["document_type", "layout_type", "functional_category", "start_page"]


def run_target(
    embeddings: np.ndarray, manifest: pd.DataFrame, target: str, device: torch.device,
    epochs: int = 15, batch_size: int = 64, lr: float = 1e-3,
) -> dict:
    train_rows = manifest[manifest["split"] == "train"]
    classes = sorted(train_rows[target].dropna().unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def make_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        rows = manifest[manifest["split"] == split]
        rows = rows[rows[target].isin(class_to_idx)]
        X = torch.from_numpy(embeddings[rows["row_id"].to_numpy()]).float()
        y = torch.tensor([class_to_idx[v] for v in rows[target]])
        return X, y

    X_train, y_train = make_split("train")
    X_val, y_val = make_split("val")
    X_test, y_test = make_split("test")

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
    with torch.no_grad():
        test_preds = model(X_test.to(device)).argmax(dim=1).cpu()
    test_acc = (test_preds == y_test).float().mean().item()
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    report = classification_report(
        y_test, test_preds, labels=list(range(len(classes))), target_names=classes, zero_division=0
    )
    return {"n_classes": len(classes), "test_accuracy": test_acc, "test_macro_f1": test_f1, "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, default=None, help="default: test all four")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    embeddings = np.load(args.embeddings)
    manifest = pd.read_csv(args.manifest, sep="\t")
    print(f"{embeddings.shape[0]} pages, embed_dim={embeddings.shape[1]}")

    for target in [args.target] if args.target else TARGETS:
        result = run_target(embeddings, manifest, target, device, args.epochs, args.batch_size, args.lr)
        print(
            f"\n=== {target} ({result['n_classes']} classes) ===\n"
            f"test_accuracy={result['test_accuracy']:.3f}  test_macro_f1={result['test_macro_f1']:.3f}"
        )
        print(result["report"])


if __name__ == "__main__":
    main()
