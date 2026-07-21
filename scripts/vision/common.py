"""Shared utilities for the document-type classification pipeline
(scripts/vision/train.py).
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(image_size: int, train: bool, augment_strength: str = "moderate") -> transforms.Compose:
    """Augmentations chosen for the described material: varying paper colour and
    background, rotation/skew, stains/tears, mixed print/handwriting, uneven
    lighting. Colour jitter and random erasing matter more here than the mild
    crops/flips typical of natural-image pipelines - documents are not
    rotation- or flip-invariant in the usual sense (a form upside down is a
    different signal), so we keep rotation small and never flip.
    """
    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    strong = augment_strength == "strong"
    ops = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(degrees=8 if not strong else 12, fill=255),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2 if not strong else 0.35, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3 if not strong else 0.5, scale=(0.02, 0.12)),  # simulates stains/tears
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)


def _finalize_dataloaders(
    train_ds,
    get_eval_ds,
    classes: list[str],
    batch_size: int,
    num_workers: int,
    balance_classes: bool,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, list[str]]:
    """Builds the weighted train sampler (for class imbalance) and the
    val/test loaders from whatever Dataset objects the caller hands in, as
    long as they expose `.samples` (list of (path, label_idx)) the way
    torchvision.datasets.ImageFolder does."""
    sampler = None
    shuffle = True
    if balance_classes:
        counts = torch.zeros(len(classes))
        for _, label in train_ds.samples:
            counts[label] += 1
        weights = 1.0 / counts.clamp(min=1)
        sample_weights = [weights[label] for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers
    )

    def loader_for(split: str) -> DataLoader | None:
        ds = get_eval_ds(split)
        if ds is None or len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, loader_for("val"), loader_for("test"), classes


class ManifestImageDataset(Dataset):
    """A dataset defined by a TSV/CSV manifest with an image-path column and
    a label column, rather than requiring one directory per class. Exposes
    `.samples` / `.targets` the way torchvision.datasets.ImageFolder does,
    so it plugs into `_finalize_dataloaders` unchanged."""

    def __init__(self, rows: pd.DataFrame, image_root: Path, image_col: str, label_col: str,
                 transform, classes: list[str]):
        self.image_root = Path(image_root)
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = [
            (str(self.image_root / row[image_col]), self.class_to_idx[row[label_col]])
            for _, row in rows.iterrows()
            if row[label_col] in self.class_to_idx
        ]
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def assign_stratified_splits(labels: pd.Series, ratios=(0.7, 0.15, 0.15), seed: int = 0) -> pd.Series:
    """A 70/15/15 split, stratified per class; classes too small to appear in
    every split (e.g. singleton real-world labels) fall back to train-only."""
    rng = random.Random(seed)
    splits = pd.Series(index=labels.index, dtype=object)
    for label, idx in labels.groupby(labels).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            splits.loc[idx] = "train"
            continue
        n_train = max(1, round(n * ratios[0]))
        n_val = max(1, round(n * ratios[1])) if n - n_train >= 2 else 0
        assigned = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        splits.loc[idx] = assigned
    return splits


def build_dataloaders_from_manifest(
    manifest_path: Path,
    image_root: Path,
    image_size: int,
    batch_size: int,
    image_col: str = "image",
    label_col: str = "label",
    split_col: str = "split",
    augment_strength: str = "moderate",
    num_workers: int = 2,
    balance_classes: bool = True,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, list[str]]:
    """Builds train/val/test loaders from a TSV/CSV manifest with one row
    per image: an image-path column (resolved relative to `image_root`) and
    a label column. If `split_col` isn't present, a stratified 70/15/15
    train/val/test split is assigned automatically and a warning is printed -
    real annotation exports like this project's merged_annotations.tsv won't
    have a split column, only image path + label."""
    sep = "\t" if str(manifest_path).endswith(".tsv") else ","
    manifest = pd.read_csv(manifest_path, sep=sep)

    if split_col not in manifest.columns:
        manifest[split_col] = assign_stratified_splits(manifest[label_col], seed=seed)
        print(f"No '{split_col}' column in {manifest_path} - assigned a stratified "
              f"70/15/15 train/val/test split automatically (seed={seed}).")

    classes = sorted(manifest[label_col].dropna().unique())

    train_ds = ManifestImageDataset(
        manifest[manifest[split_col] == "train"], image_root, image_col, label_col,
        build_transforms(image_size, train=True, augment_strength=augment_strength), classes=classes,
    )

    def get_eval_ds(split: str):
        rows = manifest[manifest[split_col] == split]
        if rows.empty:
            return None
        return ManifestImageDataset(
            rows, image_root, image_col, label_col, build_transforms(image_size, train=False), classes=classes
        )

    return _finalize_dataloaders(train_ds, get_eval_ds, classes, batch_size, num_workers, balance_classes)
