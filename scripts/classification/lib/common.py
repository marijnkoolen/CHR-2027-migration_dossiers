"""Shared utilities for the document-type classification pipeline
(scripts/classification/train.py).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from markdown_text import extract_text as _extract_markdown_text
from pagexml import extract_text as _extract_pagexml_text


def get_text_extractor(text_source: str) -> Callable[[str], str]:
    """extract_text(path) -> str for --text-source ("pagexml" or
    "markdown") - shared by precompute_embeddings.py, train.py, and
    predict.py so all three resolve a text source identically. pagexml:
    PageXML transcriptions (lib/pagexml.py). markdown: Qwen2.5-VL
    markdown-mode OCR output, stripped of markup (lib/markdown_text.py) -
    see that module's docstring for why markup is stripped, and why
    markdown is this project's default text source."""
    if text_source == "markdown":
        return _extract_markdown_text
    if text_source == "pagexml":
        return _extract_pagexml_text
    raise ValueError(f"unknown text_source {text_source!r} - expected 'pagexml' or 'markdown'")


def validate_manifest_paths(
    manifest: pd.DataFrame, image_root: Path, image_col: str | None, text_col: str | None,
    allow_missing: bool = False, max_examples: int = 10,
) -> None:
    """Checks every non-null path in image_col and text_col (pass None for
    either to skip it - e.g. image_col=None for --modality text) resolves
    to an existing file under image_root, before any (slow) model loading
    or data extraction happens. Call this right after reading the manifest,
    in every script that reads raw images/text (train.py, precompute_
    embeddings.py) - NOT needed for train.py --cached-embeddings, which
    never touches raw files at all.

    Raises SystemExit listing example missing paths if any are found and
    allow_missing is False (the default). This exists because silently
    tolerating broken paths - which is what extract_text()'s "" fallback,
    combined with only a >50%-empty-text WARNING, previously allowed - can
    produce a manifest where every row's text is empty, and therefore every
    text embedding is near-identical, without erroring at all; this project
    has been bitten by exactly that failure mode more than once (see
    precompute_embeddings.py's module docstring), from different root
    causes each time (a wrong .txt/.xml path, and a wrong markdown OCR
    path) - a strict, load-time check catches the general problem instead
    of each specific instance of it after the fact.

    A NaN/missing *value* in a column (no path given at all) is not an
    error - some pages legitimately have no transcription (e.g. photos);
    only a given, non-null path that doesn't resolve to an existing file is
    treated as a mistake.

    allow_missing=True downgrades this to a warning and continues - for
    text_col, that's coherent (matches extract_text's own defined ""
    fallback for a missing page); for image_col there's no such fallback
    anywhere in this codebase, so allowing a missing image just defers the
    failure to a harder-to-diagnose crash later, during actual data
    loading, rather than avoiding it - allow_missing is meant for
    tolerating a few genuinely-expected gaps, not as a way to skip fixing a
    systematic path mistake."""
    problems = []
    for col in (image_col, text_col):
        if not col or col not in manifest.columns:
            continue
        missing = [
            str(image_root / p) for p in manifest[col]
            if pd.notna(p) and not (image_root / p).exists()
        ]
        if missing:
            problems.append((col, missing))

    if not problems:
        return

    total_missing = sum(len(missing) for _, missing in problems)
    lines = [f"{total_missing} file(s) referenced in the manifest do not exist on disk:"]
    for col, missing in problems:
        lines.append(f"  column {col!r}: {len(missing)} missing, e.g.:")
        for p in missing[:max_examples]:
            lines.append(f"    {p}")
        if len(missing) > max_examples:
            lines.append(f"    ... and {len(missing) - max_examples} more")
    message = "\n".join(lines)

    if allow_missing:
        print(f"WARNING: {message}\n(continuing anyway - --allow-missing-files was set)")
        return
    raise SystemExit(
        f"{message}\n\nThis usually means a wrong --image-root, a wrong --*-col, or (for text) a wrong "
        f"--text-source/OCR output directory - fix the paths, or pass --allow-missing-files to proceed anyway "
        f"(missing text paths fall back to empty text for that page; missing image paths will still fail "
        f"later instead, when that page is actually loaded, since there's no fallback for a missing image)."
    )


def format_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> str:
    """A readable aligned text grid (true label = row, predicted = column).
    Can get wide for many classes, but that's inherent to confusion
    matrices - fine once redirected to a file."""
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "true \\ pred"
    return df.to_string()


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
