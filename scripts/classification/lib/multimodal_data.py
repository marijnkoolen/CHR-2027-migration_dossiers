"""Dataset for the late-fusion image+text classifier: pairs each page's
image with its transcribed text (see common.get_text_extractor - markdown
OCR output by default, PageXML as an alternative)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets.folder import default_loader

from common import assign_stratified_splits, build_transforms, get_text_extractor


class MultimodalManifestDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        image_root: Path,
        image_col: str,
        text_col: str,
        label_col: str,
        transform,
        classes: list[str],
        text_extractor: Callable[[str], str],
    ):
        self.image_root = Path(image_root)
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.text_extractor = text_extractor
        self.samples = [
            (
                str(self.image_root / row[image_col]),
                str(self.image_root / row[text_col]) if pd.notna(row.get(text_col)) else "",
                self.class_to_idx[row[label_col]],
            )
            for _, row in rows.iterrows()
            if row[label_col] in self.class_to_idx
        ]
        self._text_cache: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _text_for(self, text_path: str) -> str:
        if text_path not in self._text_cache:
            self._text_cache[text_path] = self.text_extractor(text_path) if text_path else ""
        return self._text_cache[text_path]

    def __getitem__(self, idx: int):
        image_path, text_path, label = self.samples[idx]
        image = self.transform(default_loader(image_path))
        text = self._text_for(text_path)
        return image, text, label


def make_collate_fn(tokenizer, max_length: int):
    """Images stack normally; text is tokenized+padded per batch (dynamic
    padding - cheaper than padding every batch to a fixed max length)."""

    def collate(batch):
        images, texts, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return images, encoded["input_ids"], encoded["attention_mask"], labels

    return collate


def build_multimodal_dataloaders(
    manifest_path: Path,
    image_root: Path,
    label_col: str,
    tokenizer,
    image_col: str = "image",
    text_col: str = "pagexml",
    text_source: str = "markdown",
    split_col: str = "split",
    image_size: int = 224,
    batch_size: int = 16,
    max_text_length: int = 256,
    augment_strength: str = "moderate",
    balance_classes: bool = True,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, list[str]]:
    """The multimodal equivalent of common.build_dataloaders_from_manifest -
    same manifest conventions (auto-assigns a stratified split if split_col
    is missing), plus a text_col for each page's transcription and
    text_source ("markdown" or "pagexml" - see common.get_text_extractor)
    for how to read it."""
    sep = "\t" if str(manifest_path).endswith(".tsv") else ","
    manifest = pd.read_csv(manifest_path, sep=sep)
    text_extractor = get_text_extractor(text_source)

    if split_col not in manifest.columns:
        manifest[split_col] = assign_stratified_splits(manifest[label_col], seed=seed)
        print(f"No '{split_col}' column in {manifest_path} - assigned a stratified "
              f"70/15/15 train/val/test split automatically (seed={seed}).")

    classes = sorted(manifest.loc[manifest[split_col] == "train", label_col].dropna().unique())
    collate = make_collate_fn(tokenizer, max_text_length)

    def make_ds(split: str, train: bool) -> MultimodalManifestDataset:
        rows = manifest[manifest[split_col] == split]
        return MultimodalManifestDataset(
            rows, image_root, image_col, text_col, label_col,
            build_transforms(image_size, train=train, augment_strength=augment_strength), classes, text_extractor,
        )

    train_ds = make_ds("train", train=True)

    sampler, shuffle = None, True
    if balance_classes:
        counts = Counter(label for *_, label in train_ds.samples)
        weights = [1.0 / counts[label] for *_, label in train_ds.samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, collate_fn=collate)

    def eval_loader(split: str) -> DataLoader | None:
        ds = make_ds(split, train=False)
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, eval_loader("val"), eval_loader("test"), classes
