"""Loads a page-level manifest as per-PDF page sequences (ordered by page
number) for the sequence-context model in sequence_model.py.

Column names default to this project's real annotation schema (image path /
page number / Document type / Layout Type Classification / Functional
Categories / Start page from data/labels/merged_annotations.tsv) plus two
columns that file doesn't have yet: an actual per-page image file path, and
a split column assigned per PDF (not per page - pages from the same PDF must
stay in the same split, or document-boundary detection can't be evaluated).

Passing `pagexml_col` also loads each page's transcribed text (same PageXML
reader used by train_multimodal.py), for the sequence-context model's
multimodal mode - see train_sequence.py.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.datasets.folder import default_loader

from pagexml import extract_text

IGNORE_INDEX = -100


def build_label_vocab(manifest: pd.DataFrame, split_col: str, column: str, train_split: str = "train") -> list[str]:
    train_rows = manifest[manifest[split_col] == train_split]
    return sorted(train_rows[column].dropna().unique())


def assign_pdf_level_splits(pdf_ids: list[str], ratios=(0.7, 0.15, 0.15), seed: int = 0) -> dict[str, str]:
    rng = random.Random(seed)
    ids = list(pdf_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, round(n * ratios[0]))
    n_val = max(1, round(n * ratios[1])) if n - n_train >= 2 else 0
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    return dict(zip(ids, labels))


class PageSequenceDataset(Dataset):
    """One item = one PDF's ordered pages."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        image_root: Path,
        transform,
        doctype_classes: list[str],
        layout_classes: list[str],
        functional_classes: list[str],
        pdf_col: str = "image path",
        page_col: str = "page number",
        image_col: str = "image",
        doctype_col: str = "Document type",
        layout_col: str = "Layout Type Classification",
        functional_col: str = "Functional Categories",
        start_col: str = "Start page",
        split_col: str = "split",
        pagexml_col: str | None = None,
    ):
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_col = image_col
        self.pagexml_col = pagexml_col
        self.start_col = start_col
        self.doctype_col = doctype_col
        self.layout_col = layout_col
        self.functional_col = functional_col
        self._text_cache: dict[str, str] = {}

        self.doctype_to_idx = {c: i for i, c in enumerate(doctype_classes)}
        self.layout_to_idx = {c: i for i, c in enumerate(layout_classes)}
        self.functional_to_idx = {c: i for i, c in enumerate(functional_classes)}
        self.doctype_classes = doctype_classes
        self.layout_classes = layout_classes
        self.functional_classes = functional_classes

        rows = manifest[manifest[split_col] == split]
        self.pdfs: list[pd.DataFrame] = [
            group.sort_values(page_col) for _, group in rows.groupby(pdf_col, sort=False)
        ]

    def __len__(self) -> int:
        return len(self.pdfs)

    def page_counts(self) -> list[int]:
        """Number of pages per PDF, in dataset order - cheap (from the
        manifest, no image loading) and exactly what a page-budget batch
        sampler needs to decide how many PDFs it can fit in a batch."""
        return [len(group) for group in self.pdfs]

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
                self._text_for(str(self.image_root / p)) if pd.notna(p) else ""
                for p in group[self.pagexml_col]
            ]
        else:
            texts = [""] * len(paths)

        start = torch.tensor(
            [1.0 if str(v).strip().lower() == "yes" else 0.0 for v in group[self.start_col]]
        )
        doctype = torch.tensor([self.doctype_to_idx.get(v, IGNORE_INDEX) for v in group[self.doctype_col]])
        layout = torch.tensor([self.layout_to_idx.get(v, IGNORE_INDEX) for v in group[self.layout_col]])
        functional = torch.tensor([self.functional_to_idx.get(v, IGNORE_INDEX) for v in group[self.functional_col]])
        return images, texts, start, doctype, layout, functional


def make_pdf_collate_fn(tokenizer=None, max_text_length: int = 256):
    """Returns a collate function for PageSequenceDataset batches. Without a
    tokenizer, the batch dict carries only images (image-only sequence
    model). With one, page texts across the whole batch (every PDF's every
    page, flattened the same way images are) are tokenized and padded
    together, added as `input_ids_flat`/`attention_mask_flat` - train_sequence.py's
    embed_pages() picks a PageEmbedder or MultimodalPageEmbedder forward
    accordingly based on whether those keys are present."""

    def collate(batch: list[tuple]) -> dict:
        lengths = [item[0].shape[0] for item in batch]
        B, T_max = len(batch), max(lengths)

        images_flat = torch.cat([item[0] for item in batch], dim=0)
        batch_index = torch.cat([torch.full((n,), b, dtype=torch.long) for b, n in enumerate(lengths)])
        time_index = torch.cat([torch.arange(n) for n in lengths])

        padding_mask = torch.ones(B, T_max, dtype=torch.bool)
        start = torch.zeros(B, T_max)
        doctype = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long)
        layout = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long)
        functional = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long)

        for b, (_, _, s, d, l, f) in enumerate(batch):
            n = s.shape[0]
            padding_mask[b, :n] = False
            start[b, :n] = s
            doctype[b, :n] = d
            layout[b, :n] = l
            functional[b, :n] = f

        result = {
            "images_flat": images_flat,
            "batch_index": batch_index,
            "time_index": time_index,
            "padding_mask": padding_mask,
            "start": start,
            "doctype": doctype,
            "layout": layout,
            "functional": functional,
            "lengths": lengths,
        }

        if tokenizer is not None:
            texts_flat = [text for _, texts, *_ in batch for text in texts]
            encoded = tokenizer(
                texts_flat, padding=True, truncation=True, max_length=max_text_length, return_tensors="pt"
            )
            result["input_ids_flat"] = encoded["input_ids"]
            result["attention_mask_flat"] = encoded["attention_mask"]

        return result

    return collate


class PageBudgetBatchSampler(Sampler):
    """Groups PDF indices into batches so the *total pages* in each batch
    stays under `max_pages_per_batch`, instead of a fixed number of PDFs.

    Total pages per batch is what actually drives memory here: embed_pages()
    flattens every real page across every PDF in the batch into one tensor
    for a single backbone forward pass, so the backbone sees exactly that
    many images regardless of how the batch is composed. A fixed PDF count
    (the plain --batch-size path) doesn't bound this - two 80-page PDFs
    landing in the same batch of 2 costs as much memory as forty 4-page
    PDFs would in a batch of 40, and real documents here range from a
    handful of pages to 80+, so that combination isn't a rare edge case.

    Batches are formed greedily (shuffle order, then pack until the next
    PDF would exceed the budget) - simple, and since the backbone cost
    depends only on total pages and not on how evenly sized the PDFs in a
    batch are, there's no efficiency reason to sort by length first.
    """

    def __init__(self, page_counts: list[int], max_pages_per_batch: int, shuffle: bool = True, seed: int = 0):
        too_long = [(i, n) for i, n in enumerate(page_counts) if n > max_pages_per_batch]
        if too_long:
            longest = max(n for _, n in too_long)
            raise ValueError(
                f"{len(too_long)} PDF(s) exceed --max-pages-per-batch ({max_pages_per_batch}) on their own "
                f"(longest: {longest} pages) - raise --max-pages-per-batch to at least {longest}."
            )
        self.page_counts = page_counts
        self.max_pages_per_batch = max_pages_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self._n_calls = 0

    def __iter__(self):
        indices = list(range(len(self.page_counts)))
        if self.shuffle:
            random.Random(self.seed + self._n_calls).shuffle(indices)
            self._n_calls += 1

        batch: list[int] = []
        batch_pages = 0
        for idx in indices:
            n = self.page_counts[idx]
            if batch and batch_pages + n > self.max_pages_per_batch:
                yield batch
                batch, batch_pages = [], 0
            batch.append(idx)
            batch_pages += n
        if batch:
            yield batch

    def __len__(self) -> int:
        # Approximate: the exact count depends on shuffle order (greedy bin
        # packing), which varies call to call. Good enough for progress bars.
        return math.ceil(sum(self.page_counts) / self.max_pages_per_batch)
