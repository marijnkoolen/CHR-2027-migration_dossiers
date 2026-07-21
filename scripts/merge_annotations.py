"""
Merge the four annotators' per-page annotations into a single consensus file.

For every (image path, page number):

- "Document type" is resolved to a preferred label per annotator using the
  mappings in data/labels/label_mapping_unified.tsv (labels with shared-PDF
  evidence) and data/labels/mapped_single_annotator_labels.tsv (labels only
  ever used outside the shared PDFs), then combined across annotators.
- "Layout Type Classification", "Functional Categories" and "Start page" are
  combined directly - annotators already use the same vocabulary for these,
  so no separate label mapping is needed. For "Start page", a blank cell is
  treated as "no" (that is how most annotators mark a non-start page).

A page annotated by only one annotator (any of the 15 PDFs unique to that
annotator) simply keeps that annotator's (mapped) value. A page annotated by
all four (the 5 shared PDFs) is resolved by majority vote; ties are broken by
whichever value is used more often across the whole corpus. Every page where
the annotators did not unanimously agree is written to a separate report for
review.

Usage:
    python scripts/merge_annotations.py \
        [--data-dir data/annotations] [--labels-dir data/labels] [--out-dir data/labels]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ANNOTATION_FILES = {
    "Marijke": "annotations_Marijke.tsv.gz",
    "Marijn": "annotations_Marijn.tsv.gz",
    "Rik": "annotations_Rik.tsv.gz",
    "Yeqian": "annotations_Yeqian.tsv.gz",
}

IMAGE_COL = "image path"
IMAGE_NEW_COL = "dossier_name"
PAGE_COL = "page number"
DOCTYPE_COL = "Document type"
LAYOUT_COL = "Layout Type Classification"
FUNCTIONAL_COL = "Functional Categories"
START_PAGE_COL = "Start page"

PNG_ROOT   = Path('data/image-per-page')
TEXT_ROOT  = Path('data/text-per-page')

VOTED_COLUMNS = [DOCTYPE_COL, LAYOUT_COL, FUNCTIONAL_COL, START_PAGE_COL]


def load_annotations(data_dir: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, fname in ANNOTATION_FILES.items():
        df = pd.read_csv(data_dir / fname, sep="\t", dtype=str)
        df = df.dropna(subset=[IMAGE_COL])
        df = df[df[IMAGE_COL].str.strip() != ""]
        df = df[[IMAGE_COL, PAGE_COL, DOCTYPE_COL, LAYOUT_COL, FUNCTIONAL_COL, START_PAGE_COL]].copy()
        df = df.drop_duplicates(subset=[IMAGE_COL, PAGE_COL])
        # Blank means "not a start page" for most annotators (only Marijke
        # sometimes writes an explicit "no"), so it is a real value here,
        # unlike the other columns where a blank means missing data.
        df[START_PAGE_COL] = df[START_PAGE_COL].fillna("no")
        dfs[name] = df
    return dfs


def build_doctype_lookup(labels_dir: Path) -> dict[tuple[str, str], str]:
    """(annotator, raw Document type) -> preferred label, from the two mapping files."""
    lookup: dict[tuple[str, str], str] = {}

    unified = pd.read_csv(labels_dir / "label_mapping_unified.tsv", sep="\t", dtype=str)
    annotators = [c[len("labels_") :] for c in unified.columns if c.startswith("labels_")]
    for _, row in unified.iterrows():
        preferred = row["preferred_label"]
        for annotator in annotators:
            value = row.get(f"labels_{annotator}")
            if pd.isna(value) or value == "":
                continue
            for label in str(value).split("; "):
                lookup[(annotator, label)] = preferred

    single = pd.read_csv(labels_dir / "mapped_single_annotator_labels.tsv", sep="\t", dtype=str)
    for _, row in single.iterrows():
        lookup[(row["annotator"], row["label"])] = row["preferred_label"]

    return lookup


def apply_doctype_mapping(dfs: dict[str, pd.DataFrame], lookup: dict[tuple[str, str], str]) -> list[dict]:
    """Replace each annotator's raw Document type with its preferred label in place.
    Returns a log of any raw label that had no entry in either mapping file."""
    unmapped_log = []

    for name, df in dfs.items():
        def map_label(label, name=name):
            if pd.isna(label):
                return label
            key = (name, label)
            if key in lookup:
                return lookup[key]
            unmapped_log.append({"annotator": name, "label": label})
            return label

        df[DOCTYPE_COL] = df[DOCTYPE_COL].apply(map_label)

    return unmapped_log


def build_long_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (image path, page number, annotator)."""
    frames = []
    for name, df in dfs.items():
        sub = df.copy()
        sub["annotator"] = name
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def vote(values: list[str], freq: Counter) -> tuple[str | None, str]:
    """Majority vote among non-null values.

    Returns (winning value, status), where status is one of:
    "no-data", "single", "unanimous", "majority", "tie". Ties are broken by
    whichever value occurs more often across the whole corpus.
    """
    values = [v for v in values if pd.notna(v)]
    if not values:
        return None, "no-data"
    if len(values) == 1:
        return values[0], "single"

    counts = Counter(values)
    top = max(counts.values())
    winners = sorted((v for v, c in counts.items() if c == top), key=lambda v: -freq.get(v, 0))
    if len(winners) > 1:
        status = "tie"
    elif top == len(values):
        status = "unanimous"
    else:
        status = "majority"
    return winners[0], status


def merge(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    freqs = {col: Counter(long_df[col].dropna()) for col in VOTED_COLUMNS}

    rows = []
    disagreement_rows = []
    for (image, page), group in long_df.groupby([IMAGE_COL, PAGE_COL], sort=False):
        row = {IMAGE_NEW_COL: image, PAGE_COL: page, "n_annotators": len(group)}
        has_disagreement = False
        for col in VOTED_COLUMNS:
            winner, status = vote(group[col].tolist(), freqs[col])
            row[col] = winner
            row[f"{col}_agreement"] = status
            if status in ("majority", "tie", "no-data"):
                has_disagreement = True
        rows.append(row)

        if has_disagreement and len(group) > 1:
            detail = {IMAGE_NEW_COL: image, PAGE_COL: page}
            for _, r in group.iterrows():
                for col in VOTED_COLUMNS:
                    detail[f"{r['annotator']}: {col}"] = r[col]
            disagreement_rows.append(detail)

    merged_df = pd.DataFrame(rows)
    merged_df["_page_sort"] = pd.to_numeric(merged_df[PAGE_COL], errors="coerce")
    merged_df = merged_df.sort_values([IMAGE_NEW_COL, "_page_sort"]).drop(columns="_page_sort")

    disagreement_df = pd.DataFrame(disagreement_rows)
    return merged_df, disagreement_df


def img_path(dossier: str, page_num: int) -> Path:
    """pdf_pages_png/<dossier>/<dossier>_page_XXXX.png"""
    return PNG_ROOT / dossier / f'{dossier}_page_{int(page_num):04d}.png'

def text_path(dossier: str, page_num: int) -> Path:
    """outputs/page_text_by_page/<dossier>/page/page_XXXX.txt"""
    return TEXT_ROOT / dossier / 'page' / f'page_{int(page_num):04d}.txt'


def add_splits(merged_df: pd.DataFrame, random_seed: int = 8963764) -> pd.DataFrame:
    column_map = {
        'Start page': 'is_start',
        'page number': 'page_num',
        'Document type': 'doc_type',
        'Functional Categories': 'func_label',
    }
    merged_df = merged_df.rename(columns=column_map)
    merged_df['dossier'] = merged_df.dossier_name.apply(lambda x: x.replace('.pdf', ''))

    merged_df['img_path'] = merged_df.apply(lambda row: img_path(row['dossier'], row['page_num']), axis=1)
    merged_df['text_path'] = merged_df.apply(lambda row: text_path(row['dossier'], row['page_num']), axis=1)
    dossiers = merged_df[['dossier_name']].drop_duplicates()
    train, validate, test = np.split(dossiers.sample(frac=1, random_state=random_seed), 
                                 [int(.6*len(dossiers)), int(.8*len(dossiers))])
    train['split'] = 'train'
    validate['split'] = 'val'
    test['split'] = 'test'
    dossiers = pd.concat([train, validate, test])
    return pd.merge(merged_df, dossiers, on='dossier_name')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/labels"))
    args = parser.parse_args()

    dfs = load_annotations(args.data_dir)
    lookup = build_doctype_lookup(args.labels_dir)
    unmapped_log = apply_doctype_mapping(dfs, lookup)

    if unmapped_log:
        print(f"WARNING: {len(unmapped_log)} raw Document type value(s) had no mapping entry "
              "and were kept as-is:")
        for entry in unmapped_log:
            print(f"  - {entry['annotator']}: {entry['label']!r}")
        print()

    long_df = build_long_table(dfs)
    merged_df, disagreement_df = merge(long_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(args.out_dir / "merged_annotations.tsv", index=False, sep="\t")
    merged_split_df = add_splits(merged_df)
    merged_split_df.to_csv(args.out_dir / "dossier_labels.tsv", sep="\t", index=False)
    disagreement_df.to_csv(args.out_dir / "merge_disagreements.tsv", index=False, sep="\t")

    print(f"Merged {len(merged_df)} pages from {sum(len(df) for df in dfs.values())} source rows "
          f"across {len(dfs)} annotators.")
    print()
    for col in VOTED_COLUMNS:
        counts = merged_df[f"{col}_agreement"].value_counts()
        summary = ", ".join(f"{status}: {counts.get(status, 0)}" for status in
                             ["single", "unanimous", "majority", "tie", "no-data"])
        print(f"{col}: {summary}")
    print()

    print(f"Wrote {args.out_dir / 'merged_annotations.tsv'}")
    print(f"Wrote {args.out_dir / 'merge_disagreements.tsv'} ({len(disagreement_df)} page(s) to review)")


if __name__ == "__main__":
    main()
