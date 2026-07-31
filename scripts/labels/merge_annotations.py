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

--merge-rare-doctypes additionally collapses Document type classes with too
few examples to learn from into a per-(Layout Type, Functional Category)
"Other" bucket (see scripts/labels/merge_rare_doctypes.py for the reasoning
and the --min-count default) - written to a *separate* file, alongside the
normal full-granularity merged_annotations.tsv, not in place of it, since
downstream consumers may still want the ungrouped labels.

Usage:
    python scripts/labels/merge_annotations.py \
        [--data-dir data/annotations] [--labels-dir data/labels] [--out-dir data/labels]

    # additionally write a doctype-merged version:
    python scripts/labels/merge_annotations.py --merge-rare-doctypes [--min-count 10]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from merge_rare_doctypes import merge_rare_doctypes

ANNOTATION_FILES = {
    "Marijke": "annotations_Marijke.tsv.gz",
    "Marijn": "annotations_Marijn.tsv.gz",
    "Rik": "annotations_Rik.tsv.gz",
    "Yeqian": "annotations_Yeqian.tsv.gz",
}

IMAGE_COL = "image path"
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
        row = {IMAGE_COL: image, PAGE_COL: page, "n_annotators": len(group)}
        has_disagreement = False
        for col in VOTED_COLUMNS:
            winner, status = vote(group[col].tolist(), freqs[col])
            row[col] = winner
            row[f"{col}_agreement"] = status
            if status in ("majority", "tie", "no-data"):
                has_disagreement = True
        rows.append(row)

        if has_disagreement and len(group) > 1:
            detail = {IMAGE_COL: image, PAGE_COL: page}
            for _, r in group.iterrows():
                for col in VOTED_COLUMNS:
                    detail[f"{r['annotator']}: {col}"] = r[col]
            disagreement_rows.append(detail)

    merged_df = pd.DataFrame(rows)
    merged_df["_page_sort"] = pd.to_numeric(merged_df[PAGE_COL], errors="coerce")
    merged_df = merged_df.sort_values([IMAGE_COL, "_page_sort"]).drop(columns="_page_sort")

    disagreement_df = pd.DataFrame(disagreement_rows)
    return merged_df, disagreement_df


def img_path(dossier: str, page_num: int) -> Path:
    """<PNG_ROOT>/<dossier>/<dossier>_page_XXXX.png"""
    return PNG_ROOT / dossier / f'{dossier}_page_{int(page_num):04d}.png'

def text_path(dossier: str, page_num: int) -> Path:
    """<TEXT_ROOT>/<dossier>/page/page_XXXX.xml"""
    return TEXT_ROOT / dossier / 'page' / f'page_{int(page_num):04d}.xml'


def add_splits(merged_df: pd.DataFrame, random_seed: int = 8963764) -> pd.DataFrame:
    column_map = {
        'dossier_name': 'pdf_name',
        'page number': 'page_num',
        'Document type': 'document_type',
        'Document type_agreement': 'document_type_agreement',
        'Layout Type Classification': 'layout_type',
        'Layout Type Classification_agreement': 'layout_type_agreement',
        'Functional Categories': 'functional_category',
        'Functional Categories_agreement': 'functional_category_agreement',
        'Start page': 'start_page',
        'Start page_agreement': 'start_page_agreement',
    }
    merged_df = merged_df.rename(columns=column_map)
    merged_df['pdf_id'] = merged_df.pdf_name.apply(lambda x: x.replace('.pdf', ''))

    merged_df['img_path'] = merged_df.apply(lambda row: img_path(row['pdf_id'], row['page_num']), axis=1)
    merged_df['text_path'] = merged_df.apply(lambda row: text_path(row['pdf_id'], row['page_num']), axis=1)
    dossiers = merged_df[['pdf_name']].drop_duplicates()
    train, validate, test = np.split(dossiers.sample(frac=1, random_state=random_seed), 
                                     [int(.6*len(dossiers)), int(.8*len(dossiers))])
    train['split'] = 'train'
    validate['split'] = 'val'
    test['split'] = 'test'
    dossiers = pd.concat([train, validate, test])
    return pd.merge(merged_df, dossiers, on='pdf_name')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--merge-rare-doctypes", action="store_true",
                         help="also write merged_annotations_doctype_merged.tsv, with rare Document type "
                              "classes collapsed into a per-(Layout Type, Functional Category) 'Other' bucket - "
                              "see scripts/labels/merge_rare_doctypes.py.")
    parser.add_argument("--min-count", type=int, default=10,
                         help="--merge-rare-doctypes only: Document type classes with fewer than this many "
                              "examples get merged.")
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

    if args.merge_rare_doctypes:
        print()
        n_before = merged_df[DOCTYPE_COL].nunique()
        # No split column exists at this stage (splits are assigned later,
        # e.g. by precompute_embeddings.py) - split_col=None makes
        # merge_rare_doctypes count rarity over every row instead.
        grouped_df, mapping = merge_rare_doctypes(
            merged_df, DOCTYPE_COL, LAYOUT_COL, FUNCTIONAL_COL, split_col=None, min_count=args.min_count,
        )
        if not mapping:
            print(f"--merge-rare-doctypes: no Document type classes had fewer than {args.min_count} "
                  f"examples - nothing to merge.")
        else:
            merged_into: dict[str, list[str]] = {}
            for original, target in mapping.items():
                merged_into.setdefault(target, []).append(original)
            print(f"--merge-rare-doctypes: merged {len(mapping)} rare classes (< {args.min_count} examples) "
                  f"into {len(merged_into)} bucket(s):")
            for target, originals in merged_into.items():
                counts = merged_df[merged_df[DOCTYPE_COL].isin(originals)].groupby(DOCTYPE_COL).size()
                detail = ", ".join(f"{o} ({counts.get(o, 0)})" for o in originals)
                print(f"  {target}  <-  {detail}")
            n_after = grouped_df[DOCTYPE_COL].nunique()
            print(f"{n_before} -> {n_after} Document type classes")

        out_path = args.out_dir / "merged_annotations_doctype_merged.tsv"
        grouped_df.to_csv(out_path, index=False, sep="\t")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
