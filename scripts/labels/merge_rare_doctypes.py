"""
Merges rare document_type classes into a single, per-(layout_type,
functional_category) "Other" bucket, so the sequence model's doctype head
isn't trying to learn from classes with only a handful of training
examples. Of this project's 36 real doctype classes, 17 have fewer than 10
training examples and several have exactly 1 - those can't realistically be
learned from that few examples, and contribute large, noisy per-example
loss whenever they appear, which (see train.py's --loss-weight-doctype)
plausibly drags down the other three heads too, since all four task losses
are summed into one shared gradient.

Rarity is determined from the TRAIN split's counts only where a split
column is available (never val/test - this is purely a decision about
what's learnable from the training data, not something the evaluation set
should influence), then the SAME mapping is applied to every split, so
val/test are evaluated in the same merged label space the model is trained
on - not silently dropped, which is what would happen otherwise (a
val/test row whose original label vanished from the train-derived class
vocabulary is just ignored during evaluation). If split_col isn't given or
isn't present (e.g. run on merge_annotations.py's output, upstream of any
split assignment), rarity is determined from every row instead.

--count-by {pages,pdfs} (default pages) picks what "rare" is counted in.
pages (the default) targets learnability - a class needs enough labeled
examples to be worth its own head output. pdfs targets split-assignment
feasibility instead: a class split across only 1-2 distinct PDFs can never
appear in all of train/val/test no matter how splits are assigned (a PDF's
pages all go to one split), regardless of how many total pages it has -
use this before assign_stratified_splits.py, so every remaining class has
enough distinct PDFs to plausibly be distributed across all three splits.

Each rare doctype is merged into "Other ({layout_type}, {functional_category})"
using its own (dominant) layout/functional combination - not one flat
"Other" bucket - so classes that are rare for different reasons (e.g. a
rare form vs. a rare letter) don't get lumped together just for both being
rare. This also matches what flag_prediction_errors.py already treats as
each doctype's defining signature.

Usage (cached-embeddings manifest):
    python scripts/labels/merge_rare_doctypes.py \\
        --manifest data/embeddings_text_xlm_roberta_base/embeddings_manifest.tsv \\
        --out data/embeddings_text_xlm_roberta_base/embeddings_manifest_merged.tsv \\
        --min-count 10

    # or a live raw-image manifest (adjust column names to match):
    python scripts/labels/merge_rare_doctypes.py \\
        --manifest data/labels/dossier_labels.tsv --out data/labels/dossier_labels_merged.tsv \\
        --min-count 10 --doctype-col document_type --layout-col layout_type \\
        --functional-col functional_category --split-col split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_rare_doctypes(
    manifest: pd.DataFrame, doctype_col: str, layout_col: str, functional_col: str, split_col: str | None,
    min_count: int = 10, count_by: str = "pages", pdf_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Returns (manifest with document_type remapped for rare classes,
    {original_doctype: merged_label} for only the classes actually merged).

    split_col: if given and present in `manifest`, rarity is counted from
    split=="train" rows only. Otherwise (None, or the column doesn't exist -
    e.g. this manifest predates split assignment) every row counts.

    count_by: "pages" (default) counts rows; "pdfs" counts distinct PDFs
    (requires pdf_col) - see module docstring for when to use which."""
    if count_by not in ("pages", "pdfs"):
        raise ValueError(f"count_by must be 'pages' or 'pdfs', got {count_by!r}")
    if count_by == "pdfs" and not pdf_col:
        raise ValueError("count_by='pdfs' requires pdf_col")

    if split_col and split_col in manifest.columns:
        count_rows = manifest[manifest[split_col] == "train"]
    else:
        count_rows = manifest
    train_counts = (
        count_rows.groupby(doctype_col)[pdf_col].nunique() if count_by == "pdfs" else count_rows[doctype_col].value_counts()
    )
    rare = set(train_counts[train_counts < min_count].index)

    mapping = {}
    for doctype in rare:
        rows = manifest[manifest[doctype_col] == doctype]
        layout = rows[layout_col].mode().iat[0]
        functional = rows[functional_col].mode().iat[0]
        mapping[doctype] = f"Other ({layout}, {functional})"

    merged = manifest.copy()
    if mapping:
        merged[doctype_col] = merged[doctype_col].map(lambda v: mapping.get(v, v))
    return merged, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=10,
                         help="doctype classes with fewer than this many TRAIN examples (pages, or PDFs if "
                              "--count-by pdfs) get merged. Default 10 matches what was found empirically here "
                              "for pages: 17/36 classes had under 10 train examples, 12 had under 5.")
    parser.add_argument("--count-by", choices=["pages", "pdfs"], default="pages",
                         help="what --min-count counts - see module docstring.")
    parser.add_argument("--pdf-col", default="pdf_name", help="required when --count-by pdfs")
    parser.add_argument("--doctype-col", default="document_type")
    parser.add_argument("--layout-col", default="layout_type")
    parser.add_argument("--functional-col", default="functional_category")
    parser.add_argument("--split-col", default="split",
                         help="if this column isn't present in --manifest (e.g. a manifest from "
                              "merge_annotations.py, which runs before any split is assigned), rarity is "
                              "counted from every row instead of split=='train' only.")
    args = parser.parse_args()

    sep = "\t" if str(args.manifest).endswith(".tsv") else ","
    manifest = pd.read_csv(args.manifest, sep=sep)
    has_split = args.split_col in manifest.columns
    if not has_split:
        print(f"note: no '{args.split_col}' column in --manifest - counting rarity over all rows, not train only.")

    def n_doctypes(df: pd.DataFrame) -> int:
        rows = df[df[args.split_col] == "train"] if has_split else df
        return rows[args.doctype_col].nunique()

    n_before = n_doctypes(manifest)
    print(f"{len(manifest)} rows, {n_before} doctype classes" + (" in train" if has_split else ""))

    merged, mapping = merge_rare_doctypes(
        manifest, args.doctype_col, args.layout_col, args.functional_col,
        args.split_col if has_split else None, min_count=args.min_count,
        count_by=args.count_by, pdf_col=args.pdf_col,
    )

    unit = "PDFs" if args.count_by == "pdfs" else "examples"
    if not mapping:
        print(f"no doctype classes had fewer than {args.min_count} {unit} - nothing to merge.")
    else:
        merged_into = {}
        for original, target in mapping.items():
            merged_into.setdefault(target, []).append(original)
        print(f"merged {len(mapping)} rare classes (< {args.min_count} {unit}) into "
              f"{len(merged_into)} bucket(s):")
        for target, originals in merged_into.items():
            rows = manifest[manifest[args.doctype_col].isin(originals)]
            counts = (
                rows.groupby(args.doctype_col)[args.pdf_col].nunique() if args.count_by == "pdfs"
                else rows.groupby(args.doctype_col).size()
            )
            detail = ", ".join(f"{o} ({counts.get(o, 0)})" for o in originals)
            print(f"  {target}  <-  {detail}")

    n_after = n_doctypes(merged)
    print(f"\n{n_before} -> {n_after} doctype classes" + (" in train" if has_split else ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, sep=sep, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
