"""
Reassigns train/val/test splits at the PDF level (all of a PDF's pages go
to the same split, so sequence-mode document-boundary detection stays
valid), guaranteeing every document_type class present in the manifest gets
at least one PDF in each split - not just *proportionally likely* to, which
is what plain per-PDF random assignment gives you, and is exactly why the
existing split has 18 of 37 document types with zero test examples: with
only 65 PDFs and 37+ classes, random assignment has no way to know a rare
class needs deliberate coverage.

Run this on a manifest that's already had rare classes merged **by PDF
count** (merge_rare_doctypes.py --count-by pdfs), not page count. A class
spread across fewer than 3 distinct PDFs can never appear in all three
splits however cleverly splits are assigned - a PDF's pages can't be
divided across splits - so guaranteeing coverage for those classes is
impossible regardless of this script; merging them first is the only fix.

Algorithm (a *group*-stratified split: PDFs are the atomic unit, but each
PDF can carry several different document_type labels - a dossier usually
contains multiple documents - so this isn't a standard single-label
stratified split):
    1. Process document_type classes rarest-first (fewest distinct PDFs) -
       the tightest constraints get first claim on their own limited
       coverage, before more common classes compete for shared PDFs.
    2. For each class, and for each split not yet covered by an
       already-assigned PDF, assign it one of its own (still-unassigned)
       covering PDFs.
    3. Once every class has >=1 PDF in every split, assign every remaining
       unassigned PDF to whichever split is furthest below its target size,
       to approximate the requested ratios.

Expect the resulting split sizes to deviate somewhat from the requested
ratios - guaranteeing coverage for many classes "spends" PDFs on val/test
that a purely proportional random assignment wouldn't have needed to.

Usage:
    python scripts/labels/assign_stratified_splits.py \\
        --manifest data/labels/dossier_labels_merged_pdf10.tsv \\
        --out data/labels/dossier_labels_stratified.tsv \\
        --pdf-col pdf_name --doctype-col document_type \\
        --ratios 0.7 0.15 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd

SPLIT_NAMES = ["train", "val", "test"]


def build_class_coverage(manifest: pd.DataFrame, pdf_col: str, doctype_col: str) -> dict[str, set]:
    """{doctype: set(pdf_ids containing at least one page of that doctype)}."""
    return {cls: set(rows[pdf_col]) for cls, rows in manifest.groupby(doctype_col)}


def assign_stratified_splits(
    manifest: pd.DataFrame, pdf_col: str, doctype_col: str,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Returns ({pdf_id: split}, [(doctype, split) pairs that could not be
    covered - empty if every class got >=1 PDF in every split])."""
    rng = random.Random(seed)
    class_to_pdfs = build_class_coverage(manifest, pdf_col, doctype_col)
    all_pdfs = list(manifest[pdf_col].unique())
    rng.shuffle(all_pdfs)  # so within-class candidate order is random but reproducible

    assigned: dict[str, str] = {}
    unmet: list[tuple[str, str]] = []

    classes_by_rarity = sorted(class_to_pdfs, key=lambda c: len(class_to_pdfs[c]))
    for cls in classes_by_rarity:
        coverage = [p for p in all_pdfs if p in class_to_pdfs[cls]]
        have = {assigned[p] for p in coverage if p in assigned}
        for split in SPLIT_NAMES:
            if split in have:
                continue
            candidate = next((p for p in coverage if p not in assigned), None)
            if candidate is None:
                unmet.append((cls, split))
                continue
            assigned[candidate] = split
            have.add(split)

    n_total = len(all_pdfs)
    target = {s: round(r * n_total) for s, r in zip(SPLIT_NAMES, ratios)}
    counts = Counter(assigned.values())
    for p in all_pdfs:
        if p in assigned:
            continue
        deficits = {s: target[s] - counts.get(s, 0) for s in SPLIT_NAMES}
        best = max(deficits, key=lambda s: deficits[s])
        assigned[p] = best
        counts[best] += 1

    return assigned, unmet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                         help="should already be rare-merged by PDF count - see module docstring")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf-col", default="pdf_name")
    parser.add_argument("--doctype-col", default="document_type")
    parser.add_argument("--split-col", default="split", help="column to overwrite with the new assignment")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sep = "\t" if str(args.manifest).endswith(".tsv") else ","
    manifest = pd.read_csv(args.manifest, sep=sep)
    n_pdfs = manifest[args.pdf_col].nunique()
    n_classes = manifest[args.doctype_col].nunique()
    print(f"{len(manifest)} rows, {n_pdfs} PDFs, {n_classes} document_type classes")

    class_to_pdfs = build_class_coverage(manifest, args.pdf_col, args.doctype_col)
    thin = {c: len(p) for c, p in class_to_pdfs.items() if len(p) < 3}
    if thin:
        print(f"\nWARNING: {len(thin)} class(es) have fewer than 3 distinct PDFs - full 3-way coverage is "
              f"mathematically impossible for these regardless of split assignment (a PDF's pages can't be "
              f"split across splits). Merge these first (merge_rare_doctypes.py --count-by pdfs):")
        for c, n in sorted(thin.items(), key=lambda kv: kv[1]):
            print(f"  {c}: {n} PDF(s)")

    assigned, unmet = assign_stratified_splits(
        manifest, args.pdf_col, args.doctype_col, ratios=tuple(args.ratios), seed=args.seed,
    )

    result = manifest.copy()
    result[args.split_col] = result[args.pdf_col].map(assigned)

    split_pdf_counts = Counter(assigned.values())
    print(f"\nsplit sizes (PDFs): " + ", ".join(f"{s}={split_pdf_counts.get(s, 0)}" for s in SPLIT_NAMES)
          + f"  (requested ratios: {dict(zip(SPLIT_NAMES, args.ratios))})")

    # Verify by recomputing page-level coverage from the actual output,
    # not just trusting the assignment logic - this is the number that
    # actually matters for macro-F1 (see train.py's evaluate_sequence).
    page_counts = result.groupby([args.doctype_col, args.split_col]).size().unstack(fill_value=0)
    for s in SPLIT_NAMES:
        if s not in page_counts.columns:
            page_counts[s] = 0
    still_missing = page_counts[(page_counts[SPLIT_NAMES] == 0).any(axis=1)]

    if unmet:
        print(f"\n{len(unmet)} (class, split) pair(s) could not be covered during assignment "
              f"(ran out of unassigned covering PDFs):")
        for cls, split in unmet:
            print(f"  {cls} -> {split}")
    if len(still_missing) > 0:
        print(f"\n{len(still_missing)} class(es) still have zero pages in at least one split after assignment:")
        print(still_missing[SPLIT_NAMES])
    if not unmet and len(still_missing) == 0:
        print(f"\nAll {n_classes} document_type classes have at least one page in train, val, and test.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, sep=sep, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
