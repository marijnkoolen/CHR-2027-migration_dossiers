"""
Flags likely-wrong predictions from predict.py by checking each predicted
document against structural regularities learned from the ground-truth
training annotations, rather than against any single per-page confidence
score:

  1. Layout/functional mismatch: in the ground truth, each Document type only
     ever co-occurs with a handful of (Layout Type, Functional Category)
     combinations. A predicted document whose predicted combination was
     never observed for its predicted document type in training is
     suspicious - either the doctype, the layout, or the functional category
     prediction is probably wrong.
  2. Page-count mismatch: some document types always have exactly the same
     number of pages in the ground truth (e.g. always a single page). A
     predicted document of such a type with a different page count is
     probably a segmentation error (Start page over/under-triggered) or a
     document-type misclassification.

Neither check needs the predictions to be correct about start pages - both
operate on whatever documents predict.py's own segmentation produced
(predicted_segment_id), so a segmentation mistake shows up as a page-count
or combination anomaly on its own.

Usage:
    python scripts/vision/flag_prediction_errors.py \\
        --gt-manifest data/labels/merged_annotations.tsv \\
        --predictions predictions.tsv \\
        --out flagged_predictions.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def segment_ids_from_start_col(group: pd.DataFrame, page_col: str, start_col: str) -> pd.Series:
    """Mirrors sequence_data.PageSequenceDataset's Start-page convention
    (case-insensitive "yes"/other) and sequence_model.compute_segment_ids'
    rule that the first page of a PDF always begins a segment, regardless of
    its own Start page label."""
    ordered = group.sort_values(page_col)
    is_start = ordered[start_col].astype(str).str.strip().str.lower().eq("yes")
    is_start.iloc[0] = True
    return is_start.cumsum() - 1


def build_gt_documents(
    manifest: pd.DataFrame, pdf_col: str, page_col: str,
    doctype_col: str, layout_col: str, functional_col: str, start_col: str,
) -> pd.DataFrame:
    """One row per ground-truth document: its (majority-vote, in case of
    stray per-page annotation noise) document type/layout/functional and its
    page count."""
    docs = []
    for pdf_id, group in manifest.groupby(pdf_col, sort=False):
        seg_ids = segment_ids_from_start_col(group, page_col, start_col)
        ordered = group.sort_values(page_col)
        for seg_id, seg in ordered.groupby(seg_ids.values):
            docs.append({
                "pdf_id": pdf_id,
                "document_type": seg[doctype_col].mode().iat[0],
                "layout_type": seg[layout_col].mode().iat[0],
                "functional_category": seg[functional_col].mode().iat[0],
                "page_count": len(seg),
            })
    return pd.DataFrame(docs)


def build_predicted_documents(predictions: pd.DataFrame, pred_pdf_col: str, pred_page_col: str) -> pd.DataFrame:
    """One row per predicted document (grouped by predict.py's
    predicted_segment_id), with majority-vote predictions across its pages -
    predictions on individual pages of the same predicted document can
    occasionally disagree even though they share one segment id."""
    docs = []
    for seg_id, seg in predictions.groupby("predicted_segment_id", sort=False):
        seg = seg.sort_values(pred_page_col)
        docs.append({
            "predicted_segment_id": seg_id,
            "pdf_id": seg[pred_pdf_col].iat[0],
            "first_page": seg[pred_page_col].min(),
            "last_page": seg[pred_page_col].max(),
            "page_count": len(seg),
            "predicted_document_type": seg["predicted_document_type"].mode().iat[0],
            "predicted_layout_type": seg["predicted_layout_type"].mode().iat[0],
            "predicted_functional_category": seg["predicted_functional_category"].mode().iat[0],
            "document_type_confidence": seg["document_type_confidence"].mean(),
            "layout_type_confidence": seg["layout_type_confidence"].mean(),
            "functional_category_confidence": seg["functional_category_confidence"].mean(),
        })
    return pd.DataFrame(docs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-manifest", type=Path, required=True, help="the labeled manifest train.py was trained on")
    parser.add_argument("--predictions", type=Path, required=True, help="predict.py's output TSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gt-pdf-col", default="pdf_name")
    parser.add_argument("--gt-page-col", default="page_num")
    parser.add_argument("--gt-doctype-col", default="document_type")
    parser.add_argument("--gt-layout-col", default="layout_type")
    parser.add_argument("--gt-functional-col", default="functional_category")
    parser.add_argument("--gt-start-col", default="page_start")
    parser.add_argument("--pred-pdf-col", default="pdf_name", help="pdf-id column name in --predictions (i.e. "
                         "whatever --pdf-col was passed to predict.py for the new manifest)")
    parser.add_argument("--pred-page-col", default="page_num")
    parser.add_argument("--min-doctype-count", type=int, default=2,
                         help="require at least this many ground-truth documents of a type before treating its "
                              "page count as 'fixed' - a single instance trivially has one page count")
    args = parser.parse_args()

    gt_sep = "\t" if str(args.gt_manifest).endswith(".tsv") else ","
    gt_manifest = pd.read_csv(args.gt_manifest, sep=gt_sep)
    pred_sep = "\t" if str(args.predictions).endswith(".tsv") else ","
    predictions = pd.read_csv(args.predictions, sep=pred_sep)

    gt_docs = build_gt_documents(
        gt_manifest, args.gt_pdf_col, args.gt_page_col,
        args.gt_doctype_col, args.gt_layout_col, args.gt_functional_col, args.gt_start_col,
    )
    print(f"ground truth: {len(gt_docs)} documents across {gt_docs['document_type'].nunique()} document types")

    # Reference table 1: doctype -> set of observed (layout, functional) combinations.
    valid_combos = gt_docs.groupby("document_type").apply(
        lambda d: set(zip(d["layout_type"], d["functional_category"])), include_groups=False
    ).to_dict()

    # Reference table 2: doctype -> fixed page count, only where every
    # sufficiently-attested ground-truth instance agrees on one count.
    fixed_page_counts = {}
    for doctype, group in gt_docs.groupby("document_type"):
        if len(group) < args.min_doctype_count:
            continue
        counts = group["page_count"].unique()
        if len(counts) == 1:
            fixed_page_counts[doctype] = int(counts[0])
    print(f"{len(fixed_page_counts)}/{gt_docs['document_type'].nunique()} document types have a fixed page count "
          f"in the ground truth (n>={args.min_doctype_count}): {fixed_page_counts}")

    pred_docs = build_predicted_documents(predictions, args.pred_pdf_col, args.pred_page_col)
    print(f"predictions: {len(pred_docs)} predicted documents")

    flags = []
    for _, doc in pred_docs.iterrows():
        reasons = []
        doctype = doc["predicted_document_type"]
        combo = (doc["predicted_layout_type"], doc["predicted_functional_category"])

        if doctype not in valid_combos:
            reasons.append(f"predicted document type '{doctype}' never seen in ground truth")
        elif combo not in valid_combos[doctype]:
            observed = ", ".join(f"({l}, {f})" for l, f in sorted(valid_combos[doctype]))
            reasons.append(
                f"layout/functional combination ({combo[0]}, {combo[1]}) never observed for '{doctype}' "
                f"in training data (observed: {observed})"
            )

        if doctype in fixed_page_counts and doc["page_count"] != fixed_page_counts[doctype]:
            reasons.append(
                f"'{doctype}' always has {fixed_page_counts[doctype]} page(s) in training data, "
                f"this predicted document has {doc['page_count']}"
            )

        if reasons:
            flags.append({**doc.to_dict(), "flag_reasons": "; ".join(reasons)})

    flagged = pd.DataFrame(flags)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(args.out, sep="\t", index=False)

    print(f"\n{len(flagged)}/{len(pred_docs)} predicted documents flagged")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
