"""
Computes evaluation measures for predict.py's output against ground truth,
beyond the page-level macro-F1 used elsewhere in this project. --metrics pq
and --metrics las follow two specific published protocols for Page Stream
Segmentation, rather than being ad hoc adaptations:

    --metrics macro-f1   per-page classification: accuracy, macro F1 (matching
                         train.py's evaluate_sequence - macro-averaged over
                         the full trained class vocabulary if --classes-json
                         is given, otherwise over whatever classes appear in
                         this comparison, see train.py for why that
                         distinction matters), and weighted F1 (support-
                         weighted - closer to accuracy but still per-class,
                         useful alongside macro F1 since the two disagree
                         most exactly when rare classes are doing badly).

    --metrics pq         Panoptic Quality, following van Heusden, Kamps &
                         Marx's exact PSS protocol (WooIR, SIGIR-ICTIR 2022;
                         restated in OpenPSS, TPDL 2024, Sec 3.5), itself
                         citing Kirillov et al. 2019's original 2D image
                         panoptic segmentation formulation. A true document
                         t and predicted document p (both page sets) form a
                         True Positive iff IoU(t,p) > 0.5 - this threshold
                         guarantees at most one TP per document on either
                         side, so no bipartite-matching step is needed.
                         Unmatched predicted documents are FP, unmatched
                         true documents are FN. RQ ("recognition quality")
                         is the ordinary document-level F1 from those
                         TP/FP/FN; SQ ("segmentation quality") is the mean
                         IoU over TP pairs; PQ = SQ x RQ. Per van Heusden et
                         al.'s protocol, SQ/RQ/PQ are computed per document
                         stream (PDF) and then macro-averaged across
                         streams, not pooled across the whole test set -
                         implemented that way here. PQ deliberately ignores
                         predicted document_type/layout_type/functional_
                         category - it isolates "did the model find the
                         right document boundaries" from "did it classify
                         them correctly", which --metrics las is for.

    --metrics las         Unlabeled/Labeled Attachment Score, following
                         Demirtas et al.'s page-dependency-parsing protocol
                         ("Semantic Parsing of Interpage Relations", ICPR
                         2022). They treat each page as a dependency-parsing
                         token with a "head" page it attaches to, and report
                         UAS/LAS as the fraction of tokens (pages) attached
                         to the correct head, without (UAS) or with (LAS)
                         the correct relation label - pooled over the whole
                         test set, not averaged per document. Their full
                         formulation supports rich interpage relation types
                         (attachment/copy/back-page/continuation); this
                         project's pipeline only predicts document
                         boundaries and per-segment classification labels,
                         not those finer relations, so it's restricted here
                         to the same reduction Demirtas et al.'s own
                         plain-PSS baseline uses: every page's head is
                         simply the first page of the document segment it
                         belongs to. Per page: UAS counts it correct iff its
                         predicted head page (the first page of its
                         predicted segment) exactly equals its true head
                         page; LAS additionally requires the two segments'
                         majority-vote labels (--label-col, default
                         document_type) to agree. Unlike --metrics pq, this
                         requires an EXACT head-page match, not just falling
                         inside an IoU>0.5-matched segment pair - the same
                         strictness dependency-parsing UAS has toward
                         getting the exact head token right. LAS <= UAS
                         always, same as in parsing.

Usage:
    # single file: predict.py was run on an already-labeled manifest, so
    # predictions.tsv already has both predicted_* and ground-truth columns
    python scripts/classification/evaluate_segmentation.py \\
        --predictions predictions.tsv --metrics pq las macro-f1

    # two files: join predictions against a separate ground-truth manifest
    python scripts/classification/evaluate_segmentation.py \\
        --predictions predictions.tsv --ground-truth data/labels/dossier_labels.tsv \\
        --metrics pq las macro-f1 --classes-json runs/from_embeddings/multimodal_efficient/classes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from flag_prediction_errors import segment_ids_from_start_col

ALL_METRICS = ["macro-f1", "pq", "las"]


# --------------------------------------------------------------------------
# Segment construction
# --------------------------------------------------------------------------

def segments_from_start_col(df: pd.DataFrame, pdf_col: str, page_col: str, start_col: str) -> dict[str, dict[int, set]]:
    """{pdf_id: {head_page_num: set(page_numbers)}}, computed from a yes/no
    start_col using sequence_model.py's convention (a PDF's first real page
    always begins a segment, regardless of its own label). Segments are
    keyed by their own head page number (the smallest page number in the
    segment) rather than an arbitrary local counter, so this key can double
    as Demirtas et al.'s "head page" for --metrics las without any extra
    bookkeeping."""
    segments: dict[str, dict[int, set]] = {}
    for pdf_id, group in df.groupby(pdf_col, sort=False):
        seg_ids = segment_ids_from_start_col(group, page_col, start_col)
        ordered = group.sort_values(page_col)
        segments[pdf_id] = {
            seg_rows[page_col].min(): set(seg_rows[page_col])
            for _, seg_rows in ordered.groupby(seg_ids.values)
        }
    return segments


def segments_from_id_col(df: pd.DataFrame, pdf_col: str, page_col: str, segment_id_col: str) -> dict[str, dict]:
    """{pdf_id: {head_page_num: set(page_numbers)}}, from an existing
    per-page segment-id column (e.g. predict.py's predicted_segment_id) -
    re-keyed by head page number, see segments_from_start_col."""
    segments: dict[str, dict] = {}
    for pdf_id, group in df.groupby(pdf_col, sort=False):
        segments[pdf_id] = {
            rows[page_col].min(): set(rows[page_col]) for _, rows in group.groupby(segment_id_col)
        }
    return segments


def majority_label_from_segments(df: pd.DataFrame, pdf_col: str, page_col: str, label_col: str, segments: dict) -> dict:
    """{pdf_id: {head_page_num: majority label}}, using an already-built
    segments dict ({pdf_id: {head_page_num: set(pages)}}) directly, rather
    than needing a per-row segment-id column - avoids reassembling one via
    groupby/apply, which is awkward to do robustly across a variable number
    of PDFs."""
    result: dict[str, dict] = {}
    for pdf_id, segs in segments.items():
        rows = df[df[pdf_col] == pdf_id]
        result[pdf_id] = {
            head: rows[rows[page_col].isin(pages)][label_col].mode().iat[0] for head, pages in segs.items()
        }
    return result


def head_page_lookup(segments: dict) -> dict[str, dict[int, int]]:
    """{pdf_id: {page_num: head_page_num}} - every page's own segment's
    head (see segments_from_start_col), i.e. Demirtas et al.'s per-page
    "head token" restricted to plain PSS (no finer attachment/copy/back
    relations, since this project's pipeline doesn't predict those)."""
    lookup: dict[str, dict[int, int]] = {}
    for pdf_id, segs in segments.items():
        page_to_head: dict[int, int] = {}
        for head, pages in segs.items():
            for p in pages:
                page_to_head[p] = head
        lookup[pdf_id] = page_to_head
    return lookup


# --------------------------------------------------------------------------
# Segment matching (shared by PQ and LAS/UAS)
# --------------------------------------------------------------------------

def match_segments(true_segs: dict[int, set], pred_segs: dict[int, set], iou_threshold: float = 0.5):
    """Within one PDF. Returns (matches: list[(true_id, pred_id, iou)],
    unmatched_true_ids, unmatched_pred_ids).

    IoU > 0.5 guarantees a unique match on each side (two different segments
    on the same side can't each share more than half of a given segment's
    pages, since their own pages are disjoint from each other), so this is
    a direct pairwise check, not a bipartite-matching problem."""
    matches = []
    matched_true, matched_pred = set(), set()
    for t_id, t_pages in true_segs.items():
        for p_id, p_pages in pred_segs.items():
            union = t_pages | p_pages
            if not union:
                continue
            iou = len(t_pages & p_pages) / len(union)
            if iou > iou_threshold:
                matches.append((t_id, p_id, iou))
                matched_true.add(t_id)
                matched_pred.add(p_id)
                break
    unmatched_true = [t for t in true_segs if t not in matched_true]
    unmatched_pred = [p for p in pred_segs if p not in matched_pred]
    return matches, unmatched_true, unmatched_pred


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def panoptic_quality(true_segments: dict, pred_segments: dict, iou_threshold: float = 0.5) -> dict:
    """SQ/RQ/PQ computed per document stream (PDF), then macro-averaged
    across streams - van Heusden et al.'s explicit protocol ("All metrics
    are always calculated per stream ... we measure the performance of
    models by the averages of the metrics over the streams."), not TP/FP/FN
    pooled across the whole test set first. TP/FP/FN totals are still
    reported (informational only - they don't feed into PQ/SQ/RQ here)."""
    per_stream = []
    total_tp = total_fp = total_fn = 0
    for pdf_id, t_segs in true_segments.items():
        p_segs = pred_segments.get(pdf_id, {})
        matches, unmatched_true, unmatched_pred = match_segments(t_segs, p_segs, iou_threshold)
        tp, fp, fn = len(matches), len(unmatched_pred), len(unmatched_true)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if tp + fp + fn == 0:
            continue
        sq = float(np.mean([iou for _, _, iou in matches])) if matches else 0.0
        denom = tp + 0.5 * fp + 0.5 * fn
        rq = tp / denom if denom > 0 else 0.0
        per_stream.append({"pq": sq * rq, "sq": sq, "rq": rq})

    if not per_stream:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": 0, "n_streams": 0}
    pq = float(np.mean([s["pq"] for s in per_stream]))
    sq = float(np.mean([s["sq"] for s in per_stream]))
    rq = float(np.mean([s["rq"] for s in per_stream]))
    return {"pq": pq, "sq": sq, "rq": rq, "tp": total_tp, "fp": total_fp, "fn": total_fn, "n_streams": len(per_stream)}


def attachment_scores(true_segments: dict, pred_segments: dict, true_labels: dict, pred_labels: dict) -> dict:
    """Per-page UAS/LAS, pooled across the whole test set - Demirtas et
    al.'s protocol (see module docstring): a page counts toward UAS iff its
    predicted head page exactly equals its true head page (no IoU-threshold
    tolerance, unlike --metrics pq), and toward LAS if the two segments'
    majority labels also agree."""
    true_head = head_page_lookup(true_segments)
    pred_head = head_page_lookup(pred_segments)

    n_total = n_uas = n_las = 0
    for pdf_id in true_segments:
        t_head_map = true_head.get(pdf_id, {})
        p_head_map = pred_head.get(pdf_id, {})
        t_label_map = true_labels.get(pdf_id, {})
        p_label_map = pred_labels.get(pdf_id, {})
        for t_head, pages in true_segments[pdf_id].items():
            for page in pages:
                n_total += 1
                p_head = p_head_map.get(page)
                if p_head is not None and p_head == t_head_map.get(page):
                    n_uas += 1
                    if t_label_map.get(t_head) == p_label_map.get(p_head):
                        n_las += 1
    return {
        "uas": n_uas / n_total if n_total else 0.0,
        "las": n_las / n_total if n_total else 0.0,
        "n_pages": n_total,
    }


def macro_f1_report(df: pd.DataFrame, true_col: str, pred_col: str, classes: list[str] | None) -> dict:
    true_vals = df[true_col].astype(str)
    pred_vals = df[pred_col].astype(str)
    if classes is None:
        classes = sorted(set(true_vals) | set(pred_vals))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    keep = true_vals.isin(class_to_idx)
    y_true = true_vals[keep].map(class_to_idx)
    y_pred = pred_vals[keep].map(lambda v: class_to_idx.get(v, -1))
    labels = list(range(len(classes)))
    # labels=range(len(classes)): average over the full vocabulary, not
    # just whatever classes appear in this comparison - see train.py's
    # evaluate_sequence for why (silently excluding zero-support classes
    # inflates the score, especially for rare/merged classes). weighted_f1
    # doesn't have the same blind spot (it's support-weighted, so a
    # zero-support class contributes 0 weight either way) but is computed
    # with the same explicit `labels` for consistency.
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    return {
        "accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "n_classes": len(classes), "n_pages": int(keep.sum()),
    }


def start_page_report(df: pd.DataFrame, true_col: str, pred_col: str) -> dict:
    y_true = df[true_col].astype(str).str.strip().str.lower().eq("yes")
    y_pred = df[pred_col].astype(str).str.strip().str.lower().eq("yes")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="predict.py's output TSV")
    parser.add_argument("--ground-truth", type=Path, default=None,
                         help="only needed if --predictions doesn't already carry ground-truth columns "
                              "alongside the predicted_* ones - joined on (--pdf-col, --page-col)")
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS + ["all"], default=["all"])
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="for --metrics pq/las segment matching")
    parser.add_argument("--label-col", choices=["document_type", "layout_type", "functional_category"],
                         default="document_type", help="which task --metrics las scores")
    parser.add_argument("--classes-json", type=Path, default=None,
                         help="a run's classes.json - if given, --metrics macro-f1 averages over the full "
                              "trained vocabulary rather than whatever classes appear in this comparison")
    parser.add_argument("--pdf-col", default="pdf_name")
    parser.add_argument("--page-col", default="page_num")
    parser.add_argument("--true-start-col", default="start_page")
    parser.add_argument("--true-doctype-col", default="document_type")
    parser.add_argument("--pred-start-col", default="predicted_start_page")
    parser.add_argument("--pred-doctype-col", default="predicted_document_type")
    parser.add_argument("--pred-segment-id-col", default="predicted_segment_id")
    parser.add_argument("--out", type=Path, default=None, help="optional: write the report to a file too")
    args = parser.parse_args()

    metrics = ALL_METRICS if "all" in args.metrics else args.metrics

    sep = "\t" if str(args.predictions).endswith(".tsv") else ","
    df = pd.read_csv(args.predictions, sep=sep)

    if args.ground_truth is not None:
        gt_sep = "\t" if str(args.ground_truth).endswith(".tsv") else ","
        gt = pd.read_csv(args.ground_truth, sep=gt_sep)
        gt_cols = [args.pdf_col, args.page_col, args.true_start_col,
                   args.true_doctype_col]
        df = df.merge(gt[gt_cols], on=[args.pdf_col, args.page_col], how="inner", suffixes=("", "_gt_dup"))
        print(f"joined {len(df)} pages against --ground-truth")

    required = [args.pdf_col, args.page_col, args.true_start_col, args.pred_start_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing column(s) in the (possibly joined) data: {missing} - pass --ground-truth if "
                          f"--predictions doesn't already carry ground-truth columns.")

    classes = None
    if args.classes_json:
        all_classes = json.loads(args.classes_json.read_text())
        key = {"document_type": "document_type"}[args.label_col]
        classes = all_classes[key]

    report_lines = [f"{len(df)} pages, {df[args.pdf_col].nunique()} PDFs"]
    print(report_lines[-1])

    if "macro-f1" in metrics:
        report_lines.append("\n--- macro-f1 ---")
        for label, true_col, pred_col in [
            ("document_type", args.true_doctype_col, args.pred_doctype_col)
        ]:
            these_classes = classes if (args.classes_json and label == args.label_col) else None
            r = macro_f1_report(df, true_col, pred_col, these_classes)
            line = (f"{label}: accuracy={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f} "
                    f"weighted_f1={r['weighted_f1']:.3f} (n_classes={r['n_classes']}, n_pages={r['n_pages']})")
            report_lines.append(line)
        start_r = start_page_report(df, args.true_start_col, args.pred_start_col)
        line = f"start_page: precision={start_r['precision']:.3f} recall={start_r['recall']:.3f} f1={start_r['f1']:.3f}"
        report_lines.append(line)
        for line in report_lines[-4:]:
            print(line)

    if "pq" in metrics or "las" in metrics:
        true_segments = segments_from_start_col(df, args.pdf_col, args.page_col, args.true_start_col)
        pred_segments = segments_from_id_col(df, args.pdf_col, args.page_col, args.pred_segment_id_col)

    if "pq" in metrics:
        pq = panoptic_quality(true_segments, pred_segments, args.iou_threshold)
        line = (f"\n--- panoptic quality (van Heusden et al. 2022/2024; segmentation only, "
                f"IoU>{args.iou_threshold}, macro-averaged over {pq['n_streams']} streams) ---\n"
                f"PQ={pq['pq']:.3f}  SQ={pq['sq']:.3f}  RQ={pq['rq']:.3f}  "
                f"(TP={pq['tp']} FP={pq['fp']} FN={pq['fn']} matched/predicted/missed documents, totals)")
        print(line)
        report_lines.append(line)

    if "las" in metrics:
        true_label_col = {"document_type": args.true_doctype_col}[args.label_col]
        pred_label_col = {"document_type": args.pred_doctype_col}[args.label_col]
        true_labels = majority_label_from_segments(df, args.pdf_col, args.page_col, true_label_col, true_segments)
        pred_labels = majority_label_from_segments(df, args.pdf_col, args.page_col, pred_label_col, pred_segments)

        att = attachment_scores(true_segments, pred_segments, true_labels, pred_labels)
        line = (f"\n--- attachment scores (Demirtas et al. 2022; {args.label_col}) ---\n"
                f"UAS={att['uas']:.3f}  LAS={att['las']:.3f}  (n_pages={att['n_pages']})")
        print(line)
        report_lines.append(line)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(report_lines) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
