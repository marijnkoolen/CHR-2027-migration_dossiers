"""
Builds synthetic training PDF sequences by regrouping/reordering real,
intact documents rather than individual pages - directly increasing the
number of distinct sequence *compositions* the sequence-context model's
Transformer encoder trains on. Per sequence_model.py's module docstring:
the encoder's own weights are shaped by only as many independent examples
as there are real training PDFs, regardless of how many labeled pages or
documents happen to be packed inside them. --augment-passes (see
precompute_embeddings.py) varies each page's *appearance* but keeps every
copy's underlying document-boundary structure identical, so it doesn't
touch this at all; recomposing which documents go into which sequence does.

Since each document's own pages/labels/internal order are kept completely
intact - only which order documents appear in, or which documents get
grouped into one sequence, changes - every synthetic sequence is built
entirely from real, valid page/label combinations. Only the train split is
ever touched; val/test are never recomposed, since they need to stay
exactly the real, unmodified evaluation set.

Two modes:
    "shuffle": for each real training PDF, reorders its own documents only
        (never mixes across PDFs). Cheap and conservative - still capped at
        39-ish distinct source groupings, just more orderings of each.
    "recombine": pools every non-cover document across *all* real training
        PDFs and draws new random groupings from that shared pool -
        combinatorially many more distinct sequences than shuffle allows,
        since it isn't capped at reordering within fixed per-PDF groups.

cover_doctype (default "NAA cover"): if given and present, documents of
this type are always pinned to the very first position of a synthetic
sequence, never shuffled/omitted/mixed into the general pool - matching
real dossiers here, which always begin with a post-digitisation cover page
added by the archive, not part of the original document order.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from flag_prediction_errors import segment_ids_from_start_col


def build_documents(manifest: pd.DataFrame, pdf_col: str, page_col: str, doctype_col: str, start_col: str) -> list[dict]:
    """Segments a (single-split) page-level manifest into a flat list of
    documents: each {"source_pdf_id", "doctype", "rows"} - "rows" is that
    document's own page rows, sorted by page_col, completely untouched."""
    documents = []
    for pdf_id, group in manifest.groupby(pdf_col, sort=False):
        seg_ids = segment_ids_from_start_col(group, page_col, start_col)
        ordered = group.sort_values(page_col)
        for seg_id, seg in ordered.groupby(seg_ids.values):
            documents.append({
                "source_pdf_id": pdf_id,
                "doctype": seg[doctype_col].mode().iat[0],
                "rows": seg,
            })
    return documents


def assemble_sequence(doc_list: list[dict], new_pdf_id: str, pdf_col: str, page_col: str,
                       start_col: str, split_col: str) -> pd.DataFrame:
    """Concatenates documents (from build_documents) into one new synthetic
    PDF's worth of rows, with fresh page_col/start_col/pdf_col/split_col
    values - every other column (image/pagexml paths, row_id, doctype/
    layout/functional labels) passes through unchanged from the real page
    it was copied from."""
    rows = []
    page_num = 1
    for doc in doc_list:
        for i, (_, row) in enumerate(doc["rows"].iterrows()):
            new_row = row.to_dict()
            new_row[pdf_col] = new_pdf_id
            new_row[page_col] = page_num
            new_row[start_col] = "yes" if i == 0 else "no"
            new_row[split_col] = "train"
            rows.append(new_row)
            page_num += 1
    return pd.DataFrame(rows)


def recompose_documents(
    manifest: pd.DataFrame, pdf_col: str, page_col: str, doctype_col: str, start_col: str, split_col: str,
    mode: str, passes: int, cover_doctype: str | None = "NAA cover", seed: int = 0,
) -> pd.DataFrame:
    """Returns *additional* rows (not the originals) - `passes` synthetic
    recomposed train-split PDF sequences, built by regrouping/reordering
    real, intact documents. Concatenate the result onto the original
    manifest; never replaces it."""
    if mode not in ("shuffle", "recombine"):
        raise ValueError(f"mode must be 'shuffle' or 'recombine', got {mode!r}")

    rng = random.Random(seed)
    train_rows = manifest[manifest[split_col] == "train"]
    documents = build_documents(train_rows, pdf_col, page_col, doctype_col, start_col)
    is_cover = lambda d: cover_doctype is not None and d["doctype"] == cover_doctype

    synthetic = []

    if mode == "shuffle":
        by_pdf: dict[str, list[dict]] = {}
        for doc in documents:
            by_pdf.setdefault(doc["source_pdf_id"], []).append(doc)
        for pass_idx in range(1, passes + 1):
            for source_pdf_id, docs in by_pdf.items():
                covers = [d for d in docs if is_cover(d)]
                rest = [d for d in docs if not is_cover(d)]
                rng.shuffle(rest)
                new_pdf_id = f"{source_pdf_id}_shuffle{pass_idx}"
                synthetic.append(assemble_sequence(covers + rest, new_pdf_id, pdf_col, page_col, start_col, split_col))

    else:  # recombine
        cover_pool = [d for d in documents if is_cover(d)]
        noncover_pool = [d for d in documents if not is_cover(d)]
        # Match the real per-PDF document-count distribution, so synthetic
        # sequences look like plausible dossiers rather than arbitrary-length
        # grab-bags - sampling directly from the real counts needs no extra
        # tuning knob and reflects whatever shape that distribution has.
        real_doc_counts = [sum(1 for d in documents if d["source_pdf_id"] == pid) for pid in
                            dict.fromkeys(d["source_pdf_id"] for d in documents)]
        n_synthetic = passes * len(real_doc_counts)
        for i in range(n_synthetic):
            n_docs = rng.choice(real_doc_counts)
            n_noncover = max(1, n_docs - 1) if cover_pool else n_docs
            chosen = rng.sample(noncover_pool, k=min(n_noncover, len(noncover_pool))) if noncover_pool else []
            doc_list = ([rng.choice(cover_pool)] if cover_pool else []) + chosen
            new_pdf_id = f"synthetic{seed}_{i:05d}"
            synthetic.append(assemble_sequence(doc_list, new_pdf_id, pdf_col, page_col, start_col, split_col))

    return pd.concat(synthetic, ignore_index=True) if synthetic else manifest.iloc[0:0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="an embeddings_manifest.tsv from precompute_embeddings.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["shuffle", "recombine"], default="recombine")
    parser.add_argument("--passes", type=int, default=5,
                         help="shuffle: this many reshuffled copies per real training PDF. recombine: this many "
                              "synthetic sequences per real training PDF's worth of documents (i.e. total "
                              "synthetic sequences = passes * n_real_train_pdfs).")
    parser.add_argument("--cover-doctype", default="NAA cover",
                         help="document type always pinned to position 0 of a synthetic sequence, if present. "
                              "Pass '' to disable pinning.")
    parser.add_argument("--pdf-col", default="pdf_id")
    parser.add_argument("--page-col", default="page_number")
    parser.add_argument("--doctype-col", default="document_type")
    parser.add_argument("--start-col", default="start_page")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sep = "\t" if str(args.manifest).endswith(".tsv") else ","
    manifest = pd.read_csv(args.manifest, sep=sep)
    print(f"{len(manifest)} rows, {manifest[args.pdf_col].nunique()} PDFs "
          f"({(manifest[args.split_col] == 'train').sum()} train rows)")

    synthetic = recompose_documents(
        manifest, args.pdf_col, args.page_col, args.doctype_col, args.start_col, args.split_col,
        mode=args.mode, passes=args.passes, cover_doctype=(args.cover_doctype or None), seed=args.seed,
    )
    print(f"added {synthetic[args.pdf_col].nunique()} synthetic train PDFs ({len(synthetic)} rows), mode={args.mode}")

    # row_id (if present) must stay exactly as copied from the source row -
    # it's a pointer into embeddings.npy, which this script never touches;
    # renumbering it would point synthetic (and even original) rows at the
    # wrong embedding vectors.
    combined = pd.concat([manifest, synthetic], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, sep=sep, index=False)
    print(f"Wrote {args.out} ({len(combined)} rows, {combined[args.pdf_col].nunique()} PDFs total)")


if __name__ == "__main__":
    main()
