"""
extract_dates.py
----------------
Extract every text line that contains a recognisable date from the PageXML
files that correspond to pages listed in annotations.

Results are written to:
  data/dates/<dossier-id>.jsonl  – one JSON object per date-bearing line
  data/dates/_all_dates.jsonl    – combined file across all annotated dossiers

Run from the workspace root:
    python scripts/extract_dates.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from pagexml.parser import parse_pagexml_file


_EN_MONTHS = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)

_NL_MONTHS = (
    "Januari|Februari|Maart|April|Mei|Juni|"
    "Juli|Augustus|September|Oktober|November|December|"
    "Jan|Feb|Mrt|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)

_MONTH_NAMES = f"(?:{_EN_MONTHS}|{_NL_MONTHS})"

DATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "15 February 1909"  /  "29 JULI 1956"  /  "14 Juli 1953"
    (
        "day_monthname_year",
        re.compile(
            rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH_NAMES})\s+(?P<year>\d{{2,4}})\b",
            re.IGNORECASE,
        ),
    ),
    # "February 15, 1909"
    (
        "monthname_day_year",
        re.compile(
            rf"\b(?P<month>{_MONTH_NAMES})\s+(?P<day>\d{{1,2}}),?\s+(?P<year>\d{{4}})\b",
            re.IGNORECASE,
        ),
    ),
    # "February 1909"  (no day – still useful for context)
    (
        "monthname_year",
        re.compile(
            rf"\b(?P<month>{_MONTH_NAMES})\s+(?P<year>\d{{4}})\b",
            re.IGNORECASE,
        ),
    ),
    # DD-MM-YYYY  /  DD/MM/YYYY  /  DD.MM.YYYY  (and short years like 13-7-'56)
    (
        "numeric_dmy",
        re.compile(
            r"\b(?P<day>\d{1,2})[-/.]+(?P<month>\d{1,2})[-/.]+(?P<year>\d{2,4})\b"
        ),
    ),
    # YYYY-MM-DD  (ISO)
    (
        "numeric_ymd",
        re.compile(
            r"\b(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})\b"
        ),
    ),
]


def find_dates(text: str) -> list[dict]:
    """Return a list of date-match dicts found in *text*."""
    hits: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern_name, pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            # skip if fully covered by an earlier (higher-priority) match
            if any(s <= span[0] and span[1] <= e for s, e in seen_spans):
                continue
            seen_spans.add(span)
            gd = m.groupdict()
            hits.append(
                {
                    "pattern": pattern_name,
                    "date_string": m.group(0),
                    "day": gd.get("day"),
                    "month": gd.get("month"),
                    "year": gd.get("year"),
                    "span_start": span[0],
                    "span_end": span[1],
                }
            )

    return hits


def coords_to_dict(coords) -> dict:
    return {
        "x": coords.x,
        "y": coords.y,
        "w": coords.w,
        "h": coords.h,
        "left": coords.left,
        "top": coords.top,
        "right": coords.right,
        "bottom": coords.bottom,
        "points": coords.points,
    }


def load_annotations(csv_path: Path) -> dict[str, dict[str, dict]]:
    """Load all_annotations.csv.

    Returns a nested dict:  dossier_id -> page_number_str -> annotation_row
    The dossier key has the .pdf suffix stripped.
    """
    index: dict[str, dict[str, dict]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dossier = row["dossier"].removesuffix(".pdf")
            page = row["page_number"]
            index.setdefault(dossier, {})[page] = row
    return index


def process_page(
    xml_path: Path,
    dossier_id: str,
    page_name: str,
) -> list[dict]:

    try:
        scan = parse_pagexml_file(str(xml_path))
    except Exception as exc:
        print(f"  [WARN] could not parse {xml_path}: {exc}", file=sys.stderr)
        return []

    ann_meta: dict = {}

    records: list[dict] = []
    for line in scan.get_lines():
        text = line.text
        if not text:
            continue

        date_hits = find_dates(text)
        if not date_hits:
            continue

        coords_dict = coords_to_dict(line.coords) if line.coords else None

        for hit in date_hits:
            records.append(
                {
                    "dossier": dossier_id,
                    "page": page_name,
                    "line_id": line.id,
                    "line_text": text,
                    "coords": coords_dict,
                    **ann_meta,
                    **hit,
                }
            )

    return records


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_root = root / "extract-text-per-page"
    annotations_csv = root / "data" / "annotations" / "all_annotations.csv"
    output_root = root / "data" / "dates"
    output_root.mkdir(parents=True, exist_ok=True)

    # Load annotation index: dossier -> page_number -> metadata row
    annotations = load_annotations(annotations_csv)
    annotated_dossiers = sorted(annotations.keys())
    print(f"Loaded {sum(len(v) for v in annotations.values())} annotated pages "
          f"across {len(annotated_dossiers)} dossiers from {annotations_csv.name}")

    all_records: list[dict] = []
    missing_xml = []

    for dossier_id in annotated_dossiers:
        page_dir = input_root / dossier_id / "page"
        if not page_dir.is_dir():
            missing_xml.append(dossier_id)
            continue

        page_annotations = annotations[dossier_id]
        dossier_records: list[dict] = []

        for page_num_str in sorted(page_annotations, key=int):
            page_name = f"page_{int(page_num_str):04d}"
            xml_path = page_dir / f"{page_name}.xml"

            if not xml_path.exists():
                print(f"  [WARN] XML not found: {xml_path}", file=sys.stderr)
                continue

            hits = process_page(xml_path, dossier_id, page_name)
            dossier_records.extend(hits)

        # write per-dossier JSONL
        out_path = output_root / f"{dossier_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in dossier_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        all_records.extend(dossier_records)
        print(
            f"  {dossier_id}: {len(page_annotations)} pages, "
            f"{len(dossier_records)} date hits"
        )

    if missing_xml:
        print(f"\n[WARN] No XML pages found for {len(missing_xml)} dossiers: "
              f"{missing_xml}", file=sys.stderr)

    # write combined JSONL
    combined_path = output_root / "_all_dates.jsonl"
    with combined_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"\nDone. {len(all_records)} total date hits across "
        f"{len(annotated_dossiers) - len(missing_xml)} dossiers "
        f"→ {combined_path}"
    )


if __name__ == "__main__":
    main()
