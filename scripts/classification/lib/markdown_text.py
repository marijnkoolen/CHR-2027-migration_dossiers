"""
Extracts plain text from Qwen2.5-VL markdown-mode OCR output (see
scripts/ocr/run_qwen_vl.py/benchmark_vllm.py), as an alternative text
source to pagexml.py's PageXML transcriptions.

Strips markdown markup (headers, bold/italic, tables, lists) rather than
feeding it to the text backbone as-is: bert-base-uncased and this
project's other text backbones were pretrained on natural prose, not
markdown, so syntax characters mostly just consume --max-text-length
token budget without conveying learned signal, and the "structure" they
carry isn't a reliable feature anyway - different OCR runs render the same
kind of content differently (this project's own GOT-OCR2 output leaned
LaTeX-style headers, Qwen's leaned markdown tables) - real layout
structure is far better captured by the vision modality directly.

Renders markdown -> HTML -> plain text (rather than a regex-only strip)
so nested constructs (tables, lists) are handled correctly rather than
mangled.
"""

from __future__ import annotations

from pathlib import Path

import markdown
from bs4 import BeautifulSoup


def extract_text(markdown_path: str | Path) -> str:
    """Returns "" if the path is empty/missing rather than raising, same
    convention as pagexml.py's extract_text - some pages may legitimately
    have no OCR output (e.g. blank pages)."""
    if not markdown_path or not Path(markdown_path).exists():
        return ""
    raw = Path(markdown_path).read_text(encoding="utf-8")
    if not raw.strip():
        return ""
    html = markdown.markdown(raw, extensions=["tables"])
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    # Collapse the blank-line runs markdown/BeautifulSoup's block-element
    # spacing tends to leave behind, without losing genuine paragraph breaks.
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
