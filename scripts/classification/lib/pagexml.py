"""Minimal PageXML reader: extracts a page's transcribed text.

Namespace-agnostic (matches on local tag name, not a hardcoded namespace
URI) since PAGE XML has several schema versions in the wild depending on
the tool that produced it (Transkribus, Loghi, eScriptorium, ...).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _line_text(text_line_elem) -> str:
    """A TextLine's TextEquiv/Unicode text. A line can have several
    TextEquiv children (transcription alternates, indexed) - the first one
    found is used, which is the PAGE-XML convention for the primary reading."""
    for child in text_line_elem:
        if _local_tag(child.tag) != "TextEquiv":
            continue
        for grandchild in child:
            if _local_tag(grandchild.tag) == "Unicode" and grandchild.text:
                return grandchild.text.strip()
    return ""


def extract_text(pagexml_path: str | Path) -> str:
    """Concatenates every TextLine's transcribed text, in document order.
    Returns "" if the path is empty/missing rather than raising, since some
    pages (e.g. photos) may legitimately have no transcription."""
    if not pagexml_path or not Path(pagexml_path).exists():
        return ""
    tree = ET.parse(pagexml_path)
    lines = [_line_text(elem) for elem in tree.iter() if _local_tag(elem.tag) == "TextLine"]
    return "\n".join(line for line in lines if line)
