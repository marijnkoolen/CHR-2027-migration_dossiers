"""
Locates fields on preprinted forms by text-anchor alignment, rather than
asking a VLM to detect+transcribe every block on every page (which
run_qwen_vl.py/benchmark_vllm.py found the 3B model unreliable at on dense
pages - see conversation notes). Once a page's document_type is already
known (from train.py's classifier), and the form's *preprinted* structure
is stable across instances, a field's position can be found relative to
nearby preprinted anchor text, rather than needing per-page VLM grounding
at all:

    1. Calibrate a FormTemplate once per form type, from a few known-good
       instances: pick stable preprinted anchor strings (e.g. "Date:",
       "Supervisor/Manager:") and their positions, plus the target field
       boxes you want (e.g. the date field), defined relative to an anchor.
    2. For a new page of that type: detect text+boxes again, fuzzy-match
       against the template's anchor strings (tolerant of OCR noise),
       fit a geometric transform from however many anchors were actually
       found, and project the template's field boxes through that
       transform to get this page's field boxes.
    3. Crop and OCR just those small boxes - a good fit for
       run_got_ocr2.py's --box mode, which needs a region supplied (it
       can't detect boxes itself) - this step supplies it.

This is anchored on *detected text*, not visual/pixel features, so it's
robust to the background variation you described (cardboard folder vs.
scanning table vs. white/black borders) almost by construction - the
transform is fit purely from where anchor text was found, never touching
surrounding pixels. The residual risk is text-detection robustness at the
form's edges (clutter/occlusion), not background per se.

Text detection here uses Tesseract (via pytesseract - `pip install
pytesseract`, needs the `tesseract` binary, already on this machine) since
preprinted form labels are clean printed text, which classical OCR handles
well and fast (CPU-only, no GPU, scales easily to 114K pages for this
detection-only step - a much cheaper pass than any VLM call). Swap in
another detector if preferred; only detect_text_boxes() would need to
change.

Assumes flat scanning (translation/rotation/scale only - cv2's partial
affine, estimated via RANSAC to reject bad anchor matches). If your pages
are photographed at an angle rather than scanned flat, swap
estimate_transform's cv2.estimateAffinePartial2D for cv2.findHomography
and project_point's cv2.transform for cv2.perspectiveTransform.

STATUS: the geometric core (find_anchors/estimate_transform/
project_field_boxes) is verified against a synthetic example at the
bottom of this file (`python scripts/ocr/form_registration.py --selftest`)
- no real form data needed for that. detect_text_boxes() is written
against pytesseract's documented API but untested against your actual
scans - verify it finds your anchor strings reliably before trusting the
rest of the pipeline on real data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz, process

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class FormTemplate:
    name: str
    anchors: dict[str, tuple[float, float]]  # anchor string -> reference (x, y), e.g. box center
    fields: dict[str, Box]  # field name -> reference box, same coordinate space as anchors


@dataclass
class DetectedText:
    text: str
    box: Box

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


def detect_text_boxes(image_path: Path, min_confidence: int = 40) -> list[DetectedText]:
    """Word-level (text, box) pairs via Tesseract. Untested against real
    scans - see module docstring."""
    import pytesseract

    data = pytesseract.image_to_data(str(image_path), output_type=pytesseract.Output.DICT)
    results = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or int(data["conf"][i]) < min_confidence:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        results.append(DetectedText(text=text, box=(x, y, x + w, y + h)))
    return results


def find_anchors(
    detected: list[DetectedText], anchor_strings: list[str], score_cutoff: float = 80.0,
) -> dict[str, tuple[float, float]]:
    """{anchor_string: observed (x, y)} for whichever anchors were found -
    fuzzy substring match (rapidfuzz partial_ratio) to tolerate OCR noise
    and preprinted labels split across multiple detected words. Only the
    anchors actually found in this page are returned; the caller decides
    whether enough were found to fit a transform (>=3 for an affine fit,
    more for a robust RANSAC fit)."""
    candidates = [d.text for d in detected]
    found = {}
    for anchor in anchor_strings:
        match = process.extractOne(anchor, candidates, scorer=fuzz.partial_ratio, score_cutoff=score_cutoff)
        if match is not None:
            matched_text, _score, idx = match
            found[anchor] = detected[idx].center
    return found


def estimate_transform(
    reference_points: np.ndarray, observed_points: np.ndarray,
) -> np.ndarray | None:
    """2x3 affine matrix mapping reference (template) coordinates to
    observed (this page's) coordinates, or None if too few correspondences
    (need >=3). Uses RANSAC when there are enough points to make outlier
    rejection meaningful (>=4) - guards against one fuzzy-matched anchor
    having matched the wrong occurrence of a common word."""
    if len(reference_points) < 3:
        return None
    method = cv2.RANSAC if len(reference_points) >= 4 else cv2.LMEDS
    transform, _inliers = cv2.estimateAffinePartial2D(
        reference_points.astype(np.float32), observed_points.astype(np.float32), method=method,
    )
    return transform


def project_point(transform: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    result = cv2.transform(np.array([[point]], dtype=np.float32), transform)
    return float(result[0, 0, 0]), float(result[0, 0, 1])


def project_field_boxes(template: FormTemplate, transform: np.ndarray) -> dict[str, Box]:
    projected = {}
    for name, (x1, y1, x2, y2) in template.fields.items():
        px1, py1 = project_point(transform, (x1, y1))
        px2, py2 = project_point(transform, (x2, y2))
        projected[name] = (px1, py1, px2, py2)
    return projected


def locate_fields(template: FormTemplate, image_path: Path, min_anchors: int = 3) -> dict[str, Box] | None:
    """End-to-end: detect text on `image_path`, match against
    `template`'s anchors, fit a transform, project field boxes. Returns
    None if fewer than `min_anchors` anchors were found (not enough to
    trust a fit) - the caller should fall back to flagging the page for
    manual review or a full VLM pass rather than trusting a bad transform."""
    detected = detect_text_boxes(image_path)
    observed = find_anchors(detected, list(template.anchors.keys()))
    if len(observed) < min_anchors:
        return None

    reference_points = np.array([template.anchors[a] for a in observed])
    observed_points = np.array([observed[a] for a in observed])
    transform = estimate_transform(reference_points, observed_points)
    if transform is None:
        return None
    return project_field_boxes(template, transform)


# --------------------------------------------------------------------------
# Self-test: synthetic example verifying the geometric core, independent of
# Tesseract/real form data.
# --------------------------------------------------------------------------

def _selftest():
    # Reference template: three anchors forming an L, one field box
    # defined relative to them.
    template = FormTemplate(
        name="synthetic",
        anchors={"Name:": (100.0, 100.0), "Date:": (400.0, 100.0), "Supervisor:": (100.0, 300.0)},
        fields={"date_field": (420.0, 90.0, 520.0, 115.0)},  # just right of "Date:"
    )

    # Simulate an observed page: same content translated +50/+30, rotated
    # ~5 degrees, scaled 0.97x - like a real scan/photo of the same form,
    # nothing to do with the (irrelevant, per module docstring) background.
    angle = np.radians(5)
    scale = 0.97
    true_transform = np.array([
        [scale * np.cos(angle), -scale * np.sin(angle), 50.0],
        [scale * np.sin(angle), scale * np.cos(angle), 30.0],
    ])

    def apply_true_transform(pt):
        x, y = pt
        return true_transform[0, 0] * x + true_transform[0, 1] * y + true_transform[0, 2], \
               true_transform[1, 0] * x + true_transform[1, 1] * y + true_transform[1, 2]

    observed_detections = [
        DetectedText(text=anchor, box=(*apply_true_transform((ax - 10, ay - 8)), *apply_true_transform((ax + 10, ay + 8))))
        for anchor, (ax, ay) in template.anchors.items()
    ]
    # A distractor: same word appearing elsewhere, to check RANSAC/fuzzy
    # matching doesn't get confused (here it just tests find_anchors picks
    # a plausible match - real RANSAC robustness needs >=4 real anchors,
    # which this 3-anchor example doesn't exercise).
    observed_detections.append(DetectedText(text="Date", box=(900, 900, 950, 920)))

    found = find_anchors(observed_detections, list(template.anchors.keys()))
    assert set(found) == set(template.anchors), f"expected to find all 3 anchors, got {set(found)}"

    reference_points = np.array([template.anchors[a] for a in found])
    observed_points = np.array([found[a] for a in found])
    transform = estimate_transform(reference_points, observed_points)
    assert transform is not None, "transform estimation failed"

    projected = project_field_boxes(template, transform)
    expected = (
        *apply_true_transform((420.0, 90.0)),
        *apply_true_transform((520.0, 115.0)),
    )
    got = projected["date_field"]
    max_error = max(abs(g - e) for g, e in zip(got, expected))
    print(f"expected date_field box: {tuple(round(v, 1) for v in expected)}")
    print(f"projected date_field box: {tuple(round(v, 1) for v in got)}")
    print(f"max coordinate error: {max_error:.2f}px")
    assert max_error < 2.0, f"projected field box off by {max_error:.2f}px - geometric core has a bug"
    print("\nself-test passed: anchor matching + transform estimation + field projection are correct")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="run the synthetic geometric-core test")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        parser.print_help()
