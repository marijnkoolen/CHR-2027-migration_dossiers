"""
Runs GOT-OCR2.0 (stepfun-ai/GOT-OCR-2.0-hf) on one or more page images:
plain-text OCR, "format" (markdown) OCR, and optionally color/box-guided
region OCR. Confirmed working on Apple Silicon MPS (~20s/page plain OCR on
an M-series Mac) - no CPU fallback needed for this model.

Note: GOT-OCR2 does NOT detect and emit bounding boxes for arbitrary text
on its own - you supply a region (--color or --box) and it OCRs just that
region. For genuine detect-everything-with-coordinates output, see
run_qwen_vl.py instead (slower, but that's a trained-in capability there).

--output-dir writes one file per image (mirroring --input-dir's
subdirectory structure, if used) instead of just printing to stdout -
named after the source image with its extension swapped for .md (--format)
or .txt (plain/region OCR). Pages whose output file is already newer than
the source image are skipped, so an interrupted batch run can be resumed
with the same command instead of redoing already-OCR'd pages (--force
disables this).

Usage:
    python scripts/ocr/run_got_ocr2.py --images page1.png page2.png
    python scripts/ocr/run_got_ocr2.py --images page1.png --format
    python scripts/ocr/run_got_ocr2.py --images page1.png --color green
    python scripts/ocr/run_got_ocr2.py --images page1.png --box 100 100 400 300
    python scripts/ocr/run_got_ocr2.py --input-dir data/images --output-dir data/ocr_got --format
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, GotOcr2ForConditionalGeneration

from io_utils import collect_images, is_up_to_date, output_path

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device: str):
    t0 = time.time()
    model = GotOcr2ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    print(f"loaded model on {device} in {time.time() - t0:.1f}s")
    return model, processor


def run_ocr(model, processor, device: str, image: Image.Image, max_new_tokens: int, **processor_kwargs) -> tuple[str, float]:
    inputs = processor(image, return_tensors="pt", **processor_kwargs).to(device)
    t0 = time.time()
    generate_ids = model.generate(
        **inputs, do_sample=False, tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>", max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0
    text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, nargs="+", help="explicit list of image files")
    source.add_argument("--input-dir", type=Path, help="recursively OCR every image under this directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="write one output file per image here (mirroring --input-dir's subdirectory "
                              "structure, if used) instead of printing to stdout")
    parser.add_argument("--force", action="store_true", help="reprocess pages even if their output is up to date")
    parser.add_argument("--format", action="store_true", help="markdown-preserving output instead of plain text")
    parser.add_argument("--color", choices=["red", "green", "blue"], default=None,
                         help="only OCR the region inside a box of this color drawn on the image")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), default=None,
                         help="only OCR this pixel region")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", default=None, help="default: mps if available, else cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"torch MPS available: {torch.backends.mps.is_available()}")
    print(f"using device: {device}")

    try:
        model, processor = load_model(device)
    except Exception as e:
        print(f"failed on {device} ({e!r}), falling back to cpu")
        device = "cpu"
        model, processor = load_model(device)

    processor_kwargs = {}
    if args.format:
        processor_kwargs["format"] = True
    if args.color:
        processor_kwargs["color"] = args.color
    if args.box:
        processor_kwargs["box"] = args.box
    out_suffix = ".md" if args.format else ".txt"

    pairs = collect_images(args.images, args.input_dir)
    print(f"{len(pairs)} image(s) found")

    n_done, n_skipped = 0, 0
    for i, (src, relative_path) in enumerate(pairs):
        dst = output_path(args.output_dir, relative_path, out_suffix) if args.output_dir else None
        if dst and not args.force and is_up_to_date(src, dst):
            n_skipped += 1
            continue

        image = Image.open(src).convert("RGB")
        text, elapsed = run_ocr(model, processor, device, image, args.max_new_tokens, **processor_kwargs)
        n_done += 1

        if dst:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
            print(f"[{i + 1}/{len(pairs)}] [{elapsed:.1f}s] {src} -> {dst}")
        else:
            print(f"\n=== {src} ===\n[{elapsed:.1f}s]\n{text}")

    print(f"\n{n_done} page(s) OCR'd, {n_skipped} already up to date (skipped)")


if __name__ == "__main__":
    main()
