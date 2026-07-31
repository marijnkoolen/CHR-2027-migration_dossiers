"""
Runs GOT-OCR2.0 (stepfun-ai/GOT-OCR-2.0-hf) on one or more page images:
plain-text OCR, "format" (markdown) OCR, and optionally color/box-guided
region OCR. Confirmed working on Apple Silicon MPS (~20s/page plain OCR on
an M-series Mac) - no CPU fallback needed for this model.

Note: GOT-OCR2 does NOT detect and emit bounding boxes for arbitrary text
on its own - you supply a region (--color or --box) and it OCRs just that
region. For genuine detect-everything-with-coordinates output, see
run_qwen_vl.py instead (slower, but that's a trained-in capability there).

Usage:
    python scripts/ocr/run_got_ocr2.py --images page1.png page2.png
    python scripts/ocr/run_got_ocr2.py --images page1.png --format
    python scripts/ocr/run_got_ocr2.py --images page1.png --color green
    python scripts/ocr/run_got_ocr2.py --images page1.png --box 100 100 400 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, GotOcr2ForConditionalGeneration

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
    parser.add_argument("--images", type=Path, nargs="+", required=True)
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

    for image_path in args.images:
        print(f"\n=== {image_path} ===")
        image = Image.open(image_path).convert("RGB")
        text, elapsed = run_ocr(model, processor, device, image, args.max_new_tokens, **processor_kwargs)
        print(f"[{elapsed:.1f}s]\n{text}")


if __name__ == "__main__":
    main()
