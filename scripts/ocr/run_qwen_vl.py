"""
Runs Qwen2.5-VL-Instruct on one or more page images, with two prompt modes:
markdown-structured transcription, and detect-everything-with-bounding-boxes
(a trained-in grounding capability GOT-OCR2 doesn't have - see
run_got_ocr2.py, which can only OCR a region you already specify).

MPS: confirmed broken for this model - it hard-crashes with a native
assertion ("[MPSTemporaryNDArray ...] total bytes of NDArray > 2**32", a
real Metal single-buffer size limit hit by an intermediate attention
tensor in the vision tower), not a catchable Python exception, so this
always runs on CPU regardless of MPS availability. On an M-series Mac CPU,
expect ~2min/page for markdown transcription and ~10min/page for the
bounding-box mode - fine for a spot-check, not for real volume. For actual
throughput, run this on the A10 server instead (pass --device cuda there).

Usage:
    python scripts/ocr/run_qwen_vl.py --images page1.png page2.png
    python scripts/ocr/run_qwen_vl.py --images page1.png --mode bbox
    python scripts/ocr/run_qwen_vl.py --images page1.png --model Qwen/Qwen2.5-VL-7B-Instruct --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MARKDOWN_PROMPT = (
    "Transcribe all text in this image exactly as written, including handwritten fill-ins. "
    "Output as markdown, preserving the document's structure (headings, form fields, layout order)."
)
BBOX_PROMPT = (
    "Detect every distinct block of text in this image. For each, output its exact transcription "
    "and its bounding box. Respond as a JSON list of objects with keys \"text\" and \"bbox_2d\" "
    "(bbox_2d = [x1, y1, x2, y2] in pixel coordinates)."
)


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    # MPS hard-crashes on this model (native assertion, not catchable in
    # Python) - see module docstring - so never auto-select it here.
    return "cpu"


def load_model(model_id: str, device: str):
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"loaded {model_id} on {device} in {time.time() - t0:.1f}s")
    return model, processor


def run_prompt(model, processor, device: str, image: Image.Image, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
    ).to(device)

    t0 = time.time()
    generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    elapsed = time.time() - t0
    text = processor.batch_decode(
        generate_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    return text, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=["markdown", "bbox", "both"], default="both")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", default=None, help="default: cpu (MPS crashes on this model - see docstring)")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"torch MPS available: {torch.backends.mps.is_available()}")
    print(f"using device: {device}")

    model, processor = load_model(args.model, device)

    modes = ["markdown", "bbox"] if args.mode == "both" else [args.mode]
    prompts = {"markdown": MARKDOWN_PROMPT, "bbox": BBOX_PROMPT}

    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        for mode in modes:
            print(f"\n=== {image_path} [{mode}] ===")
            text, elapsed = run_prompt(model, processor, device, image, prompts[mode], args.max_new_tokens)
            print(f"[{elapsed:.1f}s]\n{text}")


if __name__ == "__main__":
    main()
