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

CUDA memory: defaults to bf16 weights (half the size of the fp32 used on
the cpu/mps path above - fp32 isn't needed on CUDA and doubles weight
memory for no benefit) and `sdpa` attention (`eager`, used above only
because flash-attention needs CUDA, materializes the full [heads,
seq_len, seq_len] attention matrix in memory - O(n^2) in token count, and
by far the biggest lever if you see an OOM: a real high-DPI page scan
produces far more image tokens than a small test image, so eager
attention's memory cost can blow up disproportionately to page size).
If flash-attn is installed, --attn-implementation flash_attention_2 is
more memory-efficient still. --max-pixels caps the image token budget
directly (Qwen's "dynamic resolution" scales token count with image size)
if a resolution cap turns out to matter more than the attention kernel.

--output-dir writes one file per (image, mode) instead of printing to
stdout - <image_stem>.markdown.md and/or <image_stem>.bbox.json, mirroring
--input-dir's subdirectory structure if used. Each (image, mode) pair is
skipped independently if its output file is already newer than the source
image, so an interrupted batch (or a later "also run bbox mode on what I
already OCR'd for markdown" pass) only redoes what's actually missing
(--force disables this).

Usage:
    python scripts/ocr/run_qwen_vl.py --images page1.png page2.png --device cuda
    python scripts/ocr/run_qwen_vl.py --images page1.png --mode bbox --device cuda
    python scripts/ocr/run_qwen_vl.py --images page1.png --model Qwen/Qwen2.5-VL-7B-Instruct --device cuda
    python scripts/ocr/run_qwen_vl.py --input-dir data/images --output-dir data/ocr_qwen --device cuda
    CUDA_VISIBLE_DEVICES=0 python scripts/ocr/run_qwen_vl.py --input-dir data/images/batch_a --output-dir data/ocr_qwen --device cuda &
    CUDA_VISIBLE_DEVICES=1 python scripts/ocr/run_qwen_vl.py --input-dir data/images/batch_b --output-dir data/ocr_qwen --device cuda &
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from io_utils import collect_images, is_up_to_date, output_path

OUTPUT_SUFFIX = {"markdown": ".markdown.md", "bbox": ".bbox.json"}

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


def load_model(model_id: str, device: str, dtype: str | None, attn_implementation: str | None,
                min_pixels: int | None, max_pixels: int | None):
    if dtype is None:
        dtype = "bfloat16" if device == "cuda" else "float32"
    if attn_implementation is None:
        # eager is the only thing guaranteed to work on cpu/mps; sdpa (built
        # into torch, no extra install) avoids eager's O(n^2) materialized
        # attention matrix and is the right default once on CUDA.
        attn_implementation = "sdpa" if device == "cuda" else "eager"

    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=getattr(torch, dtype), attn_implementation=attn_implementation
    ).to(device)
    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    print(f"loaded {model_id} on {device} (dtype={dtype}, attn={attn_implementation}) in {time.time() - t0:.1f}s")
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, nargs="+", help="explicit list of image files")
    source.add_argument("--input-dir", type=Path, help="recursively OCR every image under this directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="write one file per (image, mode) here (mirroring --input-dir's subdirectory "
                              "structure, if used) instead of printing to stdout")
    parser.add_argument("--force", action="store_true", help="reprocess pages even if their output is up to date")
    parser.add_argument("--mode", choices=["markdown", "bbox", "both"], default="both")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", default=None, help="default: cpu (MPS crashes on this model - see docstring)")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None,
                         help="default: bfloat16 on cuda, float32 elsewhere")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None,
                         help="default: sdpa on cuda, eager elsewhere - see docstring re: OOM on CUDA")
    parser.add_argument("--min-pixels", type=int, default=None, help="caps the image token budget - see docstring")
    parser.add_argument("--max-pixels", type=int, default=None, help="caps the image token budget - see docstring")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"torch MPS available: {torch.backends.mps.is_available()}")
    print(f"using device: {device}")

    model, processor = load_model(
        args.model, device, args.dtype, args.attn_implementation, args.min_pixels, args.max_pixels
    )

    modes = ["markdown", "bbox"] if args.mode == "both" else [args.mode]
    prompts = {"markdown": MARKDOWN_PROMPT, "bbox": BBOX_PROMPT}

    pairs = collect_images(args.images, args.input_dir)
    print(f"{len(pairs)} image(s), {len(modes)} mode(s) -> up to {len(pairs) * len(modes)} output(s)")

    n_done, n_skipped, image, image_src = 0, 0, None, None
    for i, (src, relative_path) in enumerate(pairs):
        for mode in modes:
            dst = output_path(args.output_dir, relative_path, OUTPUT_SUFFIX[mode]) if args.output_dir else None
            if dst and not args.force and is_up_to_date(src, dst):
                n_skipped += 1
                continue

            if image is None or image_src != src:
                image, image_src = Image.open(src).convert("RGB"), src
            text, elapsed = run_prompt(model, processor, device, image, prompts[mode], args.max_new_tokens)
            n_done += 1

            if dst:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(text)
                print(f"[{i + 1}/{len(pairs)}] [{mode}] [{elapsed:.1f}s] {src} -> {dst}")
            else:
                print(f"\n=== {src} [{mode}] ===\n[{elapsed:.1f}s]\n{text}")

    print(f"\n{n_done} output(s) generated, {n_skipped} already up to date (skipped)")


if __name__ == "__main__":
    main()
