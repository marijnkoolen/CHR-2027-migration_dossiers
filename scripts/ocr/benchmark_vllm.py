"""
Benchmarks (and can also drive a real production run of) Qwen2.5-VL OCR
through a vLLM server's OpenAI-compatible API, instead of the one-page-at-
a-time transformers.generate() loop in run_qwen_vl.py. vLLM's continuous
batching lets many pages' decoding overlap on the GPU concurrently rather
than finishing one page's full generate() call before starting the next -
the whole point of this script is to measure how much that's actually
worth for this workload, at a few different concurrency levels, before
committing the full 114K-page run to it.

Not runnable here (no CUDA locally) - written against vLLM's documented
OpenAI-compatible API and smoke-tested for import/argument-parsing
correctness only. Verify the request/response handling against a real
server before trusting it for the full run.

Setup (on the A10 box - vLLM needs CUDA, do this there, not here):
    pip install vllm openai

    # Single GPU first, to isolate vLLM's own effect from the second-GPU
    # doubling you already get from running two independent servers (see
    # run_qwen_vl.py's CUDA_VISIBLE_DEVICES notes) - add the second GPU as
    # a follow-up once this number looks good on its own.
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \\
        --port 8000 --dtype bfloat16 --limit-mm-per-prompt image=1 \\
        --allowed-local-media-path /path/to/your/dossier/images

Then, in a separate terminal on the same box:

    # Benchmark mode (--output-dir omitted): sweep concurrency levels over
    # a sample, print a pages/sec + latency comparison table, write nothing.
    python scripts/ocr/benchmark_vllm.py --input-dir /path/to/dossier/images \\
        --sample 200 --seed 0 --concurrency 1 4 8 16 32

    # Production mode (--output-dir given, exactly one --concurrency value):
    # writes output files exactly like run_qwen_vl.py does (same
    # --mode/--output-dir mirroring/skip-if-up-to-date behavior), just
    # through the vLLM server instead of a local transformers.generate()
    # loop.
    python scripts/ocr/benchmark_vllm.py --input-dir data/images \\
        --output-dir data/ocr_qwen_vllm --concurrency 32 --mode both

Using both A10s: don't put both GPUs in one `vllm serve` process unless
you also add --tensor-parallel-size 2 - and even then, for a model that
already fits on one GPU (this one does), that just adds cross-GPU
communication overhead for no benefit. Better: two independent servers,
one per GPU (zero cross-GPU communication, throughput roughly doubles),
each fed its own shard of the image list via --shard:

    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000 ...
    CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8001 ...

    python scripts/ocr/benchmark_vllm.py --input-dir data/images \\
        --output-dir data/ocr_qwen_vllm --concurrency 32 --mode both \\
        --base-url http://localhost:8000/v1 --shard 0/2 &
    python scripts/ocr/benchmark_vllm.py --input-dir data/images \\
        --output-dir data/ocr_qwen_vllm --concurrency 32 --mode both \\
        --base-url http://localhost:8001/v1 --shard 1/2 &
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

from io_utils import collect_images, is_up_to_date, output_path
from run_qwen_vl import BBOX_PROMPT, MARKDOWN_PROMPT, OUTPUT_SUFFIX

PROMPTS = {"markdown": MARKDOWN_PROMPT, "bbox": BBOX_PROMPT}

# Constrains bbox-mode generation to guaranteed-valid JSON matching this
# shape (vLLM's structured-outputs decoding masks out any token that would
# violate it, rather than just prompting the model to comply) - see
# --guided-json. Doesn't itself guarantee *completeness* on the densest
# pages (a max_tokens cutoff can still truncate mid-array), only syntactic
# validity.
#
# Root must be an "object", not a bare "array" - both OpenAI's and vLLM's
# json_schema response_format require this (a bare-array root schema is
# rejected/unsupported), so the array is wrapped under "detections" here.
# This changes bbox mode's on-disk output shape from a plain [...] array to
# {"detections": [...]}  - update any downstream JSON consumers (including
# your own validity-checking script) accordingly when --guided-json is on.
BBOX_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "bbox_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                },
                "required": ["text", "bbox_2d"],
            },
        },
    },
    "required": ["detections"],
}


async def ocr_one(
    client: AsyncOpenAI, semaphore: asyncio.Semaphore, model: str, image_path: Path, prompt: str, gen_kwargs: dict,
) -> tuple[str | None, float, str | None]:
    """Returns (text, elapsed, error) - error is None on success. gen_kwargs:
    see build_gen_kwargs - max_tokens, frequency_penalty (standard OpenAI
    field), and repetition_penalty (vLLM extension, via extra_body)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"file://{image_path.resolve()}"}},
            ],
        }
    ]
    async with semaphore:
        t0 = time.time()
        try:
            response = await client.chat.completions.create(
                model=model, messages=messages, temperature=0, **gen_kwargs,
            )
            elapsed = time.time() - t0
            return response.choices[0].message.content, elapsed, None
        except Exception as e:
            return None, time.time() - t0, repr(e)


def build_gen_kwargs(
    max_tokens: int, frequency_penalty: float, repetition_penalty: float, mode: str, guided_json: bool,
) -> dict:
    """frequency_penalty and repetition_penalty are two different (and
    combinable) mechanisms for discouraging the repeat-the-same-chunk
    failure mode a small model under greedy decoding is prone to on
    hard/ambiguous pages - see run_qwen_vl.py's --no-repeat-ngram-size
    docstring for the underlying cause (this is the vLLM/OpenAI-API
    equivalent mitigation, not available as an exact no_repeat_ngram_size
    match through this API). frequency_penalty (standard OpenAI field,
    logit-additive, scales with how often a token already appeared) is
    always sent; repetition_penalty (vLLM extension via extra_body,
    logit-multiplicative, binary already-appeared-or-not) is only added if
    non-default (1.0 = off), so a plain request still works against a
    vLLM version that doesn't support it.

    guided_json (bbox mode only): constrains generation so it's
    *guaranteed* syntactically valid JSON matching BBOX_JSON_SCHEMA - fixes
    genuine malformed JSON (bad nesting, dropped commas, missing required
    keys, stray prose/markdown fences around it), a different problem from
    truncation-driven incompleteness on the densest pages, which this
    doesn't fix (that needs more --max-tokens headroom instead - see that
    flag's help).

    Uses the standard OpenAI response_format={"type": "json_schema", ...}
    field, NOT extra_body={"guided_json": ...} - the guided_* parameters
    (and the --guided-decoding-backend server flag) were removed in vLLM
    0.12.0 in favor of this unified "structured outputs" mechanism. The
    default server-side backend is "auto" (picks an appropriate backend
    per-request), so no server flag should be needed - if requests error,
    check `vllm serve --help | grep structured-outputs` on your version and
    add e.g. --structured-outputs-config.backend xgrammar explicitly.

    Because a json_schema response_format's root must be an "object" (a
    bare top-level "array" schema isn't supported), BBOX_JSON_SCHEMA wraps
    the detections array under a "detections" key - bbox mode's on-disk
    output is {"detections": [...]}  when this is on, not a plain [...]
    array; update downstream consumers accordingly."""
    gen_kwargs = {"max_tokens": max_tokens, "frequency_penalty": frequency_penalty}
    if repetition_penalty != 1.0:
        gen_kwargs["extra_body"] = {"repetition_penalty": repetition_penalty}
    if guided_json and mode == "bbox":
        gen_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "bbox_detections", "schema": BBOX_JSON_SCHEMA},
        }
    return gen_kwargs


async def run_batch(
    client: AsyncOpenAI, model: str, jobs: list[tuple[Path, str, dict]], concurrency: int,
) -> list[tuple[str | None, float, str | None]]:
    """jobs: [(image_path, prompt, gen_kwargs)] - gen_kwargs varies per
    mode (see build_gen_kwargs), hence bundled per-job rather than shared.
    Runs all of them with at most `concurrency` in flight at once - this
    concurrency, not the request count, is what lets vLLM's continuous
    batching actually kick in."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [ocr_one(client, semaphore, model, path, prompt, gk) for path, prompt, gk in jobs]
    return await asyncio.gather(*tasks)


def summarize(results: list[tuple[str | None, float, str | None]], wall_time: float) -> dict:
    latencies = [elapsed for _, elapsed, error in results if error is None]
    n_failed = sum(1 for _, _, error in results if error is not None)
    return {
        "n": len(results),
        "n_failed": n_failed,
        "wall_time": wall_time,
        "pages_per_sec": len(results) / wall_time if wall_time > 0 else float("nan"),
        "latency_mean": statistics.mean(latencies) if latencies else float("nan"),
        "latency_median": statistics.median(latencies) if latencies else float("nan"),
        "latency_p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=float("nan")),
    }


async def benchmark(client: AsyncOpenAI, model: str, images: list[Path], modes: list[str],
                     concurrency_levels: list[int], gen_kwargs_by_mode: dict[str, dict]) -> None:
    jobs = [(path, PROMPTS[mode], gen_kwargs_by_mode[mode]) for path in images for mode in modes]
    print(f"{len(images)} image(s) x {len(modes)} mode(s) = {len(jobs)} request(s) per concurrency level\n")

    header = f"{'concurrency':>11} | {'pages/sec':>10} | {'wall_time':>10} | {'latency mean/median/p95 (s)':>28} | failed"
    print(header)
    print("-" * len(header))
    for concurrency in concurrency_levels:
        t0 = time.time()
        results = await run_batch(client, model, jobs, concurrency)
        wall_time = time.time() - t0
        s = summarize(results, wall_time)
        print(
            f"{concurrency:>11} | {s['pages_per_sec']:>10.2f} | {s['wall_time']:>9.1f}s | "
            f"{s['latency_mean']:>7.1f} / {s['latency_median']:>7.1f} / {s['latency_p95']:>7.1f}   | {s['n_failed']}"
        )
    print(
        "\nFor reference, the earlier one-page-at-a-time transformers.generate() run measured "
        "10-35s/page (markdown) and 18-65s/page (bbox) on a single A10 - compare pages/sec above "
        "against 1/35 ~= 0.03 and 1/65 ~= 0.015 pages/sec as the naive baseline."
    )


async def production_run(client: AsyncOpenAI, model: str, pairs: list[tuple[Path, Path]], modes: list[str],
                          output_dir: Path, concurrency: int, gen_kwargs_by_mode: dict[str, dict], force: bool) -> None:
    jobs = []  # (src, relative_path, mode, dst)
    for src, relative_path in pairs:
        for mode in modes:
            dst = output_path(output_dir, relative_path, OUTPUT_SUFFIX[mode])
            if not force and is_up_to_date(src, dst):
                continue
            jobs.append((src, relative_path, mode, dst))

    n_skipped = len(pairs) * len(modes) - len(jobs)
    print(f"{len(jobs)} output(s) to generate, {n_skipped} already up to date (skipped)")
    if not jobs:
        return

    semaphore = asyncio.Semaphore(concurrency)

    async def process(src, relative_path, mode, dst):
        text, elapsed, error = await ocr_one(client, semaphore, model, src, PROMPTS[mode], gen_kwargs_by_mode[mode])
        if error:
            print(f"[FAILED] [{mode}] {src}: {error}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print(f"[{elapsed:.1f}s] [{mode}] {src} -> {dst}")

    t0 = time.time()
    await asyncio.gather(*(process(*job) for job in jobs))
    wall_time = time.time() - t0
    print(f"\n{len(jobs)} output(s) generated in {wall_time:.1f}s ({len(jobs) / wall_time:.2f} req/s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, nargs="+", help="explicit list of image files")
    source.add_argument("--input-dir", type=Path, help="recursively OCR every image under this directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="production mode: write output files (needs exactly one --concurrency value). "
                              "Omit for benchmark mode: sweep --concurrency levels, print stats, write nothing.")
    parser.add_argument("--force", action="store_true", help="production mode: reprocess up-to-date pages too")
    parser.add_argument("--sample", type=int, default=None,
                         help="randomly sample this many images (benchmark mode - a full-corpus sweep would "
                              "defeat the point of a quick benchmark)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["markdown", "bbox", "both"], default="both")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[8],
                         help="one value in production mode, one or more (a sweep) in benchmark mode")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="must match what vllm serve is running")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=4096,
                         help="a dense page's full bbox-JSON enumeration (one text+bbox_2d entry per detected "
                              "block) can need more than the default chat-completion length - if you see "
                              "truncated/invalid JSON on dense pages, raise this further")
    parser.add_argument("--frequency-penalty", type=float, default=0.3,
                         help="discourages the repeat-the-same-chunk failure mode greedy decoding is prone to "
                              "on hard/ambiguous pages, especially with a smaller model - standard OpenAI field, "
                              "0=off, higher=stronger. See build_gen_kwargs.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                         help="a second, complementary repetition mitigation (vLLM-specific, via extra_body) - "
                              "1.0=off. Try ~1.1-1.3 in addition to --frequency-penalty if repetition persists.")
    parser.add_argument("--guided-json", action=argparse.BooleanOptionalAction, default=True,
                         help="bbox mode only: constrain generation to guaranteed-valid JSON via a "
                              "response_format json_schema (see build_gen_kwargs) - fixes genuinely malformed "
                              "JSON, not truncation-driven incompleteness (that's --max-tokens). Changes bbox "
                              "mode's on-disk output to {\"detections\": [...]}  instead of a bare [...] array "
                              "- update downstream consumers. Use --no-guided-json to fall back to unconstrained "
                              "generation if your vLLM version rejects response_format for this model.")
    parser.add_argument("--shard", default=None, metavar="I/N",
                         help="production mode with two (or more) independent vllm servers, one per GPU: run "
                              "this script once per server, each with a different I (0-indexed) and the same N "
                              "(total server count) and --base-url pointed at its own server, e.g. --shard 0/2 "
                              "against port 8000 and --shard 1/2 against port 8001. Splits the image list with a "
                              "strided slice (pairs[I::N]) rather than a contiguous chunk, so a run of similar "
                              "(e.g. same-dossier) pages doesn't all land on one shard.")
    args = parser.parse_args()

    if args.output_dir and len(args.concurrency) != 1:
        raise SystemExit("--output-dir (production mode) needs exactly one --concurrency value, not a sweep")

    modes = ["markdown", "bbox"] if args.mode == "both" else [args.mode]
    client = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url)
    gen_kwargs_by_mode = {
        mode: build_gen_kwargs(args.max_tokens, args.frequency_penalty, args.repetition_penalty, mode, args.guided_json)
        for mode in modes
    }
    pairs = collect_images(args.images, args.input_dir)

    if args.shard:
        shard_index, shard_count = (int(x) for x in args.shard.split("/"))
        pairs = pairs[shard_index::shard_count]
        print(f"shard {shard_index}/{shard_count}: {len(pairs)} image(s) assigned to this run")

    if args.output_dir:
        asyncio.run(production_run(
            client, args.model, pairs, modes, args.output_dir, args.concurrency[0], gen_kwargs_by_mode, args.force
        ))
    else:
        images = [src for src, _ in pairs]
        if args.sample and args.sample < len(images):
            random.Random(args.seed).shuffle(images)
            images = images[: args.sample]
        asyncio.run(benchmark(client, args.model, images, modes, args.concurrency, gen_kwargs_by_mode))


if __name__ == "__main__":
    main()
