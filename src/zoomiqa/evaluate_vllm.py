"""vLLM evaluator for the Zoom-IQA two-round protocol.

This backend keeps the prompts, full first-round history, and crop policy of the
Transformers evaluator, but samples at a lower temperature and stops on
``</answer>``. It is a throughput-oriented variant, so its numbers can differ
slightly from the Transformers backend.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Iterable

from .environment import collect_environment
from .evaluate import (
    EvalSample,
    _categorize_parse_error,
    _display_bbox,
    _load_image,
    _merge_unique_records,
    _read_jsonl,
    _result_record,
    _write_predictions,
    atomic_json,
    load_samples,
    model_reference,
    parse_devices,
    sha256_file,
    summarize,
)
from .protocol import (
    MAX_CROP_AREA_RATIO,
    MAX_ROUNDS,
    MIN_CROP_AREA_RATIO,
    AnswerParseError,
    build_stage1_message,
    build_stage2_messages,
    crop_for_stage2,
    parse_answer,
)


BACKEND_NAME = "vllm"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_MAX_TOKENS = 1024
STOP_STRINGS = ("</answer>",)
VLLM_PACKAGES = (
    "accelerate",
    "compressed-tensors",
    "flashinfer-python",
    "huggingface-hub",
    "msgspec",
    "ninja",
    "safetensors",
    "tokenizers",
    "torch",
    "torchvision",
    "transformers",
    "vllm",
    "qwen-vl-utils",
    "flash-attn",
    "xformers",
    "numpy",
    "pillow",
    "triton",
    "dirtyjson",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in VLLM_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_vllm_environment() -> dict[str, Any]:
    environment = collect_environment()
    environment["inference_backend"] = BACKEND_NAME
    environment["backend_packages"] = _package_versions()
    return environment


def sampling_config(
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> dict[str, Any]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("--temperature must be finite and greater than zero")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if top_k == 0 or top_k < -1:
        raise ValueError("--top-k must be -1 or a positive integer")
    if max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stop": list(STOP_STRINGS),
        "include_stop_str_in_output": True,
    }


def _physical_device(logical_device: int, original_visible: str | None) -> str:
    if original_visible:
        entries = [item.strip() for item in original_visible.split(",") if item.strip()]
        if logical_device >= len(entries):
            raise ValueError(
                f"logical CUDA device {logical_device} is outside CUDA_VISIBLE_DEVICES"
            )
        return entries[logical_device]
    return str(logical_device)


def _vllm_outputs(outputs: Iterable[Any]) -> tuple[list[str], list[int], list[Any]]:
    texts: list[str] = []
    token_lengths: list[int] = []
    finish_reasons: list[Any] = []
    for request in outputs:
        if not request.outputs:
            raise RuntimeError("vLLM returned a request without a completion")
        completion = request.outputs[0]
        texts.append(completion.text)
        token_lengths.append(len(completion.token_ids))
        finish_reasons.append(
            {
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
            }
        )
    return texts, token_lengths, finish_reasons


def _make_inputs(processor: Any, messages: list[list[dict[str, Any]]], images: list[Any]) -> list[dict[str, Any]]:
    prompts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    return [
        {"prompt": prompt, "multi_modal_data": {"image": image}}
        for prompt, image in zip(prompts, images)
    ]


def _evaluate_batch(
    samples: list[EvalSample],
    model: Any,
    processor: Any,
    sampling_params: Any,
) -> list[dict[str, Any]]:
    stage1_messages = [
        build_stage1_message(sample.question, sample.image_path) for sample in samples
    ]
    stage1_images = [_load_image(sample.image_path) for sample in samples]
    stage1_raw = model.generate(
        _make_inputs(processor, stage1_messages, stage1_images),
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    stage1_outputs, stage1_lengths, stage1_finishes = _vllm_outputs(stage1_raw)

    results: list[dict[str, Any] | None] = [None] * len(samples)
    stage2_indices: list[int] = []
    stage2_messages: list[list[dict[str, Any]]] = []
    stage2_images: list[list[Any]] = []
    pixel_bboxes: dict[int, list[int]] = {}

    for local_index, (sample, output) in enumerate(zip(samples, stage1_outputs)):
        try:
            parsed = parse_answer(output)
        except AnswerParseError as error:
            record = _result_record(
                sample=sample,
                output=output,
                history=[],
                rounds=1,
                generated_tokens=[stage1_lengths[local_index]],
                error=None,
                bbox=None,
                routing_warning=_categorize_parse_error(error),
            )
            record["finish_reasons"] = [stage1_finishes[local_index]]
            results[local_index] = record
            continue
        if not parsed.requests_crop:
            record = _result_record(
                sample=sample,
                output=output,
                history=[],
                rounds=1,
                generated_tokens=[stage1_lengths[local_index]],
                error=None,
                bbox=_display_bbox(parsed.bbox),
            )
            record["finish_reasons"] = [stage1_finishes[local_index]]
            results[local_index] = record
            continue

        original = stage1_images[local_index]
        cropped, pixel_bbox = crop_for_stage2(original, parsed.bbox)
        stage2_indices.append(local_index)
        stage2_messages.append(
            build_stage2_messages(sample.history_question, output)
        )
        stage2_images.append([original, cropped])
        pixel_bboxes[local_index] = pixel_bbox

    if stage2_indices:
        stage2_raw = model.generate(
            _make_inputs(processor, stage2_messages, stage2_images),
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        stage2_outputs, stage2_lengths, stage2_finishes = _vllm_outputs(stage2_raw)
        for position, local_index in enumerate(stage2_indices):
            sample = samples[local_index]
            record = _result_record(
                sample=sample,
                output=stage2_outputs[position],
                history=[stage1_outputs[local_index]],
                rounds=2,
                generated_tokens=[stage1_lengths[local_index], stage2_lengths[position]],
                error=None,
                bbox=pixel_bboxes[local_index],
            )
            record["finish_reasons"] = [
                stage1_finishes[local_index],
                stage2_finishes[position],
            ]
            results[local_index] = record

    if any(result is None for result in results):
        raise RuntimeError("internal error: a batch result was not finalized")
    return [record for record in results if record is not None]


def _worker(
    worker_id: int,
    logical_device: int,
    original_visible: str | None,
    model_path: str,
    revision: str | None,
    samples: list[EvalSample],
    batch_size: int,
    seed: int,
    generation: dict[str, Any],
    gpu_memory_utilization: float,
    max_model_len: int,
    shard_path: str,
) -> None:
    physical_device = _physical_device(logical_device, original_visible)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    environment_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    worker_seed = seed + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision=revision,
        use_fast=False,
    )
    processor.tokenizer.padding_side = "left"
    model = LLM(
        model=model_path,
        revision=revision,
        tensor_parallel_size=1,
        trust_remote_code=True,
        seed=worker_seed,
        enable_prefix_caching=True,
        disable_cascade_attn=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=batch_size,
        max_model_len=max_model_len,
    )
    eos_token_id = processor.tokenizer.eos_token_id
    params_payload = dict(generation)
    if eos_token_id is not None:
        params_payload["stop_token_ids"] = [eos_token_id]
    params = SamplingParams(**params_payload)

    path = Path(shard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            try:
                records = _evaluate_batch(batch, model, processor, params)
            except Exception as batch_error:
                records = [
                    _result_record(
                        sample=sample,
                        output="",
                        history=[],
                        rounds=0,
                        generated_tokens=[],
                        error=(
                            f"runtime_error:{type(batch_error).__name__}:"
                            f"{batch_error}"
                        ),
                        bbox=None,
                    )
                    for sample in batch
                ]
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--annotation", required=True, nargs="+", type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--devices", default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--worker-timeout", type=int, default=14400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    if args.max_model_len < args.max_tokens + 1:
        raise SystemExit("--max-model-len must be greater than --max-tokens")
    try:
        generation = sampling_config(
            args.temperature, args.top_p, args.top_k, args.max_tokens
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    environment = collect_vllm_environment()
    if not environment.get("cuda_available"):
        raise SystemExit("CUDA is unavailable")
    visible_count = int(environment.get("gpu_count", 0))
    try:
        devices = parse_devices(args.devices, visible_count)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    samples, annotation_meta = load_samples(
        args.annotation, args.image_root, args.seed, args.max_samples
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    config_path = args.output_dir / "run_config.json"
    environment_path = args.output_dir / "environment.json"
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    config_payload = {
        "inference_backend": BACKEND_NAME,
        "model_path": args.model_path,
        "model_revision": args.revision,
        "model_reference": model_reference(args.model_path, args.revision),
        "annotations": annotation_meta,
        "image_root": str(args.image_root.resolve()),
        "generation": generation,
        "max_rounds": MAX_ROUNDS,
        "crop_area_ratio": [MIN_CROP_AREA_RATIO, MAX_CROP_AREA_RATIO],
        "batch_size_per_gpu": args.batch_size,
        "seed": args.seed,
        "logical_devices": devices,
        "cuda_visible_devices_at_launch": original_visible,
        "worker_cuda_visible_devices": [
            _physical_device(device, original_visible) for device in devices
        ],
        "worker_seeds": [args.seed + index for index in range(len(devices))],
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "image_processor_use_fast": False,
        "backend_packages": environment.get("backend_packages"),
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config_payload["config_sha256"] = config_hash
    config_payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()

    existing: list[dict[str, Any]] = []
    shard_dir = args.output_dir / ".shards"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("config_sha256") != config_hash:
            raise SystemExit("output directory belongs to a different run configuration")
        if not args.resume:
            raise SystemExit("output exists; pass --resume or choose another --output-dir")
        groups = [_read_jsonl(predictions_path)]
        groups.extend(
            _read_jsonl(path, tolerate_truncated_final_line=True)
            for path in sorted(shard_dir.glob("worker-*.jsonl"))
        )
        existing = _merge_unique_records(groups, len(samples))
        if existing:
            _write_predictions(predictions_path, existing)
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
    elif predictions_path.exists():
        raise SystemExit(
            "predictions exist without run_config.json; choose another output directory"
        )
    else:
        atomic_json(config_path, config_payload)
        atomic_json(environment_path, environment)

    completed = {int(record["index"]) for record in existing}
    remaining = [sample for sample in samples if sample.index not in completed]
    if not remaining:
        summary = summarize(existing)
        summary["inference_backend"] = BACKEND_NAME
        summary["generation"] = generation
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    chunk_size = math.ceil(len(samples) / len(devices))
    chunks = [
        [
            sample
            for sample in samples[position * chunk_size : (position + 1) * chunk_size]
            if sample.index not in completed
        ]
        for position in range(len(devices))
    ]
    shard_dir.mkdir(parents=True)
    context = multiprocessing.get_context("spawn")
    processes = []
    for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
        if not chunk:
            continue
        process = context.Process(
            target=_worker,
            args=(
                worker_id,
                device,
                original_visible,
                args.model_path,
                args.revision,
                chunk,
                args.batch_size,
                args.seed,
                generation,
                args.gpu_memory_utilization,
                args.max_model_len,
                str(shard_dir / f"worker-{worker_id}.jsonl"),
            ),
        )
        process.start()
        processes.append(process)

    deadline = time.monotonic() + args.worker_timeout
    failure: str | None = None
    try:
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                failure = f"worker {process.pid} exceeded --worker-timeout"
                break
            if process.exitcode != 0:
                failure = f"worker {process.pid} failed with exit code {process.exitcode}"
                break
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        raise
    if failure is not None:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        raise SystemExit(failure)

    new_records: list[dict[str, Any]] = []
    for path in sorted(shard_dir.glob("worker-*.jsonl")):
        new_records.extend(_read_jsonl(path))
    all_records = _merge_unique_records([existing, new_records], len(samples))
    if len(all_records) != len(samples):
        raise SystemExit(
            f"incomplete run: produced {len(all_records)} of {len(samples)} samples"
        )
    _write_predictions(predictions_path, all_records)
    summary = summarize(all_records)
    summary["inference_backend"] = BACKEND_NAME
    summary["generation"] = generation
    atomic_json(args.output_dir / "summary.json", summary)
    shutil.rmtree(shard_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
