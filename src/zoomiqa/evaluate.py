"""Multi-GPU benchmark evaluator for the Zoom-IQA two-round protocol."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as multiprocessing_stdlib
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Iterable

from .environment import collect_environment
from .metrics import correlations
from .protocol import (
    GENERATION_CONFIG,
    MAX_CROP_AREA_RATIO,
    MAX_ROUNDS,
    MIN_CROP_AREA_RATIO,
    RANDOM_QUESTIONS,
    AnswerParseError,
    build_stage1_message,
    build_stage2_messages,
    crop_for_stage2,
    extract_rating,
    parse_answer,
    rating_to_score,
)


@dataclass(frozen=True)
class EvalSample:
    index: int
    dataset: str
    source_index: int
    sample_id: str
    image: str
    image_path: str
    question: str
    history_question: str
    ground_truth: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def model_reference(model_path: str, revision: str | None) -> dict[str, Any]:
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return {"kind": "huggingface_hub", "id": model_path, "revision": revision}
    return {
        "kind": "local_directory",
        "path": str(path.resolve()),
        "revision": revision,
    }


def _questions_from_row(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        first = conversations[0]
        if isinstance(first, dict):
            question = first.get("value") or first.get("content")
            if isinstance(question, str) and question.strip():
                return question.replace("<|image|>", ""), question
    question = rng.choice(RANDOM_QUESTIONS)
    return question, question


def load_samples(
    annotation_paths: Iterable[Path],
    image_root: Path,
    seed: int,
    max_samples: int | None,
) -> tuple[list[EvalSample], list[dict[str, Any]]]:
    samples: list[EvalSample] = []
    annotation_meta = []
    global_index = 0
    seen_datasets: set[str] = set()
    resolved_image_root = image_root.resolve()
    for annotation_path in annotation_paths:
        with annotation_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"{annotation_path}: expected a JSON list")
        dataset = annotation_path.stem
        if dataset in seen_datasets:
            raise ValueError(f"duplicate annotation stem: {dataset}")
        seen_datasets.add(dataset)
        rng = random.Random(seed)  # Each dataset draws its questions independently.
        selected_rows = rows if max_samples is None else rows[:max_samples]
        for source_index, row in enumerate(selected_rows):
            if not isinstance(row, dict):
                raise ValueError(f"{annotation_path}: row {source_index} is not an object")
            relative_image = row.get("image") or row.get("img_path")
            if not isinstance(relative_image, str) or not relative_image:
                raise ValueError(f"{annotation_path}: row {source_index} has no image path")
            relative_path = Path(relative_image)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"{annotation_path}: row {source_index} image must be a safe relative path"
                )
            image_path = (resolved_image_root / relative_path).resolve()
            if not image_path.is_relative_to(resolved_image_root):
                raise ValueError(
                    f"{annotation_path}: row {source_index} image escapes --image-root"
                )
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{annotation_path}: row {source_index} image not found: {image_path}"
                )
            try:
                ground_truth = float(row["gt_score"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{annotation_path}: row {source_index} has invalid gt_score"
                ) from error
            if not math.isfinite(ground_truth):
                raise ValueError(
                    f"{annotation_path}: row {source_index} gt_score is not finite"
                )
            sample_id = str(row.get("id", f"{dataset}:{source_index}"))
            question, history_question = _questions_from_row(row, rng)
            samples.append(
                EvalSample(
                    index=global_index,
                    dataset=dataset,
                    source_index=source_index,
                    sample_id=sample_id,
                    image=relative_image,
                    image_path=str(image_path),
                    question=question,
                    history_question=history_question,
                    ground_truth=ground_truth,
                )
            )
            global_index += 1
        annotation_meta.append(
            {
                "dataset": dataset,
                "path": str(annotation_path.resolve()),
                "sha256": sha256_file(annotation_path),
                "source_count": len(rows),
                "selected_count": len(selected_rows),
            }
        )
    if not samples:
        raise ValueError("no evaluation samples were loaded")
    return samples, annotation_meta


def _categorize_parse_error(error: AnswerParseError) -> str:
    message = str(error)
    if message.startswith("missing <answer>"):
        return "missing_answer"
    if "JSON" in message:
        return "malformed_json"
    if "rating" in message:
        return "invalid_rating"
    if "bbox" in message:
        return "invalid_bbox"
    if "tool" in message:
        return "invalid_tool"
    return "parse_error"


def _load_image(path: str) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.copy()


def _display_bbox(value: Any) -> list[float] | None:
    try:
        converted = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError):
        return None
    return converted if len(converted) == 4 else None


def _move_inputs(inputs: Any, device: int) -> Any:
    return inputs.to(f"cuda:{device}")


def _completion_token_lengths(
    completion_ids: Iterable[Any], stop_token_ids: Iterable[int]
) -> list[int]:
    stops = {int(token_id) for token_id in stop_token_ids}
    lengths = []
    for sequence in completion_ids:
        values = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
        stop_position = next(
            (position for position, token_id in enumerate(values) if int(token_id) in stops),
            None,
        )
        lengths.append(len(values) if stop_position is None else stop_position + 1)
    return lengths


def _decode_batch(model: Any, processor: Any, inputs: Any, generation_config: Any) -> tuple[list[str], list[int]]:
    generated = model.generate(**inputs, generation_config=generation_config)
    input_width = inputs["input_ids"].shape[1]
    completion_ids = [output[input_width:] for output in generated]
    texts = processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    stop_token_ids: set[int] = set()
    for config in (generation_config, model.generation_config):
        for attribute in ("eos_token_id", "pad_token_id"):
            value = getattr(config, attribute, None)
            if isinstance(value, (list, tuple, set)):
                stop_token_ids.update(int(token_id) for token_id in value)
            elif value is not None:
                stop_token_ids.add(int(value))
    lengths = _completion_token_lengths(completion_ids, stop_token_ids)
    return texts, lengths


def _evaluate_batch(
    samples: list[EvalSample],
    model: Any,
    processor: Any,
    tokenizer: Any,
    generation_config: Any,
    device: int,
) -> list[dict[str, Any]]:
    import torch
    from qwen_vl_utils import process_vision_info

    stage1_messages = [build_stage1_message(sample.question, sample.image_path) for sample in samples]
    stage1_text = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in stage1_messages
    ]
    image_inputs, video_inputs = process_vision_info(stage1_messages)
    stage1_inputs = processor(
        text=stage1_text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    stage1_inputs = _move_inputs(stage1_inputs, device)
    with torch.inference_mode():
        stage1_outputs, stage1_lengths = _decode_batch(
            model, processor, stage1_inputs, generation_config
        )

    results: list[dict[str, Any] | None] = [None] * len(samples)
    stage2_indices: list[int] = []
    stage2_images: list[list[Any]] = []
    stage2_text: list[str] = []
    pixel_bboxes: dict[int, list[int]] = {}

    for local_index, (sample, output) in enumerate(zip(samples, stage1_outputs)):
        try:
            parsed = parse_answer(output)
        except AnswerParseError as error:
            results[local_index] = _result_record(
                sample=sample,
                output=output,
                history=[],
                rounds=1,
                generated_tokens=[stage1_lengths[local_index]],
                error=None,
                bbox=None,
                routing_warning=_categorize_parse_error(error),
            )
            continue
        if not parsed.requests_crop:
            results[local_index] = _result_record(
                sample=sample,
                output=output,
                history=[],
                rounds=1,
                generated_tokens=[stage1_lengths[local_index]],
                error=None,
                bbox=_display_bbox(parsed.bbox),
            )
            continue

        original_image = _load_image(sample.image_path)
        cropped_image, pixel_bbox = crop_for_stage2(original_image, parsed.bbox)
        messages = build_stage2_messages(sample.history_question, output)
        stage2_indices.append(local_index)
        stage2_images.append([original_image, cropped_image])
        stage2_text.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        pixel_bboxes[local_index] = pixel_bbox

    if stage2_indices:
        stage2_inputs = processor(
            text=stage2_text,
            images=stage2_images,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        stage2_inputs = _move_inputs(stage2_inputs, device)
        with torch.inference_mode():
            stage2_outputs, stage2_lengths = _decode_batch(
                model, processor, stage2_inputs, generation_config
            )
        for stage2_position, local_index in enumerate(stage2_indices):
            sample = samples[local_index]
            output = stage2_outputs[stage2_position]
            results[local_index] = _result_record(
                sample=sample,
                output=output,
                history=[stage1_outputs[local_index]],
                rounds=2,
                generated_tokens=[stage1_lengths[local_index], stage2_lengths[stage2_position]],
                error=None,
                bbox=pixel_bboxes[local_index],
            )

    if any(result is None for result in results):
        raise RuntimeError("internal error: a batch result was not finalized")
    return [result for result in results if result is not None]


def _result_record(
    sample: EvalSample,
    output: str,
    history: list[str],
    rounds: int,
    generated_tokens: list[int],
    error: str | None,
    bbox: list[float] | list[int] | None,
    routing_warning: str | None = None,
) -> dict[str, Any]:
    rating: float | None = None
    if error is None:
        try:
            rating = min(5.0, max(1.0, extract_rating(output)))
        except AnswerParseError as parse_error:
            error = _categorize_parse_error(parse_error)
    valid = error is None and rating is not None
    return {
        "index": sample.index,
        "dataset": sample.dataset,
        "source_index": sample.source_index,
        "id": sample.sample_id,
        "image": sample.image,
        "question": sample.question,
        "history_question": sample.history_question,
        "ground_truth": sample.ground_truth,
        "rating": rating if valid else -100,
        "model_score": rating_to_score(rating) if valid else -100,
        "valid": valid,
        "error": error,
        "routing_warning": routing_warning,
        "rounds": rounds,
        "crop_bbox_pixels": bbox,
        "generated_tokens": generated_tokens,
        "model_output": output,
        "output_history": history,
    }


def _worker(
    worker_id: int,
    device: int,
    model_path: str,
    revision: str | None,
    samples: list[EvalSample],
    batch_size: int,
    seed: int,
    attention_implementation: str,
    shard_path: str,
) -> None:
    import torch
    from transformers import AutoProcessor, GenerationConfig
    from transformers import Qwen2_5_VLForConditionalGeneration

    torch.cuda.set_device(device)
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

    processor = AutoProcessor.from_pretrained(model_path, revision=revision)
    tokenizer = processor.tokenizer
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_implementation,
    ).to(f"cuda:{device}")
    model.eval()
    generation_config = GenerationConfig(**GENERATION_CONFIG)

    path = Path(shard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            try:
                records = _evaluate_batch(
                    batch, model, processor, tokenizer, generation_config, device
                )
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


def _read_jsonl(
    path: Path, tolerate_truncated_final_line: bool = False
) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                if (
                    tolerate_truncated_final_line
                    and line_number == len(lines)
                    and not line.endswith("\n")
                ):
                    break
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
    return records


def _write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _merge_unique_records(
    groups: Iterable[Iterable[dict[str, Any]]], sample_count: int
) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for records in groups:
        for record in records:
            try:
                index = int(record["index"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("prediction record has no integer index") from error
            if index < 0 or index >= sample_count:
                raise ValueError(f"prediction index {index} is outside this run")
            previous = by_index.get(index)
            if previous is not None and previous != record:
                raise ValueError(f"conflicting prediction records for index {index}")
            by_index[index] = record
    return [by_index[index] for index in sorted(by_index)]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in sorted({record["dataset"] for record in records}):
        subset = [record for record in records if record["dataset"] == dataset]
        valid = [record for record in subset if record.get("valid")]
        metrics: dict[str, float | None] | None = None
        if len(valid) >= 2:
            metrics = correlations(
                (record["model_score"] for record in valid),
                (record["ground_truth"] for record in valid),
            )
        errors = Counter(record["error"] for record in subset if record.get("error"))
        routing_warnings = Counter(
            record["routing_warning"]
            for record in subset
            if record.get("routing_warning")
        )
        datasets[dataset] = {
            "total": len(subset),
            "valid": len(valid),
            "invalid": len(subset) - len(valid),
            "format_valid_rate": len(valid) / len(subset) if subset else 0.0,
            "average_extra_rounds": (
                sum(max(0, int(record.get("rounds", 0)) - 1) for record in subset)
                / len(subset)
                if subset
                else 0.0
            ),
            "second_round_samples": sum(
                int(record.get("rounds", 0)) == 2 for record in subset
            ),
            "average_final_output_characters": (
                sum(len(str(record.get("model_output", ""))) for record in subset)
                / len(subset)
                if subset
                else 0.0
            ),
            "max_new_token_hits": sum(
                any(
                    int(length) >= int(GENERATION_CONFIG["max_new_tokens"])
                    for length in record.get("generated_tokens", [])
                )
                for record in subset
            ),
            "errors": dict(sorted(errors.items())),
            "routing_warnings": dict(sorted(routing_warnings.items())),
            "metrics": metrics,
        }
    return {
        "total": len(records),
        "valid": sum(bool(record.get("valid")) for record in records),
        "invalid": sum(not bool(record.get("valid")) for record in records),
        "datasets": datasets,
    }


def parse_devices(value: str, visible_count: int) -> list[int]:
    if value == "all":
        devices = list(range(visible_count))
        if not devices:
            raise ValueError("no visible CUDA devices")
        return devices
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--devices must be 'all' or comma-separated integers") from error
    if not devices:
        raise ValueError("no devices selected")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices contains duplicates")
    if any(device < 0 or device >= visible_count for device in devices):
        raise ValueError(f"selected device is outside visible range 0..{visible_count - 1}")
    return devices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model ID or local checkpoint")
    parser.add_argument(
        "--revision",
        default=None,
        help="HF model revision; strongly recommended when --model-path is a Hub ID",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        nargs="+",
        type=Path,
        help="One or more local benchmark annotation JSON files",
    )
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--devices", default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--worker-timeout", type=int, default=14400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--attention-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")

    environment = collect_environment()
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

    config_payload = {
        "inference_backend": "transformers",
        "model_path": args.model_path,
        "model_revision": args.revision,
        "model_reference": model_reference(args.model_path, args.revision),
        "annotations": annotation_meta,
        "image_root": str(args.image_root.resolve()),
        "generation": GENERATION_CONFIG,
        "max_rounds": MAX_ROUNDS,
        "crop_area_ratio": [MIN_CROP_AREA_RATIO, MAX_CROP_AREA_RATIO],
        "batch_size_per_gpu": args.batch_size,
        "seed": args.seed,
        "attention_implementation": args.attention_implementation,
        "devices": devices,
        "worker_seeds": [args.seed + worker_id for worker_id in range(len(devices))],
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
        raise SystemExit("predictions exist without run_config.json; choose another output directory")
    else:
        atomic_json(config_path, config_payload)
        atomic_json(environment_path, environment)

    completed_indices = {int(record["index"]) for record in existing}
    remaining = [sample for sample in samples if sample.index not in completed_indices]
    if not remaining:
        summary = summarize(sorted(existing, key=lambda record: int(record["index"])))
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    # Historical multi-GPU evaluation used contiguous, ceil-sized partitions.
    chunk_size = math.ceil(len(samples) / len(devices))
    chunks = [
        [
            sample
            for sample in samples[position * chunk_size : (position + 1) * chunk_size]
            if sample.index not in completed_indices
        ]
        for position in range(len(devices))
    ]
    shard_dir.mkdir(parents=True)
    context = multiprocessing_stdlib.get_context("spawn")
    processes = []
    for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
        if not chunk:
            continue
        shard_path = shard_dir / f"worker-{worker_id}.jsonl"
        process = context.Process(
            target=_worker,
            args=(
                worker_id,
                device,
                args.model_path,
                args.revision,
                chunk,
                args.batch_size,
                args.seed,
                args.attention_implementation,
                str(shard_path),
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

    new_records = []
    for shard_path in sorted(shard_dir.glob("worker-*.jsonl")):
        new_records.extend(_read_jsonl(shard_path))
    all_records = _merge_unique_records([existing, new_records], len(samples))
    if len(all_records) != len(samples):
        raise SystemExit(f"incomplete run: produced {len(all_records)} of {len(samples)} samples")
    _write_predictions(predictions_path, all_records)
    summary = summarize(all_records)
    atomic_json(args.output_dir / "summary.json", summary)
    shutil.rmtree(shard_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
