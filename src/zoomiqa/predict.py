"""Score one user-provided image with the Zoom-IQA two-round protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from .evaluate import EvalSample, _evaluate_batch, parse_devices
from .protocol import GENERATION_CONFIG, RANDOM_QUESTIONS


def _resolve_image(image: str | Path) -> Path:
    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"image not found: {path}")

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as opened:
            opened.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"cannot decode image: {path}") from error
    return path


def _prepare_question(
    question: str | None, rng: random.Random
) -> tuple[str, str]:
    history_question = rng.choice(RANDOM_QUESTIONS) if question is None else question
    if not isinstance(history_question, str) or not history_question.strip():
        raise ValueError("question must contain non-whitespace text")
    stage1_question = history_question.replace("<|image|>", "")
    if not stage1_question.strip():
        raise ValueError("question must contain text other than <|image|>")
    return stage1_question, history_question


def _single_image_sample(
    image: str | Path,
    question: str | None,
    rng: random.Random,
) -> EvalSample:
    path = _resolve_image(image)
    stage1_question, history_question = _prepare_question(question, rng)
    return EvalSample(
        index=0,
        dataset="single_image",
        source_index=0,
        sample_id=path.stem,
        image=str(path),
        image_path=str(path),
        question=stage1_question,
        history_question=history_question,
        ground_truth=0.0,
    )


def _public_prediction(record: dict[str, Any]) -> dict[str, Any]:
    valid = bool(record["valid"])
    history = record.get("output_history") or []
    return {
        "image": record["image"],
        "question": record["question"],
        "rating": float(record["rating"]) if valid else None,
        "score_0_100": float(record["model_score"]) if valid else None,
        "valid": valid,
        "error": record.get("error"),
        "routing_warning": record.get("routing_warning"),
        "rounds": int(record["rounds"]),
        "crop_bbox_pixels": record.get("crop_bbox_pixels"),
        "generated_tokens": record.get("generated_tokens", []),
        "response": record.get("model_output", ""),
        "first_round_response": history[0] if history else None,
    }


class ZoomIQAPredictor:
    """Load Zoom-IQA once and score local images one at a time."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        device: int = 0,
        seed: int = 42,
        attention_implementation: str = "flash_attention_2",
    ) -> None:
        if attention_implementation not in {"flash_attention_2", "sdpa", "eager"}:
            raise ValueError(
                "attention_implementation must be flash_attention_2, sdpa, or eager"
            )
        import torch
        from transformers import AutoProcessor, GenerationConfig
        from transformers import Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.device = parse_devices(str(device), torch.cuda.device_count())[0]
        torch.cuda.set_device(self.device)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.processor = AutoProcessor.from_pretrained(model_path, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            revision=revision,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
        ).to(f"cuda:{self.device}")
        self.model.eval()
        self.generation_config = GenerationConfig(**GENERATION_CONFIG)
        self._question_rng = random.Random(seed)

    def predict(
        self, image: str | Path, *, question: str | None = None
    ) -> dict[str, Any]:
        """Return a structured prediction for one local image path."""
        sample = _single_image_sample(image, question, self._question_rng)
        record = _evaluate_batch(
            [sample],
            self.model,
            self.processor,
            self.tokenizer,
            self.generation_config,
            self.device,
        )[0]
        return _public_prediction(record)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model ID or local checkpoint")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--image", required=True, type=Path, help="local image path")
    parser.add_argument("--question", default=None, help="optional custom IQA question")
    parser.add_argument("--device", type=int, default=0, help="logical CUDA device index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        image = _resolve_image(args.image)
        predictor = ZoomIQAPredictor(
            args.model_path,
            revision=args.revision,
            device=args.device,
            seed=args.seed,
            attention_implementation=args.attention_implementation,
        )
        prediction = predictor.predict(image, question=args.question)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    rendered = json.dumps(prediction, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if prediction["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
