"""Prompts, answer parsing, and crop policy for the Zoom-IQA two-round protocol.

The second round receives the complete first-round answer as chat history, and
``STAGE2_PROMPT`` is sent verbatim: its literal ``{question}`` token and doubled
braces are part of the prompt the model was trained against, so changing them
changes the model's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Sequence


MAX_ROUNDS = 2
MIN_CROP_AREA_RATIO = 0.25
MAX_CROP_AREA_RATIO = 0.75
MIN_CROP_SIZE = 28

GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 1024,
    "use_cache": True,
}

RANDOM_QUESTIONS = (
    "How would you judge the quality of this image?",
    "Could you evaluate the quality of this image?",
    "Can you rate the quality of this picture?",
    "What is your quality rating for this image?",
    "What's your opinion on the quality of this picture?",
)

STAGE1_PROMPT = """
You are an image quality expert. Rate the image (1.00–5.00) and decide if you need a crop for closer inspection.

OUTPUT FORMAT: <think>your_reasoning</think><answer>json</answer>

In <think>, write EXACTLY FOUR labeled sections (1–3 sentences each):
1) Image Quality Summary:
2) Directions for Improvement (positive-only, grounded):
3) Issues to Avoid (negative-only, grounded):
4) Decision & Rationale:

SECTION REQUIREMENTS
- Image Quality Summary: Concise, human-readable assessment of technical flaws (focus, motion blur, noise, exposure, lens flare, resolution, compression). End with a clear quality verdict.
- Directions for Improvement (positive-only, grounded): Purely aspirational description of the perfect final image **anchored to specific regions** you see (e.g., “central relief panel,” “flanking statues,” “upper frieze/lattice,” “lower glowing niche,” “surrounding columns,” “left/right chandeliers”). Mention **key relevant regions (aiming for 2-3)**. Only positive wording (no “remove/fix/correct,” no negatives).
- Issues to Avoid (negative-only, grounded): State existing technical problems and **tie each to one or more regions** (e.g., “blur on the central relief,” “blown highlights in the lower niche,” “color cast over the surrounding columns”). Use negative terms here only. Mention **key relevant regions (aiming for 2-3)**.
- Decision & Rationale: Give the initial rating and whether a crop is needed. If cropping, justify the bbox and what uncertainty it addresses; if final, explain why no crop is required. Briefly defend the numeric rating.

GLOBAL CONSTRAINTS
- No bullet lists; write compact paragraphs.
- Keep sections (2) strictly positive and (3) strictly negative.
- If a region name is unclear, use spatial anchors (e.g., “upper-right columns,” “lower-center panel”).
- Do not reference these instructions.

ANSWER JSON (inside <answer>)
- Keys: {{"bbox_2d": [x1, y1, x2, y2], "rating": <two-decimal>, "tool": "crop"|"final"}}
- When tool="final": bbox_2d=[0,0,0,0]
- rating: 1.00–5.00 with exactly two decimals
- where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the region that you want to zoom in, respectively (suppose the width and height of the image are 1.0)

ACTION RULES
- If image quality is clear: bbox_2d=[0,0,0,0], tool="final"
- If you need a closer look: bbox_2d=[x1,y1,x2,y2], tool="crop"
- Prefer a single, most-informative crop (faces, text, fine detail, high-contrast edges).
- Choose HARD CASE when you have ANY uncertainty to investigate.

FORMAT EXAMPLE:
<think>
Image Quality Summary: [...]
Directions for Improvement (positive-only, grounded): [...]
Issues to Avoid (negative-only, grounded): [...]
Decision & Rationale: [...]
</think><answer>{{"bbox_2d": [x1, y1, x2, y2],"rating": <two-decimal>,"tool": "crop"|"final"}}</answer>

USER QUESTION: {question}
"""

STAGE2_PROMPT = """
You are analyzing the original image and the provided crop for a final quality assessment. You must reference your previous uncertainty.

OUTPUT FORMAT: <think>your_reasoning</think><answer>json</answer>

In <think>, write EXACTLY FOUR labeled sections (1–3 sentences each):
1) Crop Inspection Summary:
2) Directions for Improvement (positive-only, grounded):
3) Issues to Avoid (negative-only, grounded):
4) Final Decision & Rationale:

SECTION REQUIREMENTS
- Crop Inspection Summary: Summarize what the crop **confirmed or revealed** about technical flaws (focus, noise, compression, etc.) that were uncertain in the full view (as noted **in your previous assessment**). End with the final quality verdict.
- Directions for Improvement (positive-only, grounded): Based on the **new details** from the crop, describe the aspirational perfect image. Anchor to **specific sub-regions** visible *within the crop* (e.g., "stitching on the collar," "fine text on the sign," "wood grain").
- Issues to Avoid (negative-only, grounded): State the technical problems **confirmed or newly discovered** in the crop. Tie each issue to **specific sub-regions** (e.g., "chromatic aberration on the window frame," "soft focus on the left eye," "compression artifacts in the background").
- Final Decision & Rationale: State the previous rating and the reason for the crop (from context history). Explain exactly what the crop revealed. Justify the new final rating by explaining how the crop's findings led to an upgrade, downgrade, or confirmation. Your adjustment must be proportional to the evidence. Avoid "token" or "imperceptible" adjustments (e.g., changes of 0.01-0.04), as these fail to meaningfully reflect the new findings. If the crop's evidence is clear and significant, ensure your rating adjustment is decisive and substantial.

GLOBAL CONSTRAINTS
- No bullet lists; write compact paragraphs.
- Keep sections (2) strictly positive and (3) strictly negative.
- Do not reference these instructions.

ANSWER JSON (inside <answer>)
- Keys: {{"bbox_2d": [0, 0, 0, 0], "rating": <two-decimal>, "tool": "final"}}
- rating: 1.00–5.00 with exactly two decimals. This final score must reflect the decision (upgrade, downgrade, or confirmation) explained in your rationale.

USER QUESTION: {question}
"""


@dataclass(frozen=True)
class ParsedAnswer:
    bbox: Any
    rating: Any
    tool: Any
    requests_crop: bool


class AnswerParseError(ValueError):
    """Raised when a completion does not contain a usable answer object."""


def _loads_relaxed(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as strict_error:
        try:
            import dirtyjson
        except ImportError:
            raise AnswerParseError("answer is not strict JSON and dirtyjson is unavailable") from strict_error
        try:
            return dirtyjson.loads(payload)
        except Exception as relaxed_error:
            raise AnswerParseError("answer JSON could not be parsed") from relaxed_error


def _answer_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise AnswerParseError("completion is not text")
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, re.DOTALL)
    if match is None:
        raise AnswerParseError("missing <answer>{...}</answer>")
    value = _loads_relaxed(match.group(1))
    if not isinstance(value, dict):
        raise AnswerParseError("answer payload is not an object")
    return value


def extract_rating(text: str) -> float:
    """Extract the final score from an ``<answer>`` object.

    Only a numeric ``rating`` is required here. ``bbox_2d`` and ``tool`` are
    validated for crop routing, not for deciding whether a score is usable.
    """
    value = _answer_object(text)
    try:
        rating = float(value["rating"])
    except (KeyError, TypeError, ValueError) as error:
        raise AnswerParseError("rating is missing or non-numeric") from error
    if not math.isfinite(rating):
        raise AnswerParseError("rating is not finite")
    return rating


def parse_answer(text: str) -> ParsedAnswer:
    """Parse the object needed for the crop-routing decision."""
    value = _answer_object(text)
    try:
        raw_bbox = value["bbox_2d"]
    except KeyError as error:
        raise AnswerParseError("bbox_2d is missing") from error
    try:
        rating = value["rating"]
    except KeyError as error:
        raise AnswerParseError("rating is missing") from error
    try:
        tool = value["tool"]
    except KeyError as error:
        raise AnswerParseError("tool is missing") from error
    try:
        all_zero = all(coordinate == 0 for coordinate in raw_bbox)
    except TypeError as error:
        raise AnswerParseError("bbox_2d is not iterable") from error
    return ParsedAnswer(
        bbox=raw_bbox,
        rating=rating,
        tool=tool,
        requests_crop=not (all_zero or tool == "final"),
    )


def rating_to_score(rating: float) -> float:
    """Clip a 1–5 rating and map it to the benchmark's 0–100 scale."""
    return (min(5.0, max(1.0, float(rating))) - 1.0) * 25.0


def build_stage1_message(question: str, image_path: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": STAGE1_PROMPT.format(question=question)},
                {"type": "image", "image": f"file://{image_path}"},
            ],
        }
    ]


def build_stage2_messages(question: str, first_answer: str) -> list[dict[str, Any]]:
    """Build the second-round chat context; images are supplied separately."""
    return [
        {"role": "user", "content": [{"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": first_answer}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": STAGE2_PROMPT},
                {"type": "text", "text": "Original image: "},
                {"type": "image"},
                {"type": "text", "text": "Cropped Image: "},
                {"type": "image"},
            ],
        },
    ]


def adjust_bbox(
    bbox: Sequence[float],
    width: int,
    height: int,
    min_size: int = MIN_CROP_SIZE,
    min_area_ratio: float = MIN_CROP_AREA_RATIO,
    max_area_ratio: float = MAX_CROP_AREA_RATIO,
) -> tuple[list[int], list[str]]:
    """Clamp a bbox into the allowed crop-area and minimum-size range."""
    reasons: list[str] = []
    x1, y1, x2, y2 = (int(value) for value in bbox)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x1 == x2:
        x2 = min(x1 + 1, width)
    if y1 == y2:
        y2 = min(y1 + 1, height)

    bbox_width, bbox_height = x2 - x1, y2 - y1
    bbox_area = bbox_width * bbox_height
    full_area = width * height
    scale_factor = 1.0
    if 0 < bbox_area < full_area * min_area_ratio:
        scale_factor = math.sqrt(full_area * min_area_ratio / bbox_area)
        reasons.append("area too small")
    elif bbox_area > full_area * max_area_ratio:
        scale_factor = math.sqrt(full_area * max_area_ratio / bbox_area)
        reasons.append("area too large")
    if scale_factor != 1.0:
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        new_width, new_height = bbox_width * scale_factor, bbox_height * scale_factor
        x1, x2 = max(0, center_x - new_width / 2), min(width, center_x + new_width / 2)
        y1, y2 = max(0, center_y - new_height / 2), min(height, center_y + new_height / 2)

    bbox_width, bbox_height = x2 - x1, y2 - y1
    if bbox_width < min_size:
        delta = min_size - bbox_width
        reasons.append(f"width < {min_size}px")
        x1 -= delta // 2
        x2 += (delta + 1) // 2
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > width:
            x1 -= x2 - width
            x2 = width
        x1 = max(0, x1)
    if bbox_height < min_size:
        delta = min_size - bbox_height
        reasons.append(f"height < {min_size}px")
        y1 -= delta // 2
        y2 += (delta + 1) // 2
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 > height:
            y1 -= y2 - height
            y2 = height
        y1 = max(0, y1)

    target_width, target_height = min(min_size, width), min(min_size, height)
    bbox_width, bbox_height = x2 - x1, y2 - y1
    if bbox_width < target_width:
        need = target_width - bbox_width
        add_left = min(need // 2, max(0, x1))
        add_right = min(need - add_left, max(0, width - x2))
        x1, x2 = x1 - add_left, x2 + add_right
        if x2 - x1 < target_width and width >= target_width:
            x1 = max(0, min(x1 - (target_width - (x2 - x1)), width - target_width))
            x2 = x1 + target_width
        reasons.append(f"final enforce width >= {target_width}px")
    bbox_height = y2 - y1
    if bbox_height < target_height:
        need = target_height - bbox_height
        add_top = min(need // 2, max(0, y1))
        add_bottom = min(need - add_top, max(0, height - y2))
        y1, y2 = y1 - add_top, y2 + add_bottom
        if y2 - y1 < target_height and height >= target_height:
            y1 = max(0, min(y1 - (target_height - (y2 - y1)), height - target_height))
            y2 = y1 + target_height
        reasons.append(f"final enforce height >= {target_height}px")

    result = [
        int(max(0, min(x1, width))),
        int(max(0, min(y1, height))),
        int(max(0, min(x2, width))),
        int(max(0, min(y2, height))),
    ]
    return result, reasons


def crop_bbox_for_stage2(
    bbox: Any, width: int, height: int
) -> tuple[list[int], bool]:
    """Return the pixel bbox and whether the central-crop fallback was used."""
    needs_central_crop = not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4
    if not needs_central_crop:
        try:
            values = [float(coordinate) for coordinate in bbox]
        except (TypeError, ValueError):
            needs_central_crop = True
        else:
            needs_central_crop = all(value == 0 for value in values)
    if needs_central_crop:
        return [width // 4, height // 4, width * 3 // 4, height * 3 // 4], True

    x1, y1, x2, y2 = values
    if all(0.0 <= coordinate <= 1.0 for coordinate in values):
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x1 == x2:
        if x2 < width:
            x2 += 1
        else:
            x1 -= 1
    if y1 == y2:
        if y2 < height:
            y2 += 1
        else:
            y1 -= 1
    pixel_bbox, _ = adjust_bbox([x1, y1, x2, y2], width, height)
    return pixel_bbox, False


def crop_for_stage2(image: Any, normalized_bbox: Sequence[float]) -> tuple[Any, list[int]]:
    """Crop the requested region and resize it to the original aspect ratio."""
    width, height = image.size
    pixel_bbox, used_central_fallback = crop_bbox_for_stage2(
        normalized_bbox, width, height
    )
    cropped = image.crop(pixel_bbox)
    if used_central_fallback:
        return cropped, pixel_bbox
    original_ratio = cropped.width / cropped.height
    target_ratio = width / height
    if original_ratio > target_ratio:
        new_width, new_height = width, int(width / original_ratio)
    else:
        new_height, new_width = height, int(height * original_ratio)
    from PIL import Image

    resized = cropped.resize(
        (max(1, new_width), max(1, new_height)),
        resample=Image.Resampling.LANCZOS,
    )
    return resized, pixel_bbox
