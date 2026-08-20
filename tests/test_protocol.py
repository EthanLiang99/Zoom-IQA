from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path
import unittest
import unittest.mock

from PIL import Image

from zoomiqa.inference.evaluate import (
    _completion_token_lengths,
    _merge_unique_records,
    _read_jsonl,
    load_samples,
    parse_devices,
    summarize,
)
from zoomiqa.inference.metrics import correlations, fit_isotonic
from zoomiqa.inference.predict import _prepare_question, _public_prediction, _single_image_sample
from zoomiqa.inference.evaluate_vllm import _physical_device, sampling_config
from zoomiqa.inference.protocol import (
    STAGE1_PROMPT,
    STAGE2_PROMPT,
    AnswerParseError,
    adjust_bbox,
    build_stage2_messages,
    crop_bbox_for_stage2,
    extract_rating,
    parse_answer,
    rating_to_score,
)


class ProtocolTests(unittest.TestCase):
    def test_single_image_sample_accepts_user_image_and_custom_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "example.png"
            Image.new("RGB", (8, 6), color="white").save(image_path)
            sample = _single_image_sample(
                image_path,
                "Rate this image. <|image|>",
                random.Random(42),
            )

        self.assertEqual(sample.dataset, "single_image")
        self.assertEqual(sample.image_path, str(image_path.resolve()))
        self.assertEqual(sample.question, "Rate this image. ")
        self.assertEqual(sample.history_question, "Rate this image. <|image|>")

    def test_single_image_default_question_is_seeded(self) -> None:
        expected_rng = random.Random(42)
        from zoomiqa.inference.protocol import RANDOM_QUESTIONS

        expected = expected_rng.choice(RANDOM_QUESTIONS)
        self.assertEqual(_prepare_question(None, random.Random(42)), (expected, expected))
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            _prepare_question("   ", random.Random(42))

    def test_single_image_rejects_non_image_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-an-image.jpg"
            path.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot decode image"):
                _single_image_sample(path, None, random.Random(42))

    def test_public_single_image_result_has_no_benchmark_sentinel(self) -> None:
        base = {
            "image": "/tmp/example.png",
            "question": "Rate it",
            "valid": True,
            "rating": 4.25,
            "model_score": 81.25,
            "error": None,
            "routing_warning": None,
            "rounds": 2,
            "crop_bbox_pixels": [1, 2, 3, 4],
            "generated_tokens": [100, 50],
            "model_output": "final",
            "output_history": ["first"],
        }
        prediction = _public_prediction(base)
        self.assertEqual(prediction["rating"], 4.25)
        self.assertEqual(prediction["score_0_100"], 81.25)
        self.assertEqual(prediction["first_round_response"], "first")

        invalid = _public_prediction(
            {**base, "valid": False, "rating": -100, "model_score": -100}
        )
        self.assertIsNone(invalid["rating"])
        self.assertIsNone(invalid["score_0_100"])

    def test_vllm_sampling_is_nonzero_and_keeps_answer_stop(self) -> None:
        config = sampling_config(0.1, 0.95, 50, 1024)
        self.assertEqual(config["temperature"], 0.1)
        self.assertEqual(config["stop"], ["</answer>"])
        self.assertTrue(config["include_stop_str_in_output"])
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            sampling_config(0.0, 0.95, 50, 1024)

    def test_vllm_worker_preserves_cuda_visible_mapping(self) -> None:
        self.assertEqual(_physical_device(1, "0,3"), "3")
        self.assertEqual(_physical_device(4, None), "4")

    def test_batched_generation_lengths_ignore_right_padding(self) -> None:
        self.assertEqual(
            _completion_token_lengths(
                [[10, 11, 151643, 151643], [20, 21, 22, 23]],
                {151643, 151645},
            ),
            [3, 4],
        )
        self.assertEqual(_completion_token_lengths([[1, 2, 3]], set()), [3])

    def test_stage1_formats_question_and_stage2_is_sent_verbatim(self) -> None:
        self.assertIn("USER QUESTION: rate it", STAGE1_PROMPT.format(question="rate it"))
        self.assertIn("USER QUESTION: {question}", STAGE2_PROMPT)

    def test_full_first_answer_is_preserved(self) -> None:
        first = "<think>complete first analysis</think><answer>{\"rating\": 3}</answer>"
        messages = build_stage2_messages("question", first)
        self.assertEqual(messages[1]["content"][0]["text"], first)

    def test_routing_requires_full_answer_but_scoring_needs_only_rating(self) -> None:
        final_only = '<answer>{"rating": 3.25}</answer>'
        self.assertEqual(extract_rating(final_only), 3.25)
        with self.assertRaises(AnswerParseError):
            parse_answer(final_only)

        routed = '<answer>{"bbox_2d": [0.1, 0.2, 0.8, 0.9], "rating": "4.2", "tool": "crop"}</answer>'
        parsed = parse_answer(routed)
        self.assertTrue(parsed.requests_crop)
        self.assertEqual(parsed.rating, "4.2")

        odd_tool = '<answer>{"bbox_2d": [1, 2], "rating": "bad", "tool": "other"}</answer>'
        self.assertTrue(parse_answer(odd_tool).requests_crop)

    def test_rating_is_clipped_and_scaled(self) -> None:
        self.assertEqual(rating_to_score(0), 0)
        self.assertEqual(rating_to_score(3), 50)
        self.assertEqual(rating_to_score(6), 100)

    def test_bbox_policy_golden_cases(self) -> None:
        self.assertEqual(adjust_bbox([0, 0, 100, 100], 100, 100)[0], [6, 6, 93, 93])
        self.assertEqual(adjust_bbox([40, 40, 50, 50], 100, 100)[0], [20, 20, 70, 70])
        self.assertEqual(adjust_bbox([-10, 90, 200, 90], 100, 100)[0], [0, 72, 100, 100])
        self.assertEqual(
            crop_bbox_for_stage2([1, 0.5, 1, 0.5], 100, 80),
            ([72, 18, 100, 62], False),
        )
        self.assertEqual(
            crop_bbox_for_stage2([0, 0, 0, 0], 100, 80),
            ([25, 20, 75, 60], True),
        )
        self.assertEqual(
            crop_bbox_for_stage2([1, 2], 100, 80),
            ([25, 20, 75, 60], True),
        )

    def test_default_question_is_drawn_from_the_run_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").touch()
            annotation = root / "test_kadid_2k.json"
            annotation.write_text(
                '[{"id":"one","image":"image.png","gt_score":4.7}]',
                encoding="utf-8",
            )
            samples, _ = load_samples([annotation], root, seed=42, max_samples=None)

        from zoomiqa.inference.protocol import RANDOM_QUESTIONS

        expected = random.Random(42).choice(RANDOM_QUESTIONS)
        self.assertEqual(samples[0].question, expected)

    def test_existing_question_keeps_raw_history_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").touch()
            annotation = root / "demo.json"
            annotation.write_text(
                '[{"image":"image.png","gt_score":1,"conversations":'
                '[{"value":"Rate it\\n<|image|>"}]}]',
                encoding="utf-8",
            )
            samples, _ = load_samples([annotation], root, seed=42, max_samples=None)
        self.assertEqual(samples[0].question, "Rate it\n")
        self.assertEqual(samples[0].history_question, "Rate it\n<|image|>")

    def test_zero_visible_devices_fails_cleanly(self) -> None:
        with self.assertRaisesRegex(ValueError, "no visible CUDA devices"):
            parse_devices("all", 0)

    def test_resume_keeps_complete_records_before_truncated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.jsonl"
            path.write_text('{"index": 0}\n{"index":', encoding="utf-8")
            self.assertEqual(
                _read_jsonl(path, tolerate_truncated_final_line=True),
                [{"index": 0}],
            )
            with self.assertRaises(ValueError):
                _read_jsonl(path)

    def test_resume_deduplicates_identical_records_and_rejects_conflicts(self) -> None:
        record = {"index": 0, "valid": True}
        self.assertEqual(_merge_unique_records([[record], [dict(record)]], 1), [record])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            _merge_unique_records([[record], [{"index": 0, "valid": False}]], 1)

    def test_summary_tracks_format_and_routing_separately(self) -> None:
        summary = summarize(
            [
                {
                    "dataset": "demo",
                    "valid": True,
                    "rounds": 1,
                    "error": None,
                    "routing_warning": "invalid_bbox",
                }
            ]
        )
        dataset = summary["datasets"]["demo"]
        self.assertEqual(dataset["invalid"], 0)
        self.assertEqual(dataset["routing_warnings"], {"invalid_bbox": 1})

    def test_correlations_report_the_fitted_pair(self) -> None:
        try:
            scores = correlations([1, 2, 3], [10, 20, 30])
        except RuntimeError:
            self.skipTest("scikit-learn is not installed")
        self.assertEqual(sorted(scores), ["fit_plcc", "fit_srcc"])
        self.assertAlmostEqual(scores["fit_plcc"], 1.0, places=5)
        self.assertAlmostEqual(scores["fit_srcc"], 1.0, places=5)
        nonlinear = correlations([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0])
        self.assertAlmostEqual(nonlinear["fit_plcc"], 1.0, places=5)

    def test_inverted_ranking_collapses_the_increasing_fit(self) -> None:
        try:
            inverted = correlations([1, 1, 3, 4], [4, 3, 2, 1])
        except RuntimeError:
            self.skipTest("scikit-learn is not installed")
        # increasing=True maps every prediction to one constant here, so the
        # fitted correlations are undefined instead of strongly negative.
        self.assertIsNone(inverted["fit_plcc"])
        self.assertIsNone(inverted["fit_srcc"])

    def test_isotonic_fit_pools_tied_predictions(self) -> None:
        try:
            # Two samples share a prediction, so the fit must give them one
            # value: a plain pool-adjacent-violators pass would leave 0 and 10.
            fitted = fit_isotonic([1.0, 1.0, 2.0], [0.0, 10.0, 20.0])
        except RuntimeError:
            self.skipTest("scikit-learn is not installed")
        self.assertEqual(fitted[0], fitted[1])
        self.assertAlmostEqual(fitted[0], 5.0, places=5)

    def test_missing_scikit_learn_is_reported_not_silently_skipped(self) -> None:
        with unittest.mock.patch.dict(sys.modules, {"sklearn.isotonic": None}):
            with self.assertRaisesRegex(RuntimeError, "scikit-learn is required"):
                fit_isotonic([1.0, 2.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
