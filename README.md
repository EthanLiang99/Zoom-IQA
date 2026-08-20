<h1 align="center">
  <img src="https://raw.githubusercontent.com/EthanLiang99/ZOOMIQA-Projectpage/main/zoomiqa/zoomiqa_logo.png" width="32" alt="">
  Zoom-IQA
</h1>

<p align="center">
  <b>Image Quality Assessment with Reliable Region-Aware Reasoning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ECCV-2026-6f42c1.svg" alt="ECCV 2026">
  <a href="https://arxiv.org/abs/2601.02918"><img src="https://img.shields.io/badge/arXiv-2601.02918-b31b1b.svg" alt="arXiv"></a>
  <a href="https://ethanliang99.github.io/ZOOMIQA-Projectpage/"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg" alt="Project page"></a>
  <a href="https://huggingface.co/Ethanliang99/Zoom-IQA-7B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Zoom--IQA--7B-ffcc4d.svg" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-3da639.svg" alt="License"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/EthanLiang99/ZOOMIQA-Projectpage/main/zoomiqa/cvpr26_teaser.png" width="100%" alt="Zoom-IQA teaser">
</p>

Official evaluation code for **Zoom-IQA: Image Quality Assessment with Reliable
Region-Aware Reasoning**.

[Paper](https://arxiv.org/abs/2601.02918) ·
[Project page](https://ethanliang99.github.io/ZOOMIQA-Projectpage/)

The inference checkpoint is available on Hugging Face:
[Ethanliang99/Zoom-IQA-7B](https://huggingface.co/Ethanliang99/Zoom-IQA-7B).
The GR-IQA training data and training code are planned for a later release.

Benchmark images are not redistributed. Download each benchmark from its
official source and provide the local paths described below.

<p align="center">
  <img src="https://raw.githubusercontent.com/EthanLiang99/ZOOMIQA-Projectpage/main/zoomiqa/cvpr26_framework.png" width="100%" alt="Zoom-IQA framework">
</p>

## Installation

Use separate environments for the Transformers and vLLM evaluators.

### Transformers

```bash
conda create -n zoom-iqa python=3.11 -y
conda activate zoom-iqa
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e . --no-deps
zoomiqa-check-env
```

### vLLM

```bash
conda create -n zoom-iqa-vllm python=3.11 -y
conda activate zoom-iqa-vllm
python -m pip install -r requirements-vllm.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e . --no-deps
```

Do not install both requirements files in the same environment: they use
different PyTorch, Transformers, and image-processing versions.

## Try your own image

You do not need benchmark annotations to score a single image. Run this entrypoint
from the Transformers environment:

```bash
zoomiqa-score \
  --model-path Ethanliang99/Zoom-IQA-7B \
  --image /path/to/your_image.jpg \
  --device 0
```

Add `--question "How would you judge the quality of this image?"` to use a
custom question. The command prints JSON containing the 1–5 rating, normalized
0–100 score, crop location, round count, and model responses, and exits with
status `2` if the model produced an unparsable answer.

Decoding is sampled, so repeated calls on the same image do not return an
identical score. Benchmark numbers should come from `zoomiqa-eval` over a full
test set, not from single-image runs.

For applications that score more than one user image, load the model once:

```python
from zoomiqa.predict import ZoomIQAPredictor

predictor = ZoomIQAPredictor("Ethanliang99/Zoom-IQA-7B", device=0)
result = predictor.predict("/path/to/your_image.jpg")
print(result["rating"])
```

## Data format

Each annotation file is a JSON list. Every row must contain an image path and a
ground-truth score:

```json
{
  "id": "optional-stable-id",
  "image": "KONIQ/images/example.jpg",
  "gt_score": 3.7
}
```

`image` is resolved relative to `--image-root`. The legacy key `img_path` is
also accepted. A question may be supplied in `conversations[0]`; otherwise the
evaluator selects one of the frozen prompts with the run seed. Missing images
and malformed labels fail before model inference.

## Evaluation

The Transformers evaluator follows the two-round protocol:

```bash
zoomiqa-eval \
  --model-path Ethanliang99/Zoom-IQA-7B \
  --annotation /path/to/test_koniq_2k.json \
  --image-root /path/to/iqa \
  --output-dir outputs/koniq \
  --devices 0,1 \
  --batch-size 16
```

Use `--max-samples 16` for a smoke test and `--resume` to continue an
interrupted run. Each output directory contains `predictions.jsonl`,
`summary.json` with PLCC/SRCC, `run_config.json`, and `environment.json`.
Pass `--revision` to pin the checkpoint when `--model-path` is a Hub ID.

For faster inference with the separately installed vLLM environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 zoomiqa-eval-vllm \
  --model-path Ethanliang99/Zoom-IQA-7B \
  --annotation /path/to/test_koniq_2k.json \
  --image-root /path/to/iqa \
  --output-dir outputs/koniq-vllm \
  --devices 0,1 \
  --batch-size 64
```

The vLLM backend uses temperature `0.1` and an `</answer>` stop string. It is a
throughput-oriented variant, not an exact reproduction of the Transformers
runtime.

## License and citation

The code and model are released under Apache-2.0. Dataset images retain their
source-specific licenses.

```bibtex
@article{liang2026zoomiqa,
  title={Zoom-IQA: Image Quality Assessment with Reliable Region-Aware Reasoning},
  author={Liang, Guoqiang and Wang, Jianyi and Wu, Zhonghua and Zhou, Shangchen and Loy, Chen Change},
  journal={arXiv preprint arXiv:2601.02918},
  year={2026}
}
```
