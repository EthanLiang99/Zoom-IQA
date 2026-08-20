"""Runtime information recorded alongside every evaluation run."""

from __future__ import annotations

from importlib import metadata
import json
import os
import platform
import sys
from typing import Any


REPORTED_PACKAGES = (
    "accelerate",
    "dirtyjson",
    "flash-attn",
    "numpy",
    "pillow",
    "qwen-vl-utils",
    "safetensors",
    "torch",
    "torchvision",
    "transformers",
)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Describe the interpreter, packages, and GPUs used by a run."""
    payload: dict[str, Any] = {
        "python": sys.version,
        "python_version_info": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in REPORTED_PACKAGES},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        payload["torch_cuda_version"] = torch.version.cuda
        payload["cuda_available"] = torch.cuda.is_available()
        payload["gpu_count"] = torch.cuda.device_count()
        payload["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    except Exception as error:
        payload["torch_probe_error"] = f"{type(error).__name__}: {error}"
    return payload


def main() -> None:
    print(json.dumps(collect_environment(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
