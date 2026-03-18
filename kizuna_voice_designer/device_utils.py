"""Device helpers shared across CPU/CUDA/DirectML paths."""

from __future__ import annotations

from typing import Any

import torch


def normalize_device_str(device_str: str | None) -> str:
    raw = (device_str or "cpu").strip().lower()
    if raw in {"dml", "directml", "privateuseone", "privateuseone:0"}:
        return "directml" if _has_directml() else "cpu"
    if raw.startswith("cuda") and torch.cuda.is_available():
        try:
            req_idx = int(raw.split(":")[1])
        except (IndexError, ValueError):
            req_idx = 0
        max_idx = torch.cuda.device_count() - 1
        if max_idx < 0:
            return "cpu"
        return f"cuda:{min(req_idx, max_idx)}"
    return "cpu"


def resolve_device(device_str: str | None) -> Any:
    normalized = normalize_device_str(device_str)
    if normalized == "directml":
        import torch_directml

        return torch_directml.device()
    return torch.device(normalized)


def tts_device_str(device_str: str | None) -> str:
    normalized = normalize_device_str(device_str)
    if normalized == "directml":
        return "privateuseone:0"
    return normalized


def is_cuda_device(device: Any) -> bool:
    return getattr(device, "type", "") == "cuda"


def is_directml_device(device: Any) -> bool:
    return str(device).startswith("privateuseone")


def has_gpu_backend(device_str: str | None) -> bool:
    normalized = normalize_device_str(device_str)
    return normalized.startswith("cuda") or normalized == "directml"


def _has_directml() -> bool:
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        return False
    return True
