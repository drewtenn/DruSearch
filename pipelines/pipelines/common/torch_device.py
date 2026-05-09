"""Helpers for selecting PyTorch execution devices."""

from __future__ import annotations

from typing import Any

VALID_TORCH_DEVICES = {"auto", "cpu", "cuda", "mps"}


def resolve_torch_device(requested: str | None, env_name: str, torch_module: Any | None = None) -> str:
    """Resolve an explicit or auto PyTorch device name."""
    normalized = (requested or "auto").strip().lower()
    if normalized not in VALID_TORCH_DEVICES:
        valid = ", ".join(sorted(VALID_TORCH_DEVICES))
        raise ValueError(f"{env_name} must be one of: {valid}")

    torch = torch_module
    if torch is None:
        import torch as torch

    if normalized == "auto":
        if _cuda_available(torch):
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"

    if normalized == "cuda" and not _cuda_available(torch):
        raise RuntimeError(f"{env_name}=cuda requested but PyTorch CUDA is not available")
    if normalized == "mps" and not _mps_available(torch):
        raise RuntimeError(f"{env_name}=mps requested but PyTorch MPS is not available")
    return normalized


def move_wrapped_model_to_device(wrapper: Any, device: str) -> Any:
    """Move a wrapped PyTorch model to the resolved device when possible."""
    inner_model = getattr(wrapper, "model", None)
    move = getattr(inner_model, "to", None)
    if callable(move):
        move(device)
    return wrapper


def _cuda_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    return bool(cuda is not None and cuda.is_available())


def _mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps is not None and mps.is_available())
