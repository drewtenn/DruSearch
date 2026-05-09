"""Lazily-loaded sentence-transformer singleton."""

from __future__ import annotations

import os
import threading
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.torch_device import resolve_torch_device

_MODEL_NAME = os.getenv("EMBEDDER_MODEL", "BAAI/bge-base-en-v1.5")
_MODEL_DEVICE = os.getenv("EMBEDDER_DEVICE", "auto")
_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    with _LOAD_LOCK:
        device = resolve_torch_device(_MODEL_DEVICE, "EMBEDDER_DEVICE")
        return SentenceTransformer(_MODEL_NAME, device=device)


def model_name() -> str:
    return _MODEL_NAME


def embedding_dim() -> int:
    return int(get_model().get_sentence_embedding_dimension())
