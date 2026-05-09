"""Lazily-loaded sentence-transformer singleton."""

from __future__ import annotations

import os
import threading
from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

_MODEL_NAME = os.getenv("EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")
_RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_LOAD_LOCK = threading.Lock()
_RERANKER_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    with _LOAD_LOCK:
        return SentenceTransformer(_MODEL_NAME)


def model_name() -> str:
    return _MODEL_NAME


def embedding_dim() -> int:
    return int(get_model().get_sentence_embedding_dimension())


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    with _RERANKER_LOAD_LOCK:
        return CrossEncoder(_RERANKER_MODEL_NAME)


def reranker_name() -> str:
    return _RERANKER_MODEL_NAME


def rerank_scores(query: str, documents: list[str]) -> list[float]:
    pairs = [[query, doc] for doc in documents]
    scores = get_reranker().predict(pairs)
    return [float(x) for x in scores]
