"""FastAPI sidecar that turns text into normalized embedding vectors."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import embedding_dim, get_model, model_name

logger = logging.getLogger("embedder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t = time.perf_counter()
    get_model()
    logger.info("model loaded name=%s dim=%d in %.2fs", model_name(), embedding_dim(), time.perf_counter() - t)
    yield


app = FastAPI(title="drusearch-embedder", version="0.1.0", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)


class EmbedResponse(BaseModel):
    vector: List[float]
    dim: int
    model: str


class BatchEmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=256)


class BatchEmbedResponse(BaseModel):
    vectors: List[List[float]]
    dim: int
    model: str


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": model_name()}


@app.get("/readyz")
def readyz() -> dict:
    return {"status": "ok", "dim": embedding_dim()}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    try:
        vec = get_model().encode(req.text, normalize_embeddings=True).tolist()
    except Exception as exc:
        logger.exception("embed failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return EmbedResponse(vector=vec, dim=len(vec), model=model_name())


@app.post("/embed_batch", response_model=BatchEmbedResponse)
def embed_batch(req: BatchEmbedRequest) -> BatchEmbedResponse:
    vecs = get_model().encode(req.texts, normalize_embeddings=True, batch_size=64).tolist()
    return BatchEmbedResponse(vectors=vecs, dim=len(vecs[0]), model=model_name())
