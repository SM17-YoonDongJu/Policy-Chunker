"""임베딩 생성.

EMBED_BACKEND 환경변수:
  ollama              → qwen3:embedding 1024d
  sentence_transformers → BGE-M3 1024d
출처: rag/pipeline/embedder.py
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

from .models import InsuranceChunk

logger = logging.getLogger(__name__)

EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "ollama")
OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("EMBED_MODEL",  "qwen3:embedding")
OLLAMA_DIM   = int(os.environ.get("EMBED_DIM", "1024"))
ST_MODEL     = os.environ.get("ST_MODEL", "BAAI/bge-m3")
ST_DIM       = int(os.environ.get("ST_DIM",   "1024"))

_BATCH_SIZE  = 32
_RETRY_MAX   = 3
_RETRY_DELAY = 2.0


def get_embed_dim() -> int:
    return ST_DIM if EMBED_BACKEND == "sentence_transformers" else OLLAMA_DIM


def _ollama_batch(texts: list[str], url: str, model: str) -> Optional[list[list[float]]]:
    try:
        resp = requests.post(f"{url}/api/embed",
                             json={"model": model, "input": texts}, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return embeddings
    except Exception:
        pass
    return None


def _ollama_single(text: str, url: str, model: str) -> list[float]:
    for attempt in range(1, _RETRY_MAX + 1):
        try:
            resp = requests.post(f"{url}/api/embeddings",
                                 json={"model": model, "prompt": text}, timeout=60)
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            if attempt == _RETRY_MAX:
                raise
            logger.warning(f"Ollama 재시도 {attempt}/{_RETRY_MAX}: {e}")
            time.sleep(_RETRY_DELAY)
    raise RuntimeError("Ollama 임베딩 실패")


def _embed_ollama(texts: list[str], url: str, model: str) -> list[list[float]]:
    from more_itertools import chunked  # type: ignore
    results: list[list[float]] = []
    batches = list(chunked(texts, _BATCH_SIZE))
    for i, batch in enumerate(batches):
        logger.info(f"  [Ollama] 배치 {i+1}/{len(batches)}")
        vectors = _ollama_batch(list(batch), url, model) or \
                  [_ollama_single(t, url, model) for t in batch]
        results.extend(vectors)
    return results


_st_model_instance = None


def _get_st_model(model_name: str):
    global _st_model_instance
    if _st_model_instance is not None:
        return _st_model_instance
    from sentence_transformers import SentenceTransformer  # type: ignore
    logger.info(f"sentence-transformers 로딩: {model_name}")
    _st_model_instance = SentenceTransformer(model_name)
    return _st_model_instance


def _embed_st(texts: list[str], model_name: str) -> list[list[float]]:
    from more_itertools import chunked  # type: ignore
    model = _get_st_model(model_name)
    results: list[list[float]] = []
    batches = list(chunked(texts, _BATCH_SIZE))
    for i, batch in enumerate(batches):
        logger.info(f"  [BGE-M3] 배치 {i+1}/{len(batches)}")
        vecs = model.encode(list(batch), normalize_embeddings=True, show_progress_bar=False)
        results.extend(vecs.tolist())
    return results


def _check_dim(vectors: list[list[float]], expected: int) -> None:
    for v in vectors:
        if len(v) != expected:
            raise RuntimeError(
                f"임베딩 차원 불일치: 예상 {expected}d, 실제 {len(v)}d. "
                f"EMBED_BACKEND={EMBED_BACKEND}"
            )


def embed_texts(texts: list[str], ollama_url: Optional[str] = None,
                model: Optional[str] = None) -> list[list[float]]:
    if EMBED_BACKEND == "sentence_transformers":
        vectors = _embed_st(texts, model or ST_MODEL)
        _check_dim(vectors, ST_DIM)
        return vectors
    url = (ollama_url or OLLAMA_URL).rstrip("/")
    vectors = _embed_ollama(texts, url, model or OLLAMA_MODEL)
    _check_dim(vectors, OLLAMA_DIM)
    return vectors


def embed_chunks(chunks: list[InsuranceChunk], ollama_url: Optional[str] = None,
                 model: Optional[str] = None) -> list[InsuranceChunk]:
    vectors = embed_texts([c.content for c in chunks], ollama_url=ollama_url, model=model)
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec
    return chunks
