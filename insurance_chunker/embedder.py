"""임베딩 생성.

EMBED_BACKEND 환경변수:
  ollama              → qwen3-embedding:0.6b 1024d
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
OLLAMA_MODEL = os.environ.get("EMBED_MODEL",  "qwen3-embedding:0.6b")
OLLAMA_DIM   = int(os.environ.get("EMBED_DIM", "1024"))
ST_MODEL     = os.environ.get("ST_MODEL", "BAAI/bge-m3")
ST_DIM       = int(os.environ.get("ST_DIM",   "1024"))

# qwen3-embedding 비대칭 검색: 질의에만 instruct 프리픽스를 붙이고 문서는 원문 그대로
# 색인한다. eval/retrieval_eval.py와 동일 문자열이어야 평가 수치가 재현된다.
QUERY_INSTRUCT = ("Instruct: Given a Korean insurance policy question, "
                  "retrieve relevant policy clauses that answer the question.\nQuery: ")

# 배치가 클수록 HTTP 왕복이 줄어 GPU가 연속으로 일한다. 실측(T4)에서 SM 사용률이
# 중간중간 62~70%로 떨어지는 구간이 관찰됐는데, 그게 배치 사이에 GPU가 비는 시간이다.
# 다만 상한은 GPU 전력이다 — T4는 임베딩 부하에서 이미 70W 캡을 넘겨 부스트 클럭이
# 1590→1200MHz까지 깎인다. 무한정 키운다고 선형으로 빨라지지 않으므로 실측으로 찾는다.
_BATCH_SIZE  = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
_RETRY_MAX   = int(os.environ.get("EMBED_RETRY_MAX", "3"))
_RETRY_DELAY = float(os.environ.get("EMBED_RETRY_DELAY", "2.0"))

# qwen3-embedding 러너가 장문 입력에서 크래시(EOF)한다. eval/retrieval_eval.py:150이
# 같은 상한으로 절단하는데 운영 경로엔 없었다 — 값이 다르면 색인 벡터와 평가 벡터가
# 달라져 eval 수치(R@5 0.894 등)가 재현되지 않으므로 반드시 같이 움직여야 한다.
_MAX_CHARS   = int(os.environ.get("EMBED_MAX_CHARS", "1800"))

# 배치 타임아웃. GPU가 밀리면 여기서 터져 건별 폴백으로 떨어지므로, 배치를 키울 때
# 같이 올려야 한다(eval은 300초를 쓴다).
_BATCH_TIMEOUT = float(os.environ.get("EMBED_BATCH_TIMEOUT", "120"))


def _truncate(texts: list[str]) -> list[str]:
    """장문 절단 + 빈 문자열 방어. Ollama는 빈 입력에 오류를 낸다."""
    out, cut = [], 0
    for t in texts:
        if len(t) > _MAX_CHARS:
            cut += 1
        out.append(t[:_MAX_CHARS] or " ")
    if cut:
        logger.info(f"  장문 {cut}건 절단 ({_MAX_CHARS}자)",
                    extra={"event": "embed_truncated", "count": cut, "limit": _MAX_CHARS})
    return out


def get_embed_dim() -> int:
    return ST_DIM if EMBED_BACKEND == "sentence_transformers" else OLLAMA_DIM


def _ollama_batch(texts: list[str], url: str, model: str) -> Optional[list[list[float]]]:
    """배치 임베딩. 실패하면 None을 돌려주되 사유를 반드시 남긴다.

    예전엔 `except Exception: pass`로 전부 삼켰다. 폴백이 조용히 돌기 때문에 배치가
    깨져 건별로 떨어져도(배치 32건이면 최악 32 x 3회 x 2초 = 3분) 로그에 흔적이 없었고,
    "임베딩이 느린 이유"를 사후에 알 방법이 없었다.
    """
    reason: str
    try:
        resp = requests.post(f"{url}/api/embed",
                             json={"model": model, "input": texts}, timeout=_BATCH_TIMEOUT)
        if resp.status_code != 200:
            reason = f"HTTP {resp.status_code}: {resp.text[:200]}"
        else:
            embeddings = resp.json().get("embeddings")
            if not embeddings:
                reason = "응답에 embeddings 없음"
            elif len(embeddings) != len(texts):
                reason = f"개수 불일치 요청 {len(texts)} != 응답 {len(embeddings)}"
            else:
                return embeddings
    except Exception as e:  # noqa: BLE001 - 사유를 남기고 건별 폴백으로 넘긴다
        reason = f"{type(e).__name__}: {e}"

    logger.warning(
        f"  [Ollama] 배치 실패 → 건별 폴백 {len(texts)}건 ({reason})",
        extra={"event": "embed_batch_fallback", "batch_size": len(texts), "reason": reason})
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
    fallbacks = 0
    t0 = time.perf_counter()
    for i, batch in enumerate(batches):
        logger.info(f"  [Ollama] 배치 {i+1}/{len(batches)}")
        vectors = _ollama_batch(list(batch), url, model)
        if vectors is None:
            fallbacks += 1
            vectors = [_ollama_single(t, url, model) for t in batch]
        results.extend(vectors)

    elapsed = round(time.perf_counter() - t0, 1)
    # 폴백이 몇 번이었는지를 한 줄로 남긴다 — 이게 없으면 embed 시간이 실제 추론인지
    # 폴백 대기인지 사후에 구분할 수 없다(단계 타이밍은 둘을 합쳐서 보여준다).
    summary = {"event": "embed_done", "backend": "ollama", "texts": len(texts),
               "batches": len(batches), "batch_size": _BATCH_SIZE,
               "fallback_batches": fallbacks, "elapsed_s": elapsed}
    if fallbacks:
        logger.warning(f"  [Ollama] {len(batches)}배치 중 {fallbacks}배치가 건별 폴백 "
                       f"— 임베딩이 느렸다면 이것 때문이다 ({elapsed}s)", extra=summary)
    else:
        logger.info(f"  [Ollama] {len(batches)}배치 완료 ({elapsed}s)", extra=summary)
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
    # 문서와 질의 모두 같은 상한을 거쳐야 한다 — eval도 양쪽에 같은 절단을 적용하므로
    # 여기가 어긋나면 평가 수치가 재현되지 않는다.
    texts = _truncate(texts)
    if EMBED_BACKEND == "sentence_transformers":
        vectors = _embed_st(texts, model or ST_MODEL)
        _check_dim(vectors, ST_DIM)
        return vectors
    url = (ollama_url or OLLAMA_URL).rstrip("/")
    vectors = _embed_ollama(texts, url, model or OLLAMA_MODEL)
    _check_dim(vectors, OLLAMA_DIM)
    return vectors


def embed_query(query: str, ollama_url: Optional[str] = None,
                model: Optional[str] = None) -> list[float]:
    """검색 질의용 임베딩. instruct 프리픽스를 붙여 문서 벡터와 비대칭 정합시킨다.

    문서는 embed_chunks가 원문 그대로 색인하므로, 질의만 프리픽스를 붙여야
    qwen3-embedding이 학습된 사용법·eval 측정 조건과 일치한다.
    """
    return embed_texts([QUERY_INSTRUCT + query], ollama_url=ollama_url, model=model)[0]


def embed_chunks(chunks: list[InsuranceChunk], ollama_url: Optional[str] = None,
                 model: Optional[str] = None) -> list[InsuranceChunk]:
    """boilerplate 청크는 임베딩 생략 (저장은 되지만 벡터 검색 대상 아님)."""
    targets = [c for c in chunks if not c.is_boilerplate]
    skipped = len(chunks) - len(targets)
    if skipped:
        logger.info(f"boilerplate {skipped}청크 임베딩 생략")
    vectors = embed_texts([c.content for c in targets], ollama_url=ollama_url, model=model)
    for chunk, vec in zip(targets, vectors):
        chunk.embedding = vec
    return chunks
