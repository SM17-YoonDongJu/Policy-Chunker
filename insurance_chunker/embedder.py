"""임베딩 생성.

EMBED_BACKEND 환경변수:
  ollama              → qwen3-embedding:0.6b 1024d
  voyage              → voyage-4-large 1024d (Voyage API, VOYAGE_API_KEY 필요)
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
EMBED_DIM    = int(os.environ.get("EMBED_DIM", "1024"))
ST_MODEL     = os.environ.get("ST_MODEL", "BAAI/bge-m3")
ST_DIM       = int(os.environ.get("ST_DIM",   "1024"))

# Voyage(MongoDB Atlas 모델 API). 검색 쪽(AI 레포 report 워커의 EMBEDDING_PROVIDER=voyage)과
# 모델이 같아야 한다 — 둘 다 1024d라 어긋나도 오류 없이 검색 결과만 틀어진다. 차원은 모델
# 기본값에 기대지 않고 EMBED_DIM을 output_dimension으로 요청한다(코퍼스 컬럼 halfvec(1024)).
VOYAGE_URL     = os.environ.get("VOYAGE_BASE_URL", "https://ai.mongodb.com/v1").rstrip("/")
VOYAGE_MODEL   = os.environ.get("EMBED_MODEL", "voyage-4-large")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")

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
    return ST_DIM if EMBED_BACKEND == "sentence_transformers" else EMBED_DIM


def model_name(override: Optional[str] = None) -> str:
    """지금 백엔드로 임베딩할 모델 이름. 단계 캐시 키가 이 값으로 갈린다."""
    if override:
        return override
    return {"voyage": VOYAGE_MODEL, "sentence_transformers": ST_MODEL}.get(EMBED_BACKEND,
                                                                          OLLAMA_MODEL)


class EmbedConfigError(RuntimeError):
    """다시 시도해도 낫지 않는 임베딩 실패 — 키 없음, 인증 실패, 모르는 모델."""


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


def _voyage_too_many_tokens(resp) -> bool:
    """요청당 토큰 한도(voyage-4-large 120K) 초과인가. 묶음을 나누면 풀리는 유일한 400이다."""
    try:
        return resp.json().get("error_code") == "TOO_MANY_TOKENS_IN_BATCH"
    except ValueError:
        return False


def _voyage_request(texts: list[str], model: str, input_type: str,
                    stats: dict[str, int]) -> list[list[float]]:
    """POST /embeddings 한 묶음. 응답은 index로 정렬해 입력 순서로 돌려준다.

    Ollama와 달리 건별 폴백이 없다 — 일시 오류(네트워크·429·5xx)는 같은 묶음을 지수 백오프로
    다시 보내고, 토큰 한도 초과는 반으로 나눠 보낸다. 그 밖의 4xx는 키·모델 설정 문제라
    재시도하지 않고 바로 올린다. 기다려 봐야 문서마다 같은 실패를 되풀이할 뿐이다.
    """
    reason = ""
    for attempt in range(1, _RETRY_MAX + 1):
        try:
            resp = requests.post(
                f"{VOYAGE_URL}/embeddings",
                headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
                json={"model": model, "input": texts, "input_type": input_type,
                      "output_dimension": EMBED_DIM},
                timeout=_BATCH_TIMEOUT)
        except requests.RequestException as e:
            reason = f"{type(e).__name__}: {e}"
        else:
            if resp.status_code == 200:
                data = resp.json()
                items = sorted(data["data"], key=lambda d: d["index"])
                if len(items) != len(texts):
                    raise RuntimeError(f"Voyage 응답 개수 불일치 요청 {len(texts)} != "
                                       f"응답 {len(items)}")
                stats["tokens"] += (data.get("usage") or {}).get("total_tokens", 0)
                return [d["embedding"] for d in items]
            if resp.status_code == 400 and len(texts) > 1 and _voyage_too_many_tokens(resp):
                stats["splits"] += 1
                mid = len(texts) // 2
                return (_voyage_request(texts[:mid], model, input_type, stats)
                        + _voyage_request(texts[mid:], model, input_type, stats))
            reason = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code != 429 and resp.status_code < 500:
                raise EmbedConfigError(f"Voyage 임베딩 실패 — {reason}")
        if attempt < _RETRY_MAX:
            stats["retries"] += 1
            delay = _RETRY_DELAY * 2 ** (attempt - 1)
            logger.warning(f"  [Voyage] 재시도 {attempt}/{_RETRY_MAX} ({reason})",
                           extra={"event": "embed_retry", "backend": "voyage",
                                  "attempt": attempt, "reason": reason})
            time.sleep(delay)
    raise RuntimeError(f"Voyage 임베딩 실패 {_RETRY_MAX}회 — {reason}")


def _embed_voyage(texts: list[str], model: str, input_type: str) -> list[list[float]]:
    if not VOYAGE_API_KEY:
        raise EmbedConfigError("EMBED_BACKEND=voyage에는 VOYAGE_API_KEY가 필요하다")
    from more_itertools import chunked  # type: ignore
    stats = {"tokens": 0, "splits": 0, "retries": 0}
    results: list[list[float]] = []
    batches = list(chunked(texts, _BATCH_SIZE))
    t0 = time.perf_counter()
    for i, batch in enumerate(batches):
        logger.info(f"  [Voyage] 배치 {i+1}/{len(batches)}")
        results.extend(_voyage_request(list(batch), model, input_type, stats))

    elapsed = round(time.perf_counter() - t0, 1)
    # 토큰 수는 과금 단위다. 문서마다 남겨 두면 적재 비용을 로그에서 바로 합산할 수 있다.
    logger.info(f"  [Voyage] {len(batches)}배치 완료 ({elapsed}s, {stats['tokens']}토큰)",
                extra={"event": "embed_done", "backend": "voyage", "texts": len(texts),
                       "batches": len(batches), "batch_size": _BATCH_SIZE,
                       "tokens": stats["tokens"], "split_batches": stats["splits"],
                       "retries": stats["retries"], "elapsed_s": elapsed})
    return results


def _check_dim(vectors: list[list[float]], expected: int) -> None:
    for v in vectors:
        if len(v) != expected:
            raise RuntimeError(
                f"임베딩 차원 불일치: 예상 {expected}d, 실제 {len(v)}d. "
                f"EMBED_BACKEND={EMBED_BACKEND}"
            )


def embed_texts(texts: list[str], ollama_url: Optional[str] = None,
                model: Optional[str] = None, input_type: str = "document") -> list[list[float]]:
    """input_type(document | query)은 voyage만 요청에 싣는다."""
    if EMBED_BACKEND == "voyage":
        # 장문 절단을 하지 않는다. 1800자 상한은 qwen3-embedding 러너 크래시와 eval 재현 때문이고,
        # Voyage는 32K 토큰까지 받는다(넘으면 서버가 자른다). AI 레포의 재임베딩(rag.reembed)도
        # 원문 그대로 보내므로, 여기서 자르면 같은 모델로 다시 돌렸을 때 벡터가 달라진다.
        # 빈 문자열은 Voyage도 400으로 거절해 공백으로 바꾼다.
        vectors = _embed_voyage([t or " " for t in texts], model or VOYAGE_MODEL, input_type)
        _check_dim(vectors, EMBED_DIM)
        return vectors
    # 문서와 질의 모두 같은 상한을 거쳐야 한다 — eval도 양쪽에 같은 절단을 적용하므로
    # 여기가 어긋나면 평가 수치가 재현되지 않는다.
    texts = _truncate(texts)
    if EMBED_BACKEND == "sentence_transformers":
        vectors = _embed_st(texts, model or ST_MODEL)
        _check_dim(vectors, ST_DIM)
        return vectors
    url = (ollama_url or OLLAMA_URL).rstrip("/")
    vectors = _embed_ollama(texts, url, model or OLLAMA_MODEL)
    _check_dim(vectors, EMBED_DIM)
    return vectors


def embed_query(query: str, ollama_url: Optional[str] = None,
                model: Optional[str] = None) -> list[float]:
    """검색 질의용 임베딩. instruct 프리픽스를 붙여 문서 벡터와 비대칭 정합시킨다.

    문서는 embed_chunks가 원문 그대로 색인하므로, 질의만 프리픽스를 붙여야
    qwen3-embedding이 학습된 사용법·eval 측정 조건과 일치한다.
    voyage는 프리픽스 없이 input_type=query로 보낸다. 업체가 검색용 프롬프트를 직접 붙이므로
    여기서도 붙이면 이중 프리픽스가 된다(AI 레포 rag.search와 같은 방식).
    """
    if EMBED_BACKEND == "voyage":
        return embed_texts([query], model=model, input_type="query")[0]
    return embed_texts([QUERY_INSTRUCT + query], ollama_url=ollama_url, model=model)[0]


def preflight() -> None:
    """적재를 시작하기 전에 임베딩 설정을 확인한다. voyage만 본다.

    키가 없거나 틀리면 문서마다 레이아웃·VLM을 다 돌린 뒤 임베딩에서 실패하고, 그 실패가
    재시도 상한(INGEST_MAX_RETRY)에 쌓여 멀쩡한 문서까지 격리된다. 짧은 입력 하나를 먼저
    보내 설정 오류면 사이클을 시작하지 않는다. 일시 오류는 문서 단계의 재시도에 맡긴다.
    ollama는 같은 노드에서 청커보다 늦게 뜰 수 있어 여기서 막지 않는다(기존 동작 그대로).

    Raises:
        EmbedConfigError: 키가 없거나 Voyage가 키·모델을 거절했다.
    """
    if EMBED_BACKEND != "voyage":
        return
    try:
        embed_texts(["보험약관"])
    except EmbedConfigError:
        raise
    except Exception as e:  # noqa: BLE001 - 일시 오류는 막지 않고 문서 단계로 넘긴다
        logger.warning(f"임베딩 사전 확인 실패 — 일시 오류로 보고 계속한다 ({e})",
                       extra={"event": "embed_preflight_failed", "backend": "voyage"})


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
