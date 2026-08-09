"""SM17-YoonDongJu/AI 레포 계약 호환 검색 API.

`8-feature-rag-하이브리드-검색` 브랜치의 `src/rag/search.py` / `contracts.md` §5와
동일한 함수 시그니처·반환형(Chunk/Citation/RagResult)을 구현한다. 우리 policy_chunks
테이블이 그 레포의 `namespace="terms"` 데이터 소스이므로, 이 함수를 그대로 붙여
report_worker·chatbot 쪽 호출을 로컬에서 재현·검증한다.

원본과의 차이(의도적 확장, 契約 필드는 그대로):
  - namespace="case"(판례) 미지원 — 이 프로젝트는 policy_chunks(terms)만 다룸
  - rerank=True 옵션 추가 — eval/retrieval_eval.py에서 검증된 bge-reranker-v2-m3
    (기본 False로 두면 원본과 100% 동일 동작: RRF 점수만으로 반환)

사용법:
    import asyncpg
    pool = await asyncpg.create_pool(DATABASE_URL)
    result = await search(pool, "계단에서 넘어져서 골절됐는데 보험금 나와?", top_k=5)
    for c in result.ranked_chunks:
        print(c.namespace, c.score, c.text[:40])
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import asyncpg
import numpy as np
from pydantic import BaseModel

from insurance_chunker.embedder import embed_query
from insurance_chunker.tokenizer import tokenize_korean

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 8
_CANDIDATE_K = 50   # namespace별 벡터/키워드 후보 개수 (RRF 입력)
RRF_K = 60          # contracts.md/fusion.py와 동일 상수
KEYWORD_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

_RERANK_TOP = 20
_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# rerank 미지정 호출의 기본값. v5(호 분할) 청크셋은 rerank 전제로 검증됐으므로
# (embed 단독은 v3보다 낮음 — eval/GAP_ANALYSIS.md) 배포 환경에서 SEARCH_RERANK=1 권장.
_RERANK_DEFAULT = os.environ.get("SEARCH_RERANK", "0") == "1"

# 비신체보험(범위 밖) 힌트 — rag/router.py와 동일 목록
NON_BODILY_HINTS: tuple[str, ...] = (
    "자동차", "자차", "차량", "화재", "재물", "배상책임", "해상", "운송", "항공", "선박",
)


class Chunk(BaseModel):
    text: str
    namespace: str
    score: float
    source_ref: str


class Citation(BaseModel):
    clause_no: Optional[str] = None
    exhibit: Optional[str] = None
    source_url: Optional[str] = None


class RagResult(BaseModel):
    ranked_chunks: list[Chunk]
    citations: list[Citation]


class RagError(RuntimeError):
    """검색 파이프라인 실패의 기반 예외."""


def _is_out_of_scope(query: str, insurance_type: Optional[str]) -> bool:
    haystacks = [query] + ([insurance_type] if insurance_type else [])
    return any(hint in text for text in haystacks for hint in NON_BODILY_HINTS)


_KEYWORD_SQL = """
    SELECT chunk_id, content, article_number AS clause_no, section AS exhibit,
           ts_rank(content_tsv, plainto_tsquery('simple', $1)) AS rank
    FROM policy_chunks
    WHERE content_tsv @@ plainto_tsquery('simple', $1)
    ORDER BY rank DESC LIMIT $2
"""
_VECTOR_SQL = """
    SELECT chunk_id, content, article_number AS clause_no, section AS exhibit
    FROM policy_chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> $1 LIMIT $2
"""


async def _keyword_search(pool: asyncpg.Pool, tokens: str, k: int) -> list[asyncpg.Record]:
    if not tokens.strip():
        return []
    async with pool.acquire() as conn:
        return await conn.fetch(_KEYWORD_SQL, tokens, k)


async def _vector_search(pool: asyncpg.Pool, embedding: list[float], k: int) -> list[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(_VECTOR_SQL, np.array(embedding, dtype=np.float32), k)


def _rrf_fuse(rankings: list[tuple[list[str], float]], k: int = RRF_K) -> dict[str, float]:
    """rag/fusion.py의 rrf_fuse()와 동일 공식."""
    scores: dict[str, float] = {}
    for ids, weight in rankings:
        for pos, chunk_id in enumerate(ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + pos + 1)
    return scores


def build_citations(rows: list[asyncpg.Record]) -> list[Citation]:
    """상위 청크에서 인용 근거 역추적 — rag/search.py의 build_citations()와 동일 규칙."""
    seen: set[tuple[Optional[str], Optional[str]]] = set()
    citations: list[Citation] = []
    for row in rows:
        clause_no, exhibit = row["clause_no"], row["exhibit"]
        if clause_no is None and exhibit is None:
            continue
        key = (clause_no, exhibit)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(clause_no=clause_no, exhibit=exhibit))
    return citations


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(_RERANK_MODEL, max_length=512)
    return _reranker


async def search(
    pool: asyncpg.Pool,
    query: str,
    *,
    insurance_type: Optional[str] = None,
    namespaces: Optional[list[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    rerank: Optional[bool] = None,
) -> RagResult:
    """Hybrid RAG 검색. contracts.md §5 시그니처와 호환 (namespaces는 무시 — terms 고정).

    Args:
        rerank: True면 RRF 상위 후보를 bge-reranker-v2-m3로 재정렬(원본 계약 밖 확장).
            미지정(None) 시 SEARCH_RERANK 환경변수를 따름(기본 off — 계약과 동일 동작).
    """
    if rerank is None:
        rerank = _RERANK_DEFAULT
    if _is_out_of_scope(query, insurance_type):
        logger.info("rag.search.out_of_scope query=%r", query[:50])
        return RagResult(ranked_chunks=[], citations=[])

    query_tokens = tokenize_korean(query)
    embedding = embed_query(query)

    keyword_rows = await _keyword_search(pool, query_tokens, _CANDIDATE_K)
    vector_rows = await _vector_search(pool, embedding, _CANDIDATE_K)

    row_index: dict[str, asyncpg.Record] = {}
    for r in (*keyword_rows, *vector_rows):
        row_index.setdefault(r["chunk_id"], r)

    fused = _rrf_fuse([
        ([r["chunk_id"] for r in keyword_rows], KEYWORD_WEIGHT),
        ([r["chunk_id"] for r in vector_rows], VECTOR_WEIGHT),
    ])
    ranked_ids = sorted(fused, key=lambda cid: -fused[cid])

    if rerank and ranked_ids:
        cand = ranked_ids[:_RERANK_TOP]
        pairs = [(query, row_index[cid]["content"][:1500]) for cid in cand]
        ce_scores = await asyncio.to_thread(
            _get_reranker().predict, pairs, batch_size=16, show_progress_bar=False
        )
        order = np.argsort(-np.asarray(ce_scores))
        ranked_ids = [cand[i] for i in order] + ranked_ids[_RERANK_TOP:]
        scores = {cid: float(ce_scores[i]) for i, cid in zip(order, [cand[j] for j in order])}
    else:
        scores = fused

    top_ids = ranked_ids[:top_k]
    top_rows = [row_index[cid] for cid in top_ids]
    chunks = [
        Chunk(text=row["content"], namespace="terms", score=scores[cid], source_ref=cid)
        for cid, row in zip(top_ids, top_rows)
    ]
    return RagResult(ranked_chunks=chunks, citations=build_citations(top_rows))
