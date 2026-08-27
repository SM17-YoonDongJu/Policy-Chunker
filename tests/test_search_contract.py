"""db/search.py 계약 검증 — SM17-YoonDongJu/AI 레포의 test_rag_search.py와 같은
fake-pool 패턴. 실제 Postgres 없이 라우팅→하이브리드검색→RRF→인용 조립을 확인한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db.search as search_mod  # noqa: E402

_KEYWORD_ROWS = [
    {"chunk_id": "t1", "content": "약관 t1 골절진단비", "clause_no": "제3조", "exhibit": "특약A"},
    {"chunk_id": "t2", "content": "약관 t2", "clause_no": "제5조", "exhibit": None},
]
_VECTOR_ROWS = [
    {"chunk_id": "t2", "content": "약관 t2", "clause_no": "제5조", "exhibit": None},
    {"chunk_id": "t3", "content": "약관 t3", "clause_no": "제7조", "exhibit": None},
]


class _FakeConn:
    async def fetch(self, sql: str, *args: object) -> list[dict]:
        return _KEYWORD_ROWS if "plainto_tsquery" in sql else _VECTOR_ROWS


class _FakeAcquire:
    async def __aenter__(self) -> _FakeConn:
        return _FakeConn()

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakePool:
    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire()


@pytest.fixture
def fake_pool() -> _FakePool:
    return _FakePool()


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    # db/search.py는 embed_texts가 아니라 embed_query(단건, instruct 프리픽스 포함)를 쓴다.
    # 이 테스트는 CI에서 한 번도 돌지 않아 그 변경을 놓친 채 남아 있었다.
    monkeypatch.setattr(search_mod, "embed_query", lambda query: [0.0] * 1024)


@pytest.mark.asyncio
async def test_search_returns_fused_chunks_and_citations(fake_pool: _FakePool) -> None:
    result = await search_mod.search(fake_pool, "골절 후유장해 보상", top_k=8)

    assert len(result.ranked_chunks) > 0
    assert all(c.namespace == "terms" for c in result.ranked_chunks)
    clause_numbers = {c.clause_no for c in result.citations}
    assert "제3조" in clause_numbers
    scores = [c.score for c in result.ranked_chunks]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_out_of_scope_returns_empty(fake_pool: _FakePool) -> None:
    result = await search_mod.search(fake_pool, "자동차 사고 보상")
    assert result.ranked_chunks == []
    assert result.citations == []


@pytest.mark.asyncio
async def test_out_of_scope_via_insurance_type_hint(fake_pool: _FakePool) -> None:
    result = await search_mod.search(fake_pool, "다쳤는데 보상되나요", insurance_type="화재보험")
    assert result.ranked_chunks == []


@pytest.mark.asyncio
async def test_rerank_true_returns_same_shape(
    fake_pool: _FakePool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeReranker:
        def predict(self, pairs, **kw):
            return np.array([0.9 - i * 0.1 for i in range(len(pairs))])

    monkeypatch.setattr(search_mod, "_get_reranker", lambda: _FakeReranker())
    result = await search_mod.search(fake_pool, "골절 진단비", top_k=3, rerank=True)
    assert len(result.ranked_chunks) <= 3
    assert all(isinstance(c.score, float) for c in result.ranked_chunks)
