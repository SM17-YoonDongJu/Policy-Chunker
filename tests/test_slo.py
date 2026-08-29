"""인덱스 SLO 판정 — "적재됐다"와 "쓸 만하다"는 다르다.

적재 자체는 성공했는데 인덱스가 못 쓰는 상태인 경우들이 있다. 임베딩 차원이 어긋나거나,
파트너 레포가 읽는 컬럼이 없거나, 문서 하나에 임베딩이 통째로 빠지거나. 사이클 성패로는
안 잡히는 것들이라 여기서 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import slo  # noqa: E402


class _FakeCursor:
    def __init__(self, answers):
        self._answers, self._row = answers, None

    def execute(self, sql, params=()):
        for pattern, value in self._answers.items():
            if pattern in " ".join(sql.split()):
                if isinstance(value, Exception):
                    raise value
                self._row = (value,)
                return
        self._row = None

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, **answers):
        self._answers = answers.pop("answers", {})
        self.rolled_back = False

    def cursor(self):
        return _FakeCursor(self._answers)

    def rollback(self):
        self.rolled_back = True


def _conn(**answers):
    return _FakeConn(answers=answers)


def _by_name(report, name):
    return next(c for c in report.checks if c.name == name)


# ── 개별 판정 ─────────────────────────────────────────────────────────────────

def test_dimension_mismatch_is_fatal(monkeypatch):
    """모델이 바뀌면 벡터 검색이 통째로 어긋난다 — 재적재 말고 답이 없다."""
    monkeypatch.setenv("EMBED_DIM", "1024")
    c = slo._check_embedding_dim(_conn(**{"vector_dims": 768}))
    assert c.status == slo.FAIL
    assert "재적재" in c.detail


def test_missing_search_column_is_fatal():
    """파트너 레포가 직접 @@ 검색하는 컬럼이다. 없으면 그쪽 결과가 조용히 0건이 된다."""
    c = slo._check_search_contract(_conn(**{"information_schema.columns": 0}))
    assert c.status == slo.FAIL


def test_document_with_no_embeddings_is_fatal():
    """전체 NULL 비율로는 못 잡는다 — boilerplate는 일부러 비우는데 DB에 그 표시가 없다.
    반면 '한 문서에 임베딩 0개'는 명백한 고장이다."""
    c = slo._check_documents_embedded(_conn(**{"HAVING count(embedding) = 0": 3}))
    assert c.status == slo.FAIL
    assert c.value == 3


def test_all_documents_embedded_is_ok():
    c = slo._check_documents_embedded(_conn(**{"HAVING count(embedding) = 0": 0}))
    assert c.status == slo.OK


def test_stale_index_is_a_warning_not_a_failure(monkeypatch):
    """사이클이 전부 SKIPPED였다면 DB는 그대로다 — 오래됐다고 바로 고장은 아니다."""
    monkeypatch.setenv("SLO_MAX_STALE_DAYS", "10")
    c = slo._check_freshness(_conn(**{"now() - max(ingested_at)": 30.0}), 604800)
    assert c.status == slo.WARN


def test_low_coverage_warns(monkeypatch):
    monkeypatch.setenv("SLO_MIN_COVERAGE", "0.8")
    c = slo._check_catalog_coverage(
        _conn(**{"ai.corpus_file": 100, "count(DISTINCT doc_hash)": 50}))
    assert c.status == slo.WARN
    assert c.value == 0.5


def test_catalog_permission_error_is_skipped_not_failed(monkeypatch):
    """ai 스키마 권한은 AI 레포 마이그레이션 소관이다 — 우리 쪽 실패로 볼 일이 아니다."""
    import psycopg2
    monkeypatch.setenv("SLO_MIN_COVERAGE", "0.8")
    c = slo._check_catalog_coverage(
        _conn(**{"ai.corpus_file": psycopg2.errors.InsufficientPrivilege()}))
    assert c.status == slo.SKIP


def test_coverage_check_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SLO_MIN_COVERAGE", "0")
    assert slo._check_catalog_coverage(_conn()).status == slo.SKIP


# ── 리포트 ────────────────────────────────────────────────────────────────────

def test_one_broken_check_does_not_block_the_rest(monkeypatch):
    """점검 하나가 터져도 나머지는 봐야 한다."""
    monkeypatch.setenv("EMBED_DIM", "1024")
    report = slo.evaluate(_conn(**{
        "vector_dims": RuntimeError("boom"),
        "information_schema.columns": 1,
        "HAVING count(embedding) = 0": 0,
    }), 604800)
    assert _by_name(report, "embedding_dim").status == slo.SKIP
    assert _by_name(report, "search_contract").status == slo.OK


def test_worst_status_prefers_fail_over_warn():
    report = slo.Report([slo.Check("a", slo.WARN, ""), slo.Check("b", slo.FAIL, "")])
    assert report.worst == slo.FAIL
    assert len(report.violations) == 2


def test_skip_is_not_a_violation():
    """권한 없음 같은 건 위반이 아니다 — 알림이 시끄러워진다."""
    report = slo.Report([slo.Check("a", slo.OK, ""), slo.Check("b", slo.SKIP, "")])
    assert report.violations == []
    assert report.worst == slo.OK
