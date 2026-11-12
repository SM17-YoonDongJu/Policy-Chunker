"""청크 뷰어 조회 SQL을 진짜 Postgres에서 돌려 본다 — 예전 전체 집계와 같은 답을 내는지.

문서 목록·보험사 목록은 인덱스를 건너뛰는 재귀 CTE로, 문구 찾기는 전문검색 인덱스로 바꿨다
(viewer/queries.py 머리말). 그 SQL은 가짜 DB로는 검증할 수 없어서 여기서 실제로 실행한다.

VIEWER_TEST_PG_ADMIN_DSN(데이터베이스를 만들 수 있는 계정)을 줘야 돈다. 그 서버에 임시
데이터베이스를 새로 만들어 쓰고 끝나면 지운다 — 기존 데이터베이스는 건드리지 않는다.
CI에는 Postgres가 없어 건너뛴다. 로컬:

  docker run -d --rm --name viewer-pg -e POSTGRES_PASSWORD=x -p 55432:5432 postgres:16
  VIEWER_TEST_PG_ADMIN_DSN=postgresql://postgres:x@localhost:55432/postgres \
      pytest tests/test_viewer_queries_pg.py

스키마는 운영 정본(AI 레포 migrations/corpus/001, migrations/ai/002)에서 이 조회들이 쓰는 열과
인덱스만 옮겼다.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ADMIN_DSN = os.environ.get("VIEWER_TEST_PG_ADMIN_DSN")
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="VIEWER_TEST_PG_ADMIN_DSN 없음(로컬 전용)")

SCHEMA = """
CREATE SCHEMA corpus;
CREATE SCHEMA ai;
CREATE TABLE corpus.policy_chunks (
    chunk_id text PRIMARY KEY, content text NOT NULL, content_tokens text, token_count int,
    chunk_type text NOT NULL, doc_hash text NOT NULL, page_number int,
    ingested_at timestamptz DEFAULT now(), insurer text NOT NULL, product_name text NOT NULL,
    product_code text, effective_date date, article_number text, article_title text,
    generation text, section text, chunk_index int, table_id uuid,
    row_start smallint, row_end smallint);
CREATE INDEX idx_policy_doc_hash ON corpus.policy_chunks (doc_hash);
CREATE INDEX idx_policy_meta ON corpus.policy_chunks (insurer, chunk_type, effective_date);
CREATE INDEX idx_policy_fts ON corpus.policy_chunks
    USING gin (to_tsvector('simple', coalesce(content_tokens, '')));
CREATE TABLE ai.corpus_file (
    notion_page_id uuid PRIMARY KEY, product_name text, company text, source_url text,
    part_total smallint NOT NULL DEFAULT 0);
CREATE TABLE ai.corpus_file_part (
    file_page_id uuid NOT NULL REFERENCES ai.corpus_file(notion_page_id),
    part_order smallint NOT NULL, sha256 text, notion_file_name text, s3_key text);
CREATE INDEX corpus_file_part_sha256_idx ON ai.corpus_file_part (sha256)
    WHERE sha256 IS NOT NULL;
"""

A, B, C = ("a" * 64, "b" * 64, "c" * 64)
F1, F2 = uuid.uuid4(), uuid.uuid4()
# 문서 = (doc_hash, 보험사, 상품명, 적재 시각, [(쪽, 유형, 본문)])
DOCS = [
    (A, "KB손해보험", "실손의료보험", datetime(2026, 11, 1, tzinfo=timezone.utc), [
        (1, "coverage", "회사는 골절진단비를 지급합니다."),
        (1, "exclusion", "보험금을 지급하지 않는 사유는 다음과 같습니다."),
        (2, "coverage", "골절 수술비와 진단비는 따로 지급합니다.")]),
    (B, "DB손해보험", "암보험", datetime(2026, 11, 3, tzinfo=timezone.utc), [
        (5, "coverage", "암 진단확정 시 가입금액을 지급합니다."),
        (6, "schedule", "| 구분 | 금액 |\n|---|---|\n| 골절진단비 | 30만원 |")]),
    (C, "KB손해보험", "어린이보험", datetime(2026, 11, 2, tzinfo=timezone.utc), [
        (1, "definition", "이 약관에서 골절이란 뼈가 부러진 상태를 말합니다."),
        (2, "coverage", "골절진단비는 1회에 한하여 지급합니다."),
        (2, "coverage", "깁스치료비를 지급합니다."),
        (3, "general", "부칙")]),
]


@pytest.fixture
def db(monkeypatch):
    import psycopg2
    from psycopg2.extensions import make_dsn

    from insurance_chunker.tokenizer import tokenize_korean
    from viewer import env, queries

    name = f"viewer_test_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(ADMIN_DSN)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name}")
    dsn = make_dsn(ADMIN_DSN, dbname=name)
    try:
        conn = psycopg2.connect(dsn)
        with conn, conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute("INSERT INTO ai.corpus_file VALUES (%s, '실손의료보험', 'KB', NULL, 2),"
                        " (%s, '암보험', 'DB', NULL, 1)", (str(F1), str(F2)))
            cur.execute("INSERT INTO ai.corpus_file_part VALUES "
                        "(%s, 0, %s, '실손약관_part1.pdf', 'k/a.pdf'),"
                        "(%s, 0, %s, '암약관.pdf', 'k/b.pdf')", (str(F1), A, str(F2), B))
            for h, insurer, product, at, chunks in DOCS:
                for i, (page, ctype, content) in enumerate(chunks, 1):
                    cur.execute(
                        "INSERT INTO corpus.policy_chunks (chunk_id, content, content_tokens,"
                        " chunk_type, doc_hash, page_number, ingested_at, insurer, product_name,"
                        " chunk_index) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (f"{h[:4]}-{i}", content, tokenize_korean(content), ctype, h, page, at,
                         insurer, product, i))
        conn.close()
        monkeypatch.setattr(env, "DATABASE_URL", dsn)
        monkeypatch.setattr(env, "RDS_CA_PATH", None)
        monkeypatch.setattr(queries, "_pool", None)
        yield dsn
    finally:
        if queries._pool is not None:
            queries._pool.closeall()
            queries._pool = None
        admin.cursor().execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.close()


def _old_aggregate(dsn: str) -> dict[str, dict]:
    """예전 _DOCS_CTE가 내던 값 — 테이블 전체를 doc_hash로 묶는다(작은 DB라 여기선 괜찮다)."""
    import psycopg2
    import psycopg2.extras

    with psycopg2.connect(dsn) as conn, \
            conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT doc_hash, min(insurer) AS insurer, min(product_name) AS product_name,
                   max(ingested_at) AS ingested_at, count(*) AS chunks,
                   count(DISTINCT page_number) AS pages_with_chunks,
                   max(page_number) AS last_page
            FROM corpus.policy_chunks GROUP BY doc_hash""")
        return {r["doc_hash"]: dict(r) for r in cur.fetchall()}


def test_recent_documents_match_the_old_full_aggregate(db):
    from viewer import queries

    got = queries.recent_documents(limit=10)
    assert [d["doc_hash"] for d in got] == [B, C, A]  # 적재 시각 최신순
    old = _old_aggregate(db)
    for d in got:
        want = old[d["doc_hash"]]
        assert {k: d[k] for k in want} == want
    by = {d["doc_hash"]: d for d in got}
    assert (by[A]["file_name"], by[A]["part_order"], by[A]["part_total"]) == \
        ("실손약관_part1.pdf", 0, 2)
    assert by[C]["file_name"] is None and by[C]["part_total"] is None  # 카탈로그에 없는 문서


def test_recent_documents_filters_and_limit(db):
    from viewer import queries

    def hashes(**kw):
        return [d["doc_hash"] for d in queries.recent_documents(limit=10, **kw)]

    assert hashes(insurer="KB손해보험") == [C, A]
    assert hashes(product="암") == [B]
    assert hashes(name="실손") == [A]
    assert [d["doc_hash"] for d in queries.recent_documents(limit=1)] == [B]
    no_cat = queries.recent_documents(limit=10, with_catalog=False)
    assert [d["doc_hash"] for d in no_cat] == [B, C, A]
    assert all(d["file_name"] is None for d in no_cat)


def test_insurers_are_distinct_and_sorted(db):
    from viewer import queries

    assert queries.insurers() == ["DB손해보험", "KB손해보험"]


def test_empty_table_lists_nothing(db):
    import psycopg2

    from viewer import queries

    with psycopg2.connect(db) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM corpus.policy_chunks")
    assert queries.recent_documents(limit=10) == []
    assert queries.insurers() == []


def test_find_phrase_is_exact_and_newest_first(db):
    from viewer import queries

    hits = queries.find_phrase("골절진단비")
    # 형태소(골절·진단)가 같아도 글자가 그대로 이어지지 않으면 빠진다
    # — A의 2번째 청크 '골절 수술비와 진단비'
    assert [(h["doc_hash"], h["chunk_index"]) for h in hits] == [(B, 2), (C, 2), (A, 1)]
    assert all("골절진단비" in h["content"] for h in hits)
    by_insurer = queries.find_phrase("골절진단비", insurer="KB손해보험")
    assert [h["doc_hash"] for h in by_insurer] == [C, A]
    by_type = queries.find_phrase("골절진단비", chunk_types=["schedule"])
    assert [h["doc_hash"] for h in by_type] == [B]
    assert len(queries.find_phrase("골절진단비", limit=1)) == 1
