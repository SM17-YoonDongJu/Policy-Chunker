"""DB 조회 — 전부 읽기 전용이다.

연결 자체를 읽기 전용 트랜잭션으로 연다(default_transaction_read_only). 뷰어 전용 읽기 롤을
쓰는 것이 원칙이지만, 임시로 쓰기 권한이 있는 계정을 물려도 실수로 쓰지 못하게 하려는 것이다.

테이블은 스키마를 붙여 쓴다(corpus.·ai.) — 운영 롤마다 search_path가 달라서다.
원본 파일 정보(ai.corpus_file*)는 권한이 없을 수 있다. 그때는 파일명·PDF 없이 청크만 보여 준다.

화면을 그릴 때마다 도는 조회는 청크 테이블 전체를 읽지 않는다. dev RDS(db.t4g.small, 2GiB)에서
policy_chunks가 약 227만 행·13GB가 되자(2026-11-11), 전체를 doc_hash로 묶던 문서 목록이
statement_timeout(30초)에 걸려 문서 보기·처리 현황이 열리지 않았다. 그래서 인덱스만 탄다.
  - 목록에 필요한 값(적재 시각·보험사·상품명)은 문서 안에서 모두 같다. 한 문서는 한 트랜잭션으로
    지우고 다시 넣고(db/storage.upsert_chunks), ingested_at 기본값 now()는 트랜잭션 시작 시각이다.
    그래서 문서마다 첫 행 하나만 읽는다
  - 문서·보험사 목록은 인덱스를 건너뛰며 뽑는다(loose index scan) — 서로 다른 값 수만큼만 읽는다
  - 청크 수 같은 문서별 집계는 화면에 보일 문서만 센다
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

from insurance_chunker.tokenizer import tokenize_korean
from insurance_chunker.validator import VALID_CHUNK_TYPES
from viewer import env

_STATEMENT_TIMEOUT_MS = 30_000
_POOL_MAX = 8
_NO_TIME = datetime.min.replace(tzinfo=timezone.utc)  # 적재 시각이 없는 행은 맨 뒤로

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_lock = threading.Lock()


def connect_kwargs() -> dict[str, Any]:
    kw: dict[str, Any] = {
        "options": "-c default_transaction_read_only=on "
                   f"-c statement_timeout={_STATEMENT_TIMEOUT_MS}",
        "application_name": "chunk-viewer",
    }
    if env.RDS_CA_PATH:
        kw.update(sslmode="verify-full", sslrootcert=env.RDS_CA_PATH)
    return kw


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    with _lock:
        if _pool is None:
            if not env.DATABASE_URL:
                raise RuntimeError("DATABASE_URL이 없어요 — 뷰어가 읽을 DB를 지정하세요.")
            _pool = psycopg2.pool.ThreadedConnectionPool(1, _POOL_MAX, env.DATABASE_URL,
                                                         **connect_kwargs())
        return _pool


@contextmanager
def _cursor() -> Iterator[Any]:
    pool = _get_pool()
    conn = pool.getconn()
    broken = False
    try:
        conn.autocommit = True
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        broken = True  # 끊긴 연결(RDS 재시작·유휴 종료)은 풀에 돌려놓지 않는다
        raise
    finally:
        pool.putconn(conn, close=broken or bool(conn.closed))


def _rows(sql: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def like_pattern(text: str) -> str:
    """사용자 입력을 ILIKE 부분 일치 패턴으로 — %·_를 글자 그대로 찾게 이스케이프한다."""
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def catalog_access() -> bool:
    """원본 파일 카탈로그(ai.corpus_file*)를 읽을 수 있나.

    스키마·테이블이 없거나 권한이 없으면 False.
    """
    try:
        rows = _rows("""
            SELECT has_schema_privilege('ai', 'USAGE')
               AND has_table_privilege('ai.corpus_file_part', 'SELECT')
               AND has_table_privilege('ai.corpus_file', 'SELECT') AS ok
        """)
    except psycopg2.Error:
        return False
    return bool(rows and rows[0]["ok"])


# 문서 = doc_hash(원본 파일의 sha256) 하나. hashes는 doc_hash 인덱스를 건너뛰며 서로 다른 값만
# 꺼낸다(재귀 CTE — 문서 수만큼 인덱스를 내려간다). 적재 시각·보험사·상품명은 문서 안에서 같으므로
# 문서마다 첫 행 하나만 읽는다(모듈 머리말). 재귀가 끝날 때 남는 NULL 한 줄은 LATERAL에서 빠진다.
_DOCS_CTE = """
WITH RECURSIVE hashes AS (
    (SELECT doc_hash FROM corpus.policy_chunks ORDER BY doc_hash LIMIT 1)
    UNION ALL
    SELECT (SELECT c.doc_hash FROM corpus.policy_chunks c
            WHERE c.doc_hash > h.doc_hash ORDER BY c.doc_hash LIMIT 1)
    FROM hashes h
    WHERE h.doc_hash IS NOT NULL
),
docs AS (
    SELECT h.doc_hash, r.insurer, r.product_name, r.ingested_at
    FROM hashes h
    CROSS JOIN LATERAL (
        SELECT c.insurer, c.product_name, c.ingested_at
        FROM corpus.policy_chunks c
        WHERE c.doc_hash = h.doc_hash
        LIMIT 1
    ) r
    WHERE (%(insurer)s::text IS NULL OR r.insurer = %(insurer)s)
      AND (%(product)s::text IS NULL OR r.product_name ILIKE %(product)s)
)
"""

# 청크 수·쪽 수는 화면에 보일 문서(picked)만 doc_hash 인덱스로 센다.
_DOC_STATS = """
SELECT pk.*, s.chunks, s.pages_with_chunks, s.last_page
FROM picked pk
CROSS JOIN LATERAL (
    SELECT count(*) AS chunks,
           count(DISTINCT c.page_number) AS pages_with_chunks,
           max(c.page_number) AS last_page
    FROM corpus.policy_chunks c
    WHERE c.doc_hash = pk.doc_hash
) s
ORDER BY pk.ingested_at DESC NULLS LAST
"""

_RECENT_WITH_CATALOG = _DOCS_CTE + """,
picked AS (
    SELECT d.*, p.notion_file_name AS file_name, p.part_order, f.part_total
    FROM docs d
    LEFT JOIN LATERAL (
        SELECT file_page_id, notion_file_name, part_order
        FROM ai.corpus_file_part
        WHERE sha256 = d.doc_hash
        ORDER BY part_order
        LIMIT 1
    ) p ON true
    LEFT JOIN ai.corpus_file f ON f.notion_page_id = p.file_page_id
    WHERE (%(name)s::text IS NULL OR p.notion_file_name ILIKE %(name)s)
    ORDER BY d.ingested_at DESC NULLS LAST
    LIMIT %(limit)s
)
""" + _DOC_STATS

_RECENT_WITHOUT_CATALOG = _DOCS_CTE + """,
picked AS (
    SELECT d.*, NULL::text AS file_name, NULL::smallint AS part_order,
           NULL::smallint AS part_total
    FROM docs d
    ORDER BY d.ingested_at DESC NULLS LAST
    LIMIT %(limit)s
)
""" + _DOC_STATS


def recent_documents(*, limit: int = 100, insurer: Optional[str] = None,
                     product: Optional[str] = None, name: Optional[str] = None,
                     with_catalog: bool = True) -> list[dict[str, Any]]:
    """적재된 문서를 마지막 적재 시각 최신순으로.

    파일명 필터는 카탈로그를 읽을 수 있을 때만 된다.
    """
    params = {"limit": limit, "insurer": insurer or None,
              "product": like_pattern(product) if product else None,
              "name": like_pattern(name) if name else None}
    sql = _RECENT_WITH_CATALOG if with_catalog else _RECENT_WITHOUT_CATALOG
    return _rows(sql, params)


def document_meta(doc_hash: str) -> Optional[dict[str, Any]]:
    rows = _rows("""
        SELECT doc_hash, min(insurer) AS insurer, min(product_name) AS product_name,
               min(product_code) AS product_code, min(effective_date) AS effective_date,
               min(generation) AS generation, max(ingested_at) AS ingested_at,
               count(*) AS chunks, max(page_number) AS last_page
        FROM corpus.policy_chunks
        WHERE doc_hash = %(h)s
        GROUP BY doc_hash
    """, {"h": doc_hash})
    return rows[0] if rows else None


def document_chunks(doc_hash: str) -> list[dict[str, Any]]:
    """한 문서의 청크 전부를 문서 순서(chunk_index)대로."""
    return _rows("""
        SELECT chunk_id, chunk_index, page_number, chunk_type, section, article_number,
               article_title, token_count, content, table_id::text AS table_id,
               row_start, row_end
        FROM corpus.policy_chunks
        WHERE doc_hash = %(h)s
        ORDER BY chunk_index NULLS LAST, page_number NULLS LAST, chunk_id
    """, {"h": doc_hash})


def source_file(doc_hash: str) -> Optional[dict[str, Any]]:
    """원본 파일 — S3 키와 Notion 파일명.

    카탈로그 sha256이 곧 doc_hash다(ingest_catalog의 1차 중복 제거가 같은 가정을 쓴다).
    """
    rows = _rows("""
        SELECT p.s3_key, p.notion_file_name AS file_name, p.part_order, f.part_total,
               f.company, f.product_name, f.source_url, f.notion_page_id::text AS notion_page_id
        FROM ai.corpus_file_part p
        JOIN ai.corpus_file f ON f.notion_page_id = p.file_page_id
        WHERE p.sha256 = %(h)s
        ORDER BY p.part_order
        LIMIT 1
    """, {"h": doc_hash})
    return rows[0] if rows else None


def insurers() -> list[str]:
    """보험사 목록. (insurer, chunk_type, effective_date) 인덱스(idx_policy_meta)를 건너뛰며
    서로 다른 값만 읽는다 — DISTINCT는 인덱스 전체(청크 수만큼)를 훑는다."""
    return [r["insurer"] for r in _rows("""
        WITH RECURSIVE ins AS (
            (SELECT insurer FROM corpus.policy_chunks ORDER BY insurer LIMIT 1)
            UNION ALL
            SELECT (SELECT c.insurer FROM corpus.policy_chunks c
                    WHERE c.insurer > i.insurer ORDER BY c.insurer LIMIT 1)
            FROM ins i
            WHERE i.insurer IS NOT NULL
        )
        SELECT insurer FROM ins WHERE insurer IS NOT NULL ORDER BY insurer
    """)]


def chunk_types() -> list[str]:
    """청크 유형 목록. 청커가 쓰는 값은 validator가 정한 집합이다 — chunk_type으로 시작하는
    인덱스가 없어 DB에서 DISTINCT를 뽑으면 테이블 전체를 읽는다."""
    return sorted(VALID_CHUNK_TYPES)


class PhraseTooShort(ValueError):
    """형태소 인덱스로 좁힐 단어가 없는 문구. 받으면 본문 전체를 훑게 되므로 거절한다."""


def find_phrase(text: str, *, insurer: Optional[str] = None,
                chunk_types: Optional[list[str]] = None, limit: int = 50) -> list[dict[str, Any]]:
    """청크 본문에서 문구를 글자 그대로 찾는다. 찾은 것 안에서 최근 적재 문서부터.

    본문 ILIKE만으로는 13GB 본문을 전부 읽는다. 그래서 운영 키워드 검색과 같은 전문검색
    인덱스(to_tsvector('simple', content_tokens) — AI 마이그레이션 corpus/001, rag.search와 같은
    식)로 먼저 좁히고, 그 안에서 글자 그대로 맞는 것만 남긴다. content_tokens는 청커가
    tokenize_korean으로 만든 값이라 문구도 같은 함수로 자른다. 형태소는 문맥을 타므로, 단어
    중간에서 자른 문구는 못 찾을 수 있다.

    정렬은 SQL에서 하지 않는다 — 전부 정렬하려면 맞는 것을 전부 읽어야 한다. 상한만큼
    찾으면 멈추고, 찾은 것만 최근 적재 순으로 놓는다.
    """
    tokens = tokenize_korean(text).strip()
    if not tokens:
        raise PhraseTooShort("형태소로 좁힐 단어가 없어요 — 두 글자 이상인 낱말을 넣어 주세요.")
    rows = _rows("""
        SELECT chunk_id, doc_hash, page_number, chunk_index, chunk_type, article_number,
               article_title, insurer, product_name, content, ingested_at
        FROM corpus.policy_chunks
        WHERE to_tsvector('simple', coalesce(content_tokens, ''))
              @@ plainto_tsquery('simple', %(tokens)s)
          AND content ILIKE %(pat)s ESCAPE '\\'
          AND (%(insurer)s::text IS NULL OR insurer = %(insurer)s)
          AND (%(types)s::text[] IS NULL OR chunk_type = ANY(%(types)s))
        LIMIT %(limit)s
    """, {"tokens": tokens, "pat": like_pattern(text), "insurer": insurer or None,
          "types": list(chunk_types) if chunk_types else None, "limit": limit})
    # 같은 문서는 문서 순서대로, 문서끼리는 최근 적재부터(정렬이 안정적이라 두 번 나눠 건다)
    rows.sort(key=lambda r: (r["doc_hash"], r["chunk_index"] is None, r["chunk_index"] or 0))
    rows.sort(key=lambda r: r["ingested_at"] or _NO_TIME, reverse=True)
    return rows


def chunk_locations(chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    """chunk_id → 문서·쪽. 운영 검색 결과(source_ref = chunk_id)를 문서 보기로 잇는 데 쓴다."""
    if not chunk_ids:
        return {}
    rows = _rows("""
        SELECT chunk_id, doc_hash, page_number, chunk_index, insurer, product_name
        FROM corpus.policy_chunks
        WHERE chunk_id = ANY(%(ids)s)
    """, {"ids": list(chunk_ids)})
    return {r["chunk_id"]: r for r in rows}
