"""search_terms 테이블 적재 로직.

policy_chunks.content_tokens (Kiwi 형태소 토큰)에서 고유 term을 추출해
search_terms 테이블에 upsert한다.

동시 실행 방지:
  - pg_try_advisory_lock으로 non-blocking lock 획득
  - 다른 프로세스가 이미 실행 중이면 즉시 건너뜀
  - ON CONFLICT DO NOTHING → 같은 term 중복 삽입 무시
"""
from __future__ import annotations

import logging

import psycopg2

logger = logging.getLogger(__name__)

_BATCH_SIZE = 2000
_LOCK_KEY = 202606011  # search_terms rebuild 전용 advisory lock 키


def extract_terms(conn: psycopg2.extensions.connection) -> list[str]:
    """policy_chunks.content_tokens 전체에서 고유 형태소 term 목록 반환."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT unnest(string_to_array(content_tokens, ' ')) AS term
            FROM policy_chunks
            WHERE content_tokens IS NOT NULL AND content_tokens <> ''
            ORDER BY term
        """)
        rows = cur.fetchall()
    terms = [r[0] for r in rows if r[0] and len(r[0]) > 1]
    logger.info(f"추출된 고유 term: {len(terms)}개")
    return terms


def upsert_terms(
    conn: psycopg2.extensions.connection,
    terms: list[str],
) -> int:
    """term 목록을 search_terms에 upsert. 반환값: 신규 삽입 수."""
    sql = "INSERT INTO search_terms (term) VALUES (%s) ON CONFLICT (term) DO NOTHING"
    inserted = 0
    with conn.cursor() as cur:
        for i in range(0, len(terms), _BATCH_SIZE):
            batch = [(t,) for t in terms[i:i + _BATCH_SIZE]]
            cur.executemany(sql, batch)
            inserted += cur.rowcount
            logger.info(f"  upsert {min(i + _BATCH_SIZE, len(terms))}/{len(terms)}")
    conn.commit()
    return inserted


def rebuild(conn: psycopg2.extensions.connection) -> dict:
    """search_terms 전체 재구성. 반환값: 통계 dict.

    pg_try_advisory_lock으로 non-blocking lock을 획득한다.
    다른 프로세스가 이미 실행 중이면 즉시 건너뛰고 skipped=True를 반환한다.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_KEY,))
        acquired = cur.fetchone()[0]

    if not acquired:
        logger.info("search_terms rebuild: 다른 프로세스가 실행 중 — 건너뜀")
        return {"extracted": 0, "inserted": 0, "total": 0, "skipped": True}

    try:
        terms = extract_terms(conn)
        inserted = upsert_terms(conn, terms)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM search_terms")
            total = cur.fetchone()[0]

        result = {"extracted": len(terms), "inserted": inserted, "total": total, "skipped": False}
        logger.info(
            f"[search_terms] 추출 {result['extracted']}개 | "
            f"신규 {result['inserted']}개 | 누계 {result['total']}개"
        )
        return result
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
        logger.debug("advisory lock 해제")
