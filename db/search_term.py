"""search_terms 테이블 적재 로직.

policy_chunks.content_tokens (Kiwi 형태소 토큰)에서 고유 term을 추출해
search_terms 테이블에 upsert한다.

멀티 머신 동시 ingest 시 race condition 방지:
  - ON CONFLICT DO NOTHING → 같은 term 중복 삽입 무시
  - search_terms는 policy_chunks에서 언제든 재생성 가능한 파생 테이블이므로
    인라인(ingest 중) 갱신이 아닌 별도 rebuild 스크립트로 운영
"""
from __future__ import annotations

import logging
from typing import Optional

import psycopg2

logger = logging.getLogger(__name__)

_BATCH_SIZE = 2000


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
    """search_terms 전체 재구성. 반환값: 통계 dict."""
    terms = extract_terms(conn)
    inserted = upsert_terms(conn, terms)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM search_terms")
        total = cur.fetchone()[0]

    result = {"extracted": len(terms), "inserted": inserted, "total": total}
    logger.info(
        f"[search_terms] 추출 {result['extracted']}개 | "
        f"신규 {result['inserted']}개 | 누계 {result['total']}개"
    )
    return result
