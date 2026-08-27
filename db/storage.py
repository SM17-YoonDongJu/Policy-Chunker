"""pgvector 저장."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from insurance_chunker.models import InsuranceChunk

logger = logging.getLogger(__name__)
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_url: Optional[str] = None) -> psycopg2.extensions.connection:
    url = db_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DB 연결 정보 없음. --db-url 또는 DATABASE_URL 환경변수 설정 필요.")
    conn = psycopg2.connect(url)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg2.extensions.connection, skip: bool = False) -> None:
    """schema.sql을 적용한다. skip=True 또는 SKIP_INIT_SCHEMA=1이면 건너뛴다.

    운영 RDS의 corpus.* 테이블은 SM17-YoonDongJu/AI 레포 migrations/corpus/가 단일 관리한다.
    같은 테이블의 DDL 진실원이 둘이면 한쪽만 바뀔 때 드리프트가 생기므로(실제로 search_terms는
    이미 컬럼이 다르다 — 우리 1컬럼 vs corpus 3컬럼) 운영에서는 .env.prod에
    SKIP_INIT_SCHEMA=1로 끈다.
    로컬 빈 DB에서는 자동 생성이 편하므로 기본값은 실행(opt-out)이다.
    """
    if skip or os.environ.get("SKIP_INIT_SCHEMA", "").strip().lower() in ("1", "true", "yes"):
        logger.info("스키마 DDL 건너뜀 — DDL은 AI 레포 migrations/corpus가 단일 관리")
        return
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("스키마 초기화 완료")


def doc_already_ingested(conn: psycopg2.extensions.connection, doc_hash: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM policy_chunks WHERE doc_hash = %s LIMIT 1", (doc_hash,))
        return cur.fetchone() is not None


def delete_by_doc_hash(conn: psycopg2.extensions.connection, doc_hash: str) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM policy_chunks WHERE doc_hash = %s", (doc_hash,))
        deleted = cur.rowcount
    conn.commit()
    logger.info(f"기존 청크 {deleted}개 삭제")
    return deleted


def _row(c: InsuranceChunk) -> tuple:
    """INSERT 컬럼 순서와 1:1로 대응하는 값 튜플."""
    return (
        c.chunk_id,
        c.content,
        c.content_tokens,
        np.array(c.embedding, dtype=np.float32) if c.embedding else None,
        c.token_count,
        c.chunk_type,
        c.doc_hash,
        c.page_number,
        c.insurer,
        c.product_name,
        c.product_code,
        c.effective_date,
        c.article_number,
        c.article_title,
        c.generation,
        c.section,
        c.chunk_index,
        c.table_id,
        c.row_start,
        c.row_end,
        c.product_id,
    )


def _dedupe(chunks: list[InsuranceChunk]) -> list[InsuranceChunk]:
    """같은 chunk_id를 last-wins로 접는다.

    execute_values는 한 배치를 INSERT ... VALUES (..),(..) 한 문장으로 묶는다. 그 안에 같은
    chunk_id가 두 번 있으면 Postgres가 "ON CONFLICT DO UPDATE command cannot affect row a
    second time"로 죽는다. 건별로 실행하던 executemany는 뒤엣것이 앞엣것을 UPDATE하며 조용히
    넘어갔으므로, 그 동작을 유지하려고 미리 접는다(정상 문서에선 중복이 안 나온다 — 나오면
    chunk_id 생성 규칙이 깨진 것이라 경고로 드러낸다).
    """
    uniq: dict[str, InsuranceChunk] = {}
    for c in chunks:
        uniq[c.chunk_id] = c
    if len(uniq) != len(chunks):
        logger.warning(f"chunk_id 중복 {len(chunks) - len(uniq)}건 — 마지막 값만 저장한다")
    return list(uniq.values())


def upsert_chunks(
    conn: psycopg2.extensions.connection,
    chunks: list[InsuranceChunk],
    batch_size: int = 200,
) -> None:
    """청크를 배치 INSERT한다.

    execute_values를 쓴다. psycopg2의 executemany는 배치로 잘라도 파라미터 세트마다 statement를
    개별 실행해 청크 수만큼 왕복하는데, RDS가 원격이라 RTT가 그대로 곱해졌다(21,791청크 = 21,791회).
    execute_values는 배치를 한 문장으로 묶어 배치당 1회 왕복이 된다.
    """
    sql = """
        INSERT INTO policy_chunks (
            chunk_id,
            content,
            content_tokens,
            embedding,
            token_count,
            chunk_type,
            doc_hash,
            page_number,
            insurer,
            product_name,
            product_code,
            effective_date,
            article_number,
            article_title,
            generation,
            section,
            chunk_index,
            table_id,
            row_start,
            row_end,
            product_id
        ) VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
            content        = EXCLUDED.content,
            content_tokens = EXCLUDED.content_tokens,
            embedding      = EXCLUDED.embedding,
            token_count    = EXCLUDED.token_count,
            chunk_type     = EXCLUDED.chunk_type,
            article_number = EXCLUDED.article_number,
            article_title  = EXCLUDED.article_title,
            generation     = EXCLUDED.generation,
            section        = EXCLUDED.section,
            chunk_index    = EXCLUDED.chunk_index,
            table_id       = EXCLUDED.table_id,
            row_start      = EXCLUDED.row_start,
            row_end        = EXCLUDED.row_end,
            product_id     = EXCLUDED.product_id
    """
    chunks = _dedupe(chunks)
    total = len(chunks)
    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            # page_size를 배치 크기로 맞춰 배치 하나가 정확히 왕복 1회가 되게 한다
            # (기본값 100이면 execute_values가 안에서 또 쪼갠다).
            execute_values(cur, sql, [_row(c) for c in batch], page_size=len(batch))
            logger.info(f"  저장 {min(i+batch_size, total)}/{total}")
    conn.commit()
    logger.info(f"총 {total}개 청크 저장 완료")


def verify_upsert(conn: psycopg2.extensions.connection, doc_hash: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)                                    AS total,
                COUNT(embedding)                            AS with_embedding,
                COUNT(*) FILTER (WHERE embedding IS NULL)  AS without_embedding
            FROM policy_chunks
            WHERE doc_hash = %s
            """,
            (doc_hash,),
        )
        row = cur.fetchone()
        cur.execute(
            "SELECT chunk_type, COUNT(*) FROM policy_chunks "
            "WHERE doc_hash = %s GROUP BY chunk_type",
            (doc_hash,),
        )
        type_rows = cur.fetchall()
    result = {
        "total": row[0],
        "with_embedding": row[1],
        "without_embedding": row[2],
        "chunk_type_counts": {t: n for t, n in type_rows},
    }
    logger.info(f"[DB 검증] 총 {result['total']}개 | 임베딩 {result['with_embedding']}개")
    return result
