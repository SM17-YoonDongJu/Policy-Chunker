"""pgvector 저장."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

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
    이미 컬럼이 다르다 — 우리 1컬럼 vs corpus 3컬럼) 운영에서는 .env.prod에 SKIP_INIT_SCHEMA=1로 끈다.
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


def upsert_chunks(
    conn: psycopg2.extensions.connection,
    chunks: list[InsuranceChunk],
    batch_size: int = 200,
) -> None:
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
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
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
    total = len(chunks)
    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            rows = [
                (
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
                for c in batch
            ]
            cur.executemany(sql, rows)
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
