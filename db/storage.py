"""pgvector 저장.
출처: rag/db/storage.py — import 경로 수정.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from ..insurance_chunker.models import InsuranceChunk

logger = logging.getLogger(__name__)
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_url: Optional[str] = None) -> psycopg2.extensions.connection:
    url = db_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DB 연결 정보 없음. --db-url 또는 DATABASE_URL 환경변수 설정 필요.")
    conn = psycopg2.connect(url)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg2.extensions.connection) -> None:
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("스키마 초기화 완료")


def doc_already_ingested(conn: psycopg2.extensions.connection, doc_hash: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM insurance_chunks WHERE doc_hash = %s LIMIT 1", (doc_hash,))
        return cur.fetchone() is not None


def delete_by_doc_hash(conn: psycopg2.extensions.connection, doc_hash: str) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM insurance_chunks WHERE doc_hash = %s", (doc_hash,))
        deleted = cur.rowcount
    conn.commit()
    logger.info(f"기존 청크 {deleted}개 삭제")
    return deleted


def upsert_chunks(conn: psycopg2.extensions.connection,
                  chunks: list[InsuranceChunk], batch_size: int = 200) -> None:
    sql = """
        INSERT INTO insurance_chunks (
            chunk_id, parent_id, source_pdf, doc_hash,
            insurer, product_name, product_code, effective_date,
            doc_type, chunk_type,
            section_path, page_number, token_count,
            article_number, article_title, yakwan,
            content, content_tokens, structured_json, embedding
        ) VALUES (
            %s, %s, %s, %s,  %s, %s, %s, %s,
            %s, %s,  %s, %s, %s,
            %s, %s, %s,  %s, %s, %s, %s
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            content         = EXCLUDED.content,
            content_tokens  = EXCLUDED.content_tokens,
            structured_json = EXCLUDED.structured_json,
            embedding       = EXCLUDED.embedding,
            token_count     = EXCLUDED.token_count,
            section_path    = EXCLUDED.section_path,
            article_number  = EXCLUDED.article_number,
            article_title   = EXCLUDED.article_title,
            yakwan          = EXCLUDED.yakwan
    """
    total = len(chunks)
    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            rows = [(
                c.chunk_id, c.parent_id, c.source_pdf, c.doc_hash,
                c.insurer, c.product_name, c.product_code, c.effective_date,
                c.doc_type, c.chunk_type,
                c.section_path, c.page_number, c.token_count,
                c.article_number, c.article_title, c.yakwan,
                c.content, c.content_tokens,
                json.dumps(c.structured_json, ensure_ascii=False) if c.structured_json else None,
                np.array(c.embedding, dtype=np.float32) if c.embedding else None,
            ) for c in batch]
            cur.executemany(sql, rows)
            logger.info(f"  저장 {min(i+batch_size, total)}/{total}")
    conn.commit()
    logger.info(f"총 {total}개 청크 저장 완료")


def verify_upsert(conn: psycopg2.extensions.connection, doc_hash: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*), COUNT(embedding), COUNT(*) FILTER (WHERE embedding IS NULL) "
            "FROM insurance_chunks WHERE doc_hash = %s",
            (doc_hash,),
        )
        row = cur.fetchone()
        cur.execute(
            "SELECT chunk_type, COUNT(*) FROM insurance_chunks "
            "WHERE doc_hash = %s GROUP BY chunk_type",
            (doc_hash,),
        )
        type_rows = cur.fetchall()
    result = {
        "total": row[0], "with_embedding": row[1], "without_embedding": row[2],
        "chunk_type_counts": {t: n for t, n in type_rows},
    }
    logger.info(f"[DB 검증] 총 {result['total']}개 | 임베딩 {result['with_embedding']}개")
    return result
