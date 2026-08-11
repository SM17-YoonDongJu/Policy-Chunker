#!/usr/bin/env python3
"""search_terms 테이블 재구성 CLI.

policy_chunks.content_tokens에서 형태소 term을 추출해 search_terms에 적재한다.
ingest.py와 독립 실행 — 모든 문서 ingest 완료 후 한 번 돌리면 된다.

사용:
  python rebuild_search_terms.py --db-url postgresql://user:pass@host/db
  python rebuild_search_terms.py  # DATABASE_URL 환경변수 사용
"""
from __future__ import annotations

import argparse
import logging
import sys

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="search_terms 테이블 재구성")
    p.add_argument("--db-url", default=None, help="PostgreSQL 연결 URL (없으면 DATABASE_URL 환경변수)")
    p.add_argument("--no-init-schema", action="store_true",
                   help="스키마 DDL 실행 안 함 (운영은 AI 레포 migrations/corpus가 단일 관리)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from db.storage import get_connection, init_schema
    from db.search_term import rebuild

    logger.info("DB 연결 중...")
    conn = get_connection(args.db_url)
    init_schema(conn, skip=args.no_init_schema)

    logger.info("search_terms 재구성 시작")
    result = rebuild(conn)
    conn.close()

    if result.get("skipped"):
        logger.info("건너뜀 — 다른 프로세스가 이미 실행 중")
    else:
        logger.info(
            f"완료 — 추출 {result['extracted']}개 | "
            f"신규 {result['inserted']}개 | 누계 {result['total']}개"
        )


if __name__ == "__main__":
    main()
