#!/usr/bin/env python3
"""PDF 보험 문서 → pgvector 저장 파이프라인 CLI.

사용 예시:
  python ingest.py --pdf 약관.pdf --insurer 메리츠화재 --product "무배당 The건강한보험"
  python ingest.py --pdf 약관.pdf --insurer 메리츠화재 --product "..." --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import logging_setup

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

logging_setup.configure()
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF 보험 문서 파싱·청킹·임베딩·저장")
    p.add_argument("--pdf", required=True)
    p.add_argument("--doc-type", default=None,
                   choices=["product_summary", "policy_terms", "schedule"])
    p.add_argument("--insurer", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--product-code", default=None)
    p.add_argument("--effective-date", default=None, help="시행일 YYYY-MM-DD")
    p.add_argument("--generation", default=None, help="세대 (예: '4세대')")
    p.add_argument("--product-id", default=None, help="insurance_products.id (UUID)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--no-embed", action="store_true", help="임베딩 건너뜀 (청킹 테스트 시 사용)")
    p.add_argument("--no-init-schema", action="store_true",
                   help="스키마 DDL 실행 안 함 (운영은 AI 레포 migrations/corpus가 단일 관리)")
    p.add_argument("--dry-run", action="store_true",
                   help="DB 저장 없이 청킹 결과 JSON 출력 (DB 불필요)")
    p.add_argument("--dry-run-out", default=None, help="dry-run 결과 저장 경로 (없으면 stdout)")
    p.add_argument("--db-url", default=None)
    p.add_argument("--ollama-url", default=None)
    p.add_argument("--embed-model", default=None)
    p.add_argument("--target-tokens", type=int, default=500, help="병합 목표 토큰 수")
    p.add_argument("--hard-max-tokens", type=int, default=1000, help="강제 분할 상한 토큰 수")
    return p.parse_args()


def _upload_tables_to_s3(table_metas: list) -> None:
    """표 원본 markdown을 S3에 업로드. S3_BUCKET 환경변수 미설정 시 로컬 캐시로 대체."""
    import os
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        _cache_tables_locally(table_metas)
        return
    try:
        import boto3
        # boto3는 AWS_REGION을 보지 않는다(표준은 AWS_DEFAULT_REGION). 둘 다 없으면 us-east-1로
        # 떨어져 ap-northeast-2 버킷과 어긋나므로 팀 규약 변수를 읽어 직접 넘긴다.
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        for tm in table_metas:
            s3.put_object(
                Bucket=bucket,
                Key=f"policy-tables/{tm.table_id}.md",
                Body=tm.markdown.encode("utf-8"),
                ContentType="text/markdown",
            )
            logger.info(f"  S3 업로드: {tm.table_id} ({tm.row_count}행)")
    except Exception as e:
        logger.warning(f"S3 업로드 실패, 로컬 캐시로 대체: {e}")
        _cache_tables_locally(table_metas)


def _cache_tables_locally(table_metas: list) -> None:
    """S3 미설정 시 .table_cache/ 폴더에 markdown 저장."""
    from pathlib import Path
    cache_dir = Path(".table_cache")
    cache_dir.mkdir(exist_ok=True)
    for tm in table_metas:
        (cache_dir / f"{tm.table_id}.md").write_text(tm.markdown, encoding="utf-8")
    logger.info(f"  로컬 캐시 저장: .table_cache/ ({len(table_metas)}건)")


def main() -> None:
    args = _parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"파일 없음: {pdf_path}")
        sys.exit(1)

    from insurance_chunker.chunker import auto_doc_type
    from insurance_chunker.models import DocMeta, compute_doc_hash

    doc_type = args.doc_type or auto_doc_type(pdf_path.name)
    doc_hash = compute_doc_hash(str(pdf_path))
    logger.info(f"PDF: {pdf_path.name}  [{doc_type}]  {args.insurer} / {args.product}")
    logger.info(f"doc_hash={doc_hash[:16]}...")

    meta = DocMeta(
        source_pdf=pdf_path.name,
        doc_hash=doc_hash,
        doc_type=doc_type,
        insurer=args.insurer,
        product_name=args.product,
        product_code=args.product_code,
        effective_date=args.effective_date,
        generation=args.generation,
        product_id=args.product_id,
    )

    conn = None
    if not args.dry_run:
        from db.storage import delete_by_doc_hash, doc_already_ingested, get_connection, init_schema
        conn = get_connection(args.db_url)
        init_schema(conn, skip=args.no_init_schema)
        if doc_already_ingested(conn, doc_hash):
            if args.overwrite:
                delete_by_doc_hash(conn, doc_hash)
            else:
                logger.info("이미 ingestion된 파일. 건너뜀 (재처리: --overwrite)")
                conn.close()
                return

    # Phase 1: 파싱
    logger.info("Phase 1: PDF 파싱")
    from insurance_chunker.pdf_parser import parse_pdf
    pages = parse_pdf(str(pdf_path), use_ocr=not args.no_ocr,
                      use_vision=not args.no_vision)

    # Phase 2: 청킹
    logger.info("Phase 2: 청킹")
    from insurance_chunker.chunker import chunk_document
    chunks, table_metas = chunk_document(
        pages, meta,
        pdf_path=str(pdf_path) if doc_type == "policy_terms" else None,
        target=args.target_tokens,
        hard_max=args.hard_max_tokens,
    )
    logger.info(f"  {len(chunks)}개 청크 생성 | 표 분할 {len(table_metas)}건")

    # Phase 2.5: 품질 검증
    from insurance_chunker.validator import validate_chunks
    vr = validate_chunks(chunks)
    vr.log()
    chunks = vr.valid_chunks

    # Phase 3: 임베딩
    if not args.no_embed:
        logger.info("Phase 3: 임베딩")
        from insurance_chunker.embedder import embed_chunks
        chunks = embed_chunks(chunks, ollama_url=args.ollama_url, model=args.embed_model)

    # Phase 4: 저장 / dry-run
    if args.dry_run:
        tokens = [c.token_count for c in chunks]
        summary = {
            "source_pdf": meta.source_pdf,
            "doc_type": meta.doc_type,
            "insurer": meta.insurer,
            "product_name": meta.product_name,
            "generation": meta.generation,
            "chunk_count": len(chunks),
            "chunk_type_counts": dict(Counter(c.chunk_type for c in chunks)),
            "token_stats": {
                "min": min(tokens) if tokens else 0,
                "max": max(tokens) if tokens else 0,
                "avg": round(sum(tokens) / len(tokens)) if tokens else 0,
            },
            "over_600": sum(1 for t in tokens if t > 600),
            "warnings": vr.warnings,
        }

        def _to_dict(c):
            d = asdict(c)
            if d.get("embedding"):
                d["embedding"] = d["embedding"][:3] + ["..."]
            return d

        output = {"summary": summary, "chunks": [_to_dict(c) for c in chunks]}
        text = json.dumps(output, ensure_ascii=False, indent=2)
        if args.dry_run_out:
            Path(args.dry_run_out).write_text(text, encoding="utf-8")
            logger.info(f"저장: {args.dry_run_out}")
        else:
            print(text)
    else:
        logger.info("Phase 4: pgvector 저장")
        from db.storage import upsert_chunks, verify_upsert

        # 표 원본 S3 업로드 (table_id = S3 key, FK 없음)
        if table_metas:
            logger.info(f"  표 S3 업로드: {len(table_metas)}건")
            _upload_tables_to_s3(table_metas)

        upsert_chunks(conn, chunks)
        verify_upsert(conn, doc_hash)
        conn.close()

    logger.info("완료")


if __name__ == "__main__":
    main()
