#!/usr/bin/env python3
"""manifest 기반 일괄 ingestion.

사용법:
  python ingest_many.py --manifest documents.yaml
  python ingest_many.py --manifest documents.yaml --dry-run --dry-run-dir out/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-run-dir", default="dry_run_out")
    p.add_argument("--db-url", default=None)
    p.add_argument("--ollama-url", default=None)
    p.add_argument("--embed-model", default=None)
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target-tokens", type=int, default=500)
    p.add_argument("--hard-max-tokens", type=int, default=1000)
    return p.parse_args()


def _load_manifest(path: str) -> list[dict]:
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.error("PyYAML 미설치. pip install pyyaml")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    docs = data.get("documents", [])
    if not docs:
        logger.error("manifest에 documents 항목 없음")
        sys.exit(1)
    return docs


def _run_one(doc: dict, args: argparse.Namespace, dry_run_dir: Path) -> dict:
    from collections import Counter
    from dataclasses import asdict
    from insurance_chunker.models import DocMeta, compute_doc_hash
    from insurance_chunker.pdf_parser import parse_pdf
    from insurance_chunker.chunker import chunk_document, auto_doc_type
    from insurance_chunker.validator import validate_chunks

    pdf_path = Path(doc["pdf"])
    if not pdf_path.exists():
        return {"pdf": str(pdf_path), "status": "ERROR", "error": "파일 없음"}

    doc_type = doc.get("doc_type") or auto_doc_type(pdf_path.name)
    t0 = time.time()
    meta = DocMeta(
        source_pdf=pdf_path.name,
        doc_hash=compute_doc_hash(str(pdf_path)),
        doc_type=doc_type,
        insurer=doc["insurer"],
        product_name=doc["product_name"],
        product_code=doc.get("product_code"),
        effective_date=doc.get("effective_date"),
        yakwan=doc.get("yakwan"),
        generation=doc.get("generation"),
    )

    conn = None
    if not args.dry_run:
        from db.storage import get_connection, init_schema, doc_already_ingested, delete_by_doc_hash
        conn = get_connection(args.db_url)
        init_schema(conn)
        if doc_already_ingested(conn, meta.doc_hash):
            if args.overwrite:
                delete_by_doc_hash(conn, meta.doc_hash)
            else:
                conn.close()
                return {"pdf": pdf_path.name, "status": "SKIPPED", "reason": "already ingested"}

    pages = parse_pdf(str(pdf_path), use_ocr=not args.no_ocr,
                      use_vision=not args.no_vision)
    chunks = chunk_document(pages, meta,
                            pdf_path=str(pdf_path) if doc_type == "policy_terms" else None,
                            target=args.target_tokens,
                            hard_max=args.hard_max_tokens)
    vr = validate_chunks(chunks)
    chunks = vr.valid_chunks

    if not args.no_embed and doc_type != "product_summary":
        from insurance_chunker.embedder import embed_chunks
        chunks = embed_chunks(chunks, ollama_url=args.ollama_url, model=args.embed_model)

    if args.dry_run:
        tokens = [c.token_count for c in chunks]
        summary = {
            "chunk_count": len(chunks),
            "chunk_type_counts": dict(Counter(c.chunk_type for c in chunks)),
            "over_600": sum(1 for t in tokens if t > 600),
            "tok_avg": round(sum(tokens) / len(tokens)) if tokens else 0,
            "warnings": len(vr.warnings),
        }
        out_file = dry_run_dir / f"{pdf_path.stem}_dry_run.json"
        out_file.write_text(
            json.dumps({"summary": summary, "chunks": [asdict(c) for c in chunks]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        from db.storage import upsert_chunks, verify_upsert
        upsert_chunks(conn, chunks)
        verify_upsert(conn, meta.doc_hash)
        conn.close()

    return {
        "pdf": pdf_path.name, "status": "OK", "doc_type": doc_type,
        "chunks": len(chunks), "warnings": len(vr.warnings),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> None:
    args = _parse_args()
    docs = _load_manifest(args.manifest)
    dry_run_dir = Path(args.dry_run_dir)
    if args.dry_run:
        dry_run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"총 {len(docs)}개 문서 처리 시작")
    results = []
    for i, doc in enumerate(docs, 1):
        logger.info(f"\n[{i}/{len(docs)}] {doc.get('pdf', '?')}")
        result = _run_one(doc, args, dry_run_dir)
        results.append(result)
        if result["status"] == "OK":
            logger.info(f"  OK: {result['chunks']}청크 | 경고 {result['warnings']}건 | {result['elapsed_s']}s")
        elif result["status"] == "SKIPPED":
            logger.info(f"  SKIPPED: {result.get('reason')}")
        else:
            logger.error(f"  ERROR: {result.get('error')}")

    ok = sum(1 for r in results if r["status"] == "OK")
    err = sum(1 for r in results if r["status"] == "ERROR")
    total_chunks = sum(r.get("chunks", 0) for r in results if r["status"] == "OK")
    logger.info(f"\nOK={ok} ERROR={err} | 총 청크={total_chunks}개")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
