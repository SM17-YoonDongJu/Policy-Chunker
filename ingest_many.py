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

import logging_setup
import runlog

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

logging_setup.configure()
logger = logging.getLogger(__name__)


def _fmt_phases(phases: dict) -> str:
    """구간 타이밍을 로그 한 줄로. 어디가 병목인지 로그만 봐도 보이게."""
    return " ".join(f"{k}={v}s" for k, v in phases.items()) if phases else "-"


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
    p.add_argument("--no-init-schema", action="store_true",
                   help="스키마 DDL 실행 안 함 (운영은 AI 레포 migrations/corpus가 단일 관리)")
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
    """문서 1건 처리. 구간별 소요 시간(phases)까지 담아 돌려준다.

    phases가 없으면 "문서 1건에 N초"까지만 알 뿐 어디가 느린지 모른다. 파싱/청킹/임베딩/저장을
    나눠 재야 병목을 지목하고 개선 전후를 비교할 수 있다.
    """
    from collections import Counter
    from dataclasses import asdict

    from insurance_chunker.chunker import auto_doc_type, chunk_document
    from insurance_chunker.models import DocMeta, compute_doc_hash
    from insurance_chunker.pdf_parser import parse_pdf
    from insurance_chunker.validator import validate_chunks

    pdf_path = Path(doc["pdf"])
    if not pdf_path.exists():
        return {"pdf": str(pdf_path), "status": "ERROR", "error": "파일 없음", "phases": {}}

    doc_type = doc.get("doc_type") or auto_doc_type(pdf_path.name)
    t0 = time.time()
    timings: dict[str, float] = {}
    # 경계 검출 품질 — 청킹은 성공해도 섹션이 안 갈리면 조번호가 어긋난다.
    # 로그로만 흘려보내면 "성공한 문서"로 집계돼 아무도 모른다.
    chunk_report: dict = {}

    with runlog.phase(timings, "hash"):
        doc_hash = compute_doc_hash(str(pdf_path))
    meta = DocMeta(
        source_pdf=pdf_path.name,
        doc_hash=doc_hash,
        doc_type=doc_type,
        insurer=doc["insurer"],
        product_name=doc["product_name"],
        product_code=doc.get("product_code"),
        effective_date=doc.get("effective_date"),
        generation=doc.get("generation"),
        product_id=doc.get("product_id"),
    )

    conn = None
    if not args.dry_run:
        from db.storage import delete_by_doc_hash, doc_already_ingested, get_connection, init_schema
        conn = get_connection(args.db_url)
        init_schema(conn, skip=args.no_init_schema)
        if doc_already_ingested(conn, meta.doc_hash):
            if args.overwrite:
                delete_by_doc_hash(conn, meta.doc_hash)
            else:
                conn.close()
                return {"pdf": pdf_path.name, "status": "SKIPPED", "reason": "already ingested",
                        "doc_hash": meta.doc_hash, "phases": timings}

    try:
        with runlog.phase(timings, "parse"):
            pages = parse_pdf(str(pdf_path), use_ocr=not args.no_ocr,
                              use_vision=not args.no_vision)
        with runlog.phase(timings, "chunk"):
            chunks, table_metas = chunk_document(
                pages, meta,
                pdf_path=str(pdf_path) if doc_type == "policy_terms" else None,
                target=args.target_tokens,
                hard_max=args.hard_max_tokens,
                report=chunk_report,
            )
            vr = validate_chunks(chunks)
            chunks = vr.valid_chunks

        if not args.no_embed and doc_type != "product_summary":
            from insurance_chunker.embedder import embed_chunks
            with runlog.phase(timings, "embed"):
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
            from ingest import _upload_tables_to_s3
            with runlog.phase(timings, "store"):
                if table_metas:
                    _upload_tables_to_s3(table_metas)
                upsert_chunks(conn, chunks)
                verify_upsert(conn, meta.doc_hash)
    finally:
        # 예외로 빠져나가도 커넥션은 돌려준다 — 사이클이 문서 수만큼 커넥션을 새로 여므로
        # 흘리면 실패가 쌓일수록 RDS 커넥션이 고갈된다.
        if conn is not None:
            conn.close()

    return {
        "pdf": pdf_path.name, "status": "OK", "doc_type": doc_type,
        "doc_hash": meta.doc_hash,
        "chunks": len(chunks), "warnings": len(vr.warnings),
        "elapsed_s": round(time.time() - t0, 1),
        "phases": timings,
        "boundary_confidence": chunk_report.get("boundary_confidence"),
        "boundaries": chunk_report.get("boundaries"),
    }


def run_one_safe(doc: dict, args: argparse.Namespace, dry_run_dir: Path) -> dict:
    """_run_one을 감싸 어떤 예외든 ERROR 결과로 바꾼다.

    이게 없으면 손상된 PDF 한 건의 예외가 CLI 전체를 죽인다 — 그 문서는 DB에 안 남으니
    다음 주기에 또 같은 자리에서 죽고, 뒤에 있는 문서들은 영영 처리되지 않는다.
    """
    try:
        return _run_one(doc, args, dry_run_dir)
    except Exception as e:  # noqa: BLE001 - 한 건 실패가 나머지를 막지 않게
        logger.exception(f"처리 실패: {doc.get('pdf', '?')}")
        return {"pdf": str(doc.get("pdf", "?")), "status": "ERROR",
                "error": f"{type(e).__name__}: {e}", "phases": {}}


def main() -> None:
    args = _parse_args()
    docs = _load_manifest(args.manifest)
    dry_run_dir = Path(args.dry_run_dir)
    if args.dry_run:
        dry_run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"총 {len(docs)}개 문서 처리 시작")
    t_run = time.time()
    results = []
    for i, doc in enumerate(docs, 1):
        logger.info(f"\n[{i}/{len(docs)}] {doc.get('pdf', '?')}")
        result = run_one_safe(doc, args, dry_run_dir)
        results.append(result)
        if result["status"] == "OK":
            logger.info(f"  OK: {result['chunks']}청크 | 경고 {result['warnings']}건 "
                        f"| {result['elapsed_s']}s | {_fmt_phases(result['phases'])}")
        elif result["status"] == "SKIPPED":
            logger.info(f"  SKIPPED: {result.get('reason')}")
        else:
            logger.error(f"  ERROR: {result.get('error')}")
        if not args.dry_run:
            runlog.record_item(
                sha256=result.get("doc_hash"), name=result.get("pdf", "?"),
                status=result["status"], chunks=result.get("chunks", 0),
                warnings=result.get("warnings", 0), elapsed_s=result.get("elapsed_s", 0.0),
                phases=result.get("phases"), error=result.get("error"), source="manifest",
                boundary_confidence=result.get("boundary_confidence"))

    ok = sum(1 for r in results if r["status"] == "OK")
    err = sum(1 for r in results if r["status"] == "ERROR")
    total_chunks = sum(r.get("chunks", 0) for r in results if r["status"] == "OK")
    logger.info(f"\nOK={ok} ERROR={err} | 총 청크={total_chunks}개")
    if not args.dry_run:
        runlog.record_run({"source": "manifest", "total": len(docs), "ok": ok, "error": err,
                           "total_chunks": total_chunks,
                           "elapsed_s": round(time.time() - t_run, 1)})
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
