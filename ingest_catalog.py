"""카탈로그 기반 인덱싱 — corpus_worker가 S3에 스테이징한 약관을 받아 적재한다.

SM17-YoonDongJu/AI 레포의 corpus_worker가 Notion→S3 스테이징을 하고 결과를 카탈로그에 적는다:
  ai.corpus_file       company·product_name·product_code·effective_date·category·is_latest
  ai.corpus_file_part  s3_key·sha256·status('uploaded')

이 CLI는 그 카탈로그를 소스로 삼는다 — docs.yaml을 손으로 쓸 필요가 없고, 보험사·상품명 같은
메타데이터도 DB에서 그대로 온다. 파트(첨부)마다 PDF가 하나이므로 파트 단위로 적재한다.

중복은 두 번 거른다. 1차는 카탈로그의 part.sha256으로 — corpus_worker의 sha256과 우리
compute_doc_hash가 모두 파일 바이트 sha256이라 값이 같다. 덕분에 이미 적재된 문서는
S3에서 내려받지 않는다. 2차는 다운로드 후 _run_one이 실제 파일 해시로 다시 확인한다.

사용:
  python ingest_catalog.py                    # DATABASE_URL·S3_BUCKET 환경변수
  python ingest_catalog.py --limit 5          # 상위 5건만(우선순위 순)
  python ingest_catalog.py --dry-run          # DB 저장 없이 청킹 결과만
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import runlog

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# 우선순위 높은 문서부터. is_latest가 NULL인 행은 제외하지 않는다(미기입일 뿐 구판이 아니다).
_CATALOG_SQL = """
SELECT f.company, f.product_name, f.product_code, f.effective_date, f.category,
       p.s3_key, p.sha256, p.notion_file_name, p.part_order
FROM ai.corpus_file f
JOIN ai.corpus_file_part p ON p.file_page_id = f.notion_page_id
WHERE f.category = %s
  AND f.is_latest IS NOT FALSE
  AND p.status = 'uploaded'
  AND p.s3_key IS NOT NULL
ORDER BY f.priority DESC, f.notion_page_id, p.part_order
"""

_SUMMARY_PAT = re.compile(r"요약서|상품안내|상품설명서")

# corpus_worker는 원본이 무엇이든 S3 키를 .pdf로 만든다(s3.py의 _KEY_SUFFIX). 그래서 키만 보면
# .txt·.hwp.zip도 PDF처럼 보인다 — 실제 형식은 notion_file_name의 확장자로만 판별된다.
# 우리 파이프라인은 PDF 전용이라(PyMuPDF) 그 외 형식은 파싱해도 0청크로 끝난다.
_PDF_EXT = ".pdf"

# 0청크·오류로 끝난 문서를 몇 번까지 다시 시도할지. 넘으면 격리(QUARANTINED)해 건너뛴다.
# 0청크 문서는 policy_chunks에 행이 안 생겨 doc_already_ingested가 영원히 False라, 상한이
# 없으면 매 주기 S3에서 다시 받아 다시 파싱한다(주기가 7일이니 영원히).
_MAX_RETRY = int(os.environ.get("INGEST_MAX_RETRY", "3"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ai.corpus_file 카탈로그 기반 인덱싱 (S3에서 PDF 수신)")
    p.add_argument("--category", default="terms", help="corpus_file.category (기본 terms)")
    p.add_argument("--limit", type=int, default=None, help="처리할 최대 건수(우선순위 순)")
    p.add_argument("--bucket", default=None, help="S3 버킷 (없으면 S3_BUCKET 환경변수)")
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
    p.add_argument("--retry-quarantined", action="store_true",
                   help=f"{_MAX_RETRY}회 연속 실패로 격리된 문서도 다시 시도한다")
    p.add_argument("--target-tokens", type=int, default=500)
    p.add_argument("--hard-max-tokens", type=int, default=1000)
    return p.parse_args()


def _doc_type_for(filename: str, category: str) -> str:
    """파일명 기반 판정을 카탈로그 category로 보정한다.

    auto_doc_type은 패턴이 하나도 안 맞으면 product_summary로 떨어지는데, 그 값이면
    임베딩·표추출을 건너뛴다(_run_one). category='terms'는 약관임을 이미 알려주므로
    파일명에 '요약서' 같은 명시 신호가 없는 한 policy_terms로 본다.
    """
    from insurance_chunker.chunker import auto_doc_type
    dt = auto_doc_type(filename)
    if category == "terms" and dt == "product_summary" and not _SUMMARY_PAT.search(filename):
        return "policy_terms"
    return dt


def _fetch_catalog(conn, category: str, limit: int | None) -> list[dict]:
    """카탈로그에서 업로드 완료된 파트 목록을 읽는다."""
    import psycopg2
    try:
        with conn.cursor() as cur:
            cur.execute(_CATALOG_SQL, (category,))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except psycopg2.errors.InsufficientPrivilege:
        conn.rollback()
        logger.error(
            "ai.corpus_file 조회 권한이 없습니다. AI 레포 마이그레이션에 다음 GRANT가 필요합니다:\n"
            "  GRANT USAGE ON SCHEMA ai TO corpus_owner;\n"
            "  GRANT SELECT ON ai.corpus_file, ai.corpus_file_part TO corpus_owner;")
        raise
    if limit is not None:
        rows = rows[:limit]
    return rows


def _split_pdf_rows(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """PDF만 남기고, 제외된 형식을 확장자별로 센다(무엇을 안 했는지 로그로 드러내기 위함)."""
    pdfs, skipped = [], {}
    for r in rows:
        name = r["notion_file_name"]
        # 파일명을 모르면 판별할 수 없으므로 시도한다(키는 항상 .pdf라 단서가 안 된다).
        if not name or name.lower().endswith(_PDF_EXT):
            pdfs.append(r)
            continue
        ext = Path(name).suffix.lower() or "(확장자 없음)"
        skipped[ext] = skipped.get(ext, 0) + 1
    return pdfs, skipped


def _s3_client():
    """region을 명시해 S3 클라이언트를 만든다(corpus_worker와 동일).

    boto3는 AWS_REGION을 보지 않는다 — 표준 변수는 AWS_DEFAULT_REGION이고, 둘 다 없으면
    us-east-1로 떨어져 ap-northeast-2 버킷 접근이 어긋난다(실측 확인). 팀 .env 규약이
    AWS_REGION이므로 그 값을 우선 읽어 직접 넘긴다.
    """
    import boto3
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return boto3.client("s3", region_name=region) if region else boto3.client("s3")


def _should_quarantine(hist: dict | None, retry_ok: bool) -> bool:
    """이 문서를 이번 주기에 건너뛸지. hist는 runlog.attempt()의 반환값.

    0청크·오류가 _MAX_RETRY회 연속이면 다운로드도 파싱도 하지 않는다. 성공 한 번이면
    runlog가 카운터를 0으로 되돌리므로, 파일이 고쳐져 다시 적재되면 자동으로 풀린다.
    """
    if retry_ok or not hist:
        return False
    return hist.get("attempts", 0) >= _MAX_RETRY


def _download(bucket: str, key: str, dest: Path) -> bool:
    try:
        _s3_client().download_file(bucket, key, str(dest))
    except Exception as e:  # noqa: BLE001 - 한 건 실패가 전체를 멈추게 하지 않는다
        logger.error(f"  S3 다운로드 실패 s3://{bucket}/{key}: {e}")
        return False
    return True


def main() -> None:
    args = _parse_args()

    bucket = args.bucket or os.environ.get("S3_BUCKET")
    if not bucket:
        logger.error("S3 버킷 미설정 — --bucket 또는 S3_BUCKET 환경변수를 지정하세요.")
        sys.exit(1)

    from db.storage import doc_already_ingested, get_connection, init_schema
    from ingest_many import _fmt_phases, run_one_safe

    conn = get_connection(args.db_url)
    init_schema(conn, skip=args.no_init_schema)

    rows = _fetch_catalog(conn, args.category, args.limit)
    rows, skipped_ext = _split_pdf_rows(rows)
    logger.info(f"카탈로그 대상 {len(rows)}건 (category={args.category}, bucket={bucket})")
    if skipped_ext:
        detail = ", ".join(f"{k} {v}건" for k, v in sorted(skipped_ext.items(),
                                                          key=lambda kv: -kv[1]))
        logger.info(f"PDF 아님 {sum(skipped_ext.values())}건 제외 — {detail}")

    dry_run_dir = Path(args.dry_run_dir)
    if args.dry_run:
        dry_run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t_run = time.time()
    retry_ok = args.retry_quarantined or args.overwrite
    with tempfile.TemporaryDirectory(prefix="corpus-pdf-") as tmp:
        for i, row in enumerate(rows, 1):
            name = row["notion_file_name"] or Path(row["s3_key"]).name
            sha = row["sha256"]
            logger.info(f"\n[{i}/{len(rows)}] {name}  ({row['company']} / {row['product_name']})")

            # 1차 중복 제거 — 카탈로그 sha256이 곧 doc_hash라 내려받기 전에 거른다.
            if not args.dry_run and not args.overwrite and sha \
                    and doc_already_ingested(conn, sha):
                logger.info("  SKIPPED: 이미 적재됨(다운로드 생략)")
                results.append({"status": "SKIPPED", "pdf": name})
                # 이력에 남겨야 멱등 스킵률(중복 제거로 아낀 양)을 잴 수 있다.
                runlog.record_item(sha256=sha, name=name, status="SKIPPED",
                                   source="catalog", insurer=row["company"],
                                   product=row["product_name"])
                continue

            # 격리 — 0청크/오류가 연속 _MAX_RETRY회면 더는 받지도, 파싱하지도 않는다.
            hist = runlog.attempt(sha) if (sha and not args.dry_run) else None
            if _should_quarantine(hist, retry_ok):
                logger.warning(
                    f"  QUARANTINED: {hist['status']} {hist['attempts']}회 연속 "
                    f"(마지막 {hist['last_at']}) — 다운로드·파싱 생략. "
                    f"다시 시도하려면 --retry-quarantined")
                results.append({"status": "QUARANTINED", "pdf": name})
                runlog.record_item(sha256=sha, name=name, status="QUARANTINED",
                                   source="catalog", insurer=row["company"],
                                   product=row["product_name"],
                                   error=f"{hist['status']} {hist['attempts']}회 연속")
                continue

            # auto_doc_type이 파일명을 보므로 원래 이름을 유지한다.
            dest = Path(tmp) / f"{row['sha256'] or i}_{Path(name).name}"
            if not _download(bucket, row["s3_key"], dest):
                results.append({"status": "ERROR", "pdf": name})
                if not args.dry_run:
                    runlog.record_item(sha256=sha, name=name, status="ERROR",
                                       error="S3 다운로드 실패", source="catalog")
                continue

            eff = row["effective_date"]
            doc = {
                "pdf": str(dest),
                "doc_type": _doc_type_for(name, row["category"]),
                "insurer": row["company"] or "미상",
                "product_name": row["product_name"] or Path(name).stem,
                "product_code": row["product_code"],
                "effective_date": eff.isoformat() if eff else None,
            }
            result = run_one_safe(doc, args, dry_run_dir)
            # 임시 파일명이 아니라 원래 파일명으로 로그·결과를 남긴다.
            result["pdf"] = name
            # 0청크는 성공이 아니다 — 적재가 0건이므로 EMPTY로 따로 센다(재시도 상한 대상).
            if result["status"] == "OK" and result.get("chunks", 0) == 0:
                result["status"] = "EMPTY"
            results.append(result)

            if result["status"] == "EMPTY":
                left = max(0, _MAX_RETRY - ((hist or {}).get("attempts", 0) + 1))
                logger.warning(f"  0청크 — 적재 없음(PDF 아님/파싱 실패 가능). "
                               f"남은 재시도 {left}회. {result.get('elapsed_s', 0)}s")
            elif result["status"] == "OK":
                logger.info(f"  OK: {result['chunks']}청크 | 경고 {result['warnings']}건 "
                            f"| {result['elapsed_s']}s | {_fmt_phases(result['phases'])}")
            elif result["status"] == "SKIPPED":
                logger.info(f"  SKIPPED: {result.get('reason')}")
            else:
                logger.error(f"  ERROR: {result.get('error')}")

            if not args.dry_run:
                runlog.record_item(
                    sha256=sha, name=name, status=result["status"],
                    chunks=result.get("chunks", 0), warnings=result.get("warnings", 0),
                    elapsed_s=result.get("elapsed_s", 0.0), phases=result.get("phases"),
                    error=result.get("error"), source="catalog",
                    insurer=row["company"], product=row["product_name"])

            dest.unlink(missing_ok=True)  # 디스크 점유를 건별로 반환

    conn.close()

    def n(st: str) -> int:
        return sum(1 for r in results if r["status"] == st)

    ok, empty, skipped = n("OK"), n("EMPTY"), n("SKIPPED")
    quarantined, err = n("QUARANTINED"), n("ERROR")
    total_chunks = sum(r.get("chunks", 0) for r in results if r["status"] == "OK")
    elapsed = round(time.time() - t_run, 1)
    logger.info(f"\nOK={ok} 0청크={empty} SKIPPED={skipped} 격리={quarantined} ERROR={err} "
                f"| 총 청크={total_chunks}개 | {elapsed}s")
    if empty:
        logger.warning(f"0청크 {empty}건 — {_MAX_RETRY}회 연속이면 격리된다. "
                       "반복되면 대상 선정(category/파일형식)을 재검토할 것")
    if quarantined:
        logger.warning(f"격리 {quarantined}건은 이번 주기에 손대지 않았다 — "
                       "state/attempts.json에서 목록 확인")

    if not args.dry_run:
        runlog.record_run({
            "source": "catalog", "category": args.category, "total": len(rows),
            "ok": ok, "empty": empty, "skipped": skipped, "quarantined": quarantined,
            "error": err, "total_chunks": total_chunks,
            "skipped_non_pdf": sum(skipped_ext.values()), "elapsed_s": elapsed,
        })
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
