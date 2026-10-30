"""원본 PDF와 표 원문 — S3(운영) 또는 로컬 디렉터리(개발)에서 가져온다.

청커는 처리가 끝나면 로컬 사본을 지우지만 원본은 S3에 남는다. 키는 corpus_worker가
corpus/{category}/{sha256}{ext}로 만들므로 내용 주소다 — 같은 키면 같은 파일이라 다시 검증하지
않는다. 내려받은 파일은 doc_hash 이름으로 디스크에 두고 오래 안 쓴 것부터 지운다.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Optional

from viewer import env

_lock = threading.Lock()
_s3_client: Any = None
_pdf_index: Optional[dict[str, Path]] = None


def is_pdf_name(name: Optional[str]) -> bool:
    """corpus_worker는 원본이 무엇이든 키를 .pdf로 만든다 — 형식은 Notion 파일명으로만 안다.
    이름을 모르면 시도한다(ingest_catalog와 같은 규칙)."""
    return not name or name.lower().endswith(".pdf")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _s3() -> Any:
    global _s3_client
    with _lock:
        if _s3_client is None:
            import boto3

            _s3_client = boto3.client("s3", region_name=env.AWS_REGION)
        return _s3_client


def _local_pdf(doc_hash: str) -> Optional[Path]:
    """VIEWER_PDF_DIR 아래 PDF를 해시로 찾는다. 인덱스는 처음 한 번 만든다(파일을 더하면 재시작)."""
    global _pdf_index
    if not env.PDF_DIR:
        return None
    with _lock:
        if _pdf_index is None:
            _pdf_index = {sha256_of(p): p for p in Path(env.PDF_DIR).rglob("*")
                          if p.is_file() and p.suffix.lower() == ".pdf"}
        return _pdf_index.get(doc_hash)


def _prune(cache: Path) -> None:
    files = sorted(cache.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[env.CACHE_MAX_PDFS:]:
        old.unlink(missing_ok=True)


def pdf_path(doc_hash: str, s3_key: Optional[str]) -> Optional[Path]:
    """원본 PDF의 로컬 경로. 찾을 곳이 없으면 None, 받다가 실패하면 예외를 올린다."""
    local = _local_pdf(doc_hash)
    if local is not None:
        return local
    if not (s3_key and env.S3_BUCKET):
        return None
    cache = env.CACHE_DIR / "pdf"
    dest = cache / f"{doc_hash}.pdf"
    if dest.exists():
        os.utime(dest)  # 최근 사용 표시 — 정리 기준
        return dest
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f".{doc_hash}.{os.getpid()}.{threading.get_ident()}.part"
    try:
        _s3().download_file(env.S3_BUCKET, s3_key, str(tmp))
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    _prune(cache)
    return dest


def table_markdown(table_id: str) -> Optional[str]:
    """표 원문(markdown). 운영은 S3 policy-tables/, S3 없이 적재한 로컬은 .table_cache/에 있다."""
    if env.S3_BUCKET:
        from botocore.exceptions import ClientError

        try:
            obj = _s3().get_object(Bucket=env.S3_BUCKET, Key=f"policy-tables/{table_id}.md")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return obj["Body"].read().decode("utf-8")
    local = env.TABLE_DIR / f"{table_id}.md"
    return local.read_text(encoding="utf-8") if local.exists() else None


def page_count(path: Path) -> int:
    import pymupdf

    with pymupdf.open(path) as doc:
        return doc.page_count


def render_page(path: Path, page_number: int, dpi: int) -> bytes:
    """쪽을 PNG로 그린다. page_number는 1부터 센다 — 청크의 page_number와 같은 기준이다."""
    import pymupdf

    with pymupdf.open(path) as doc:
        return doc[page_number - 1].get_pixmap(dpi=dpi).tobytes("png")
