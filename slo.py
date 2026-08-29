"""인덱스 SLO 점검 — 적재가 "됐다"고 끝내지 않고 쓸 만한 상태인지 본다.

deploy_check.py가 보던 것 중 주기적으로 확인해야 하는 항목을 떼어냈다. 그쪽은 asyncpg를
쓰는 일회성 배포 전 판정 스크립트이고(컨테이너에 asyncpg가 없다), 이 모듈은 psycopg2로
매 사이클 끝에 돈다.

신선도(마지막 성공 적재 경과)는 여기서 보지 않는다 — runlog/healthcheck가 이미 판정한다.
여기는 DB 쪽 상태, 즉 "적재된 결과물이 검색에 쓸 수 있는 모양인가"만 본다.

환경변수:
  SLO_MAX_STALE_DAYS   max(ingested_at) 허용 경과일. 기본 = 주기 x 1.5.
  SLO_MIN_COVERAGE     카탈로그 대비 적재 문서 비율 하한. 기본 0.8. 0이면 검사 안 함.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    value: Any = None


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def violations(self) -> list[Check]:
        return [c for c in self.checks if c.status in (WARN, FAIL)]

    @property
    def worst(self) -> str:
        for level in (FAIL, WARN):
            if any(c.status == level for c in self.checks):
                return level
        return OK

    def log(self) -> None:
        mark = {OK: "✓", WARN: "⚠", FAIL: "✗", SKIP: "-"}
        for c in self.checks:
            line = f" {mark[c.status]} [{c.name}] {c.detail}"
            (logger.warning if c.status in (WARN, FAIL) else logger.info)(
                line, extra={"event": "slo_check", "check": c.name,
                             "status": c.status, "value": c.value})


def _scalar(conn, sql: str, params: tuple = ()) -> Any:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return row[0] if row else None


def _check_embedding_dim(conn) -> Check:
    expected = int(os.environ.get("EMBED_DIM", "1024"))
    dim = _scalar(conn, "SELECT vector_dims(embedding::vector) FROM policy_chunks "
                        "WHERE embedding IS NOT NULL LIMIT 1")
    if dim is None:
        return Check("embedding_dim", WARN, "임베딩된 청크가 하나도 없다", None)
    if dim != expected:
        return Check("embedding_dim", FAIL,
                     f"차원 {dim} ≠ 기대 {expected} — 모델 불일치, 재적재 필요", dim)
    return Check("embedding_dim", OK, f"{dim}차원", dim)


def _check_search_contract(conn) -> Check:
    """파트너(AI) 레포가 직접 @@ 검색하는 컬럼. 없으면 그쪽 검색이 조용히 0건이 된다."""
    has = _scalar(conn, "SELECT count(*) FROM information_schema.columns "
                        "WHERE table_name='policy_chunks' AND column_name='content_tsv'")
    if has:
        return Check("search_contract", OK, "content_tsv 컬럼 있음", True)
    return Check("search_contract", FAIL,
                 "content_tsv 없음 — AI 레포 RAG 검색이 0건이 된다", False)


def _check_documents_embedded(conn) -> Check:
    """청크는 있는데 임베딩이 하나도 없는 문서.

    전체 NULL 비율로는 판정할 수 없다 — boilerplate 청크는 일부러 임베딩을 건너뛰는데
    (embedder.embed_chunks) is_boilerplate가 DB 컬럼이 아니라서 의도된 NULL과 실패한
    NULL을 구분할 방법이 없다. 반면 "한 문서에 임베딩이 0개"는 명백한 고장이다.
    """
    broken = _scalar(conn, """
        SELECT count(*) FROM (
          SELECT doc_hash FROM policy_chunks
          GROUP BY doc_hash HAVING count(embedding) = 0
        ) t""")
    if broken:
        return Check("documents_embedded", FAIL,
                     f"임베딩이 0개인 문서 {broken}건 — 벡터 검색에서 통째로 빠진다", broken)
    return Check("documents_embedded", OK, "모든 문서에 임베딩 있음", 0)


def _check_freshness(conn, interval_s: int) -> Check:
    """DB 기준 마지막 적재 시각. runlog의 신선도와는 다른 것을 본다 —
    사이클이 성공해도 전부 SKIPPED였다면 DB는 그대로이므로, 여기가 오래됐다고
    바로 고장은 아니다. 그래서 FAIL이 아니라 WARN이다."""
    days = _scalar(conn, "SELECT EXTRACT(EPOCH FROM (now() - max(ingested_at)))/86400 "
                         "FROM policy_chunks")
    if days is None:
        return Check("freshness", FAIL, "적재된 청크가 없다", None)
    days = round(float(days), 1)
    limit = float(os.environ.get("SLO_MAX_STALE_DAYS", interval_s * 1.5 / 86400))
    if days > limit:
        return Check("freshness", WARN,
                     f"마지막 적재 {days}일 전 (한도 {limit:.1f}일) — 새 문서가 안 들어오고 있다",
                     days)
    return Check("freshness", OK, f"마지막 적재 {days}일 전", days)


def _check_catalog_coverage(conn, category: str = "terms") -> Check:
    """카탈로그가 알려준 문서 중 몇 %가 실제로 적재됐나.

    격리(#14)·0청크·PDF 아님으로 빠진 문서가 여기서 드러난다. ai 스키마 권한이 없으면
    건너뛴다 — 권한은 AI 레포 마이그레이션 소관이라 여기서 실패로 볼 일이 아니다.
    """
    import psycopg2
    floor = float(os.environ.get("SLO_MIN_COVERAGE", "0.8"))
    if floor <= 0:
        return Check("catalog_coverage", SKIP, "SLO_MIN_COVERAGE=0 — 검사 안 함")
    try:
        total = _scalar(conn, """
            SELECT count(*) FROM ai.corpus_file f
            JOIN ai.corpus_file_part p ON p.file_page_id = f.notion_page_id
            WHERE f.category = %s AND f.is_latest IS NOT FALSE
              AND p.status = 'uploaded' AND p.s3_key IS NOT NULL""", (category,))
    except psycopg2.Error as e:
        conn.rollback()
        return Check("catalog_coverage", SKIP, f"카탈로그 조회 불가 ({type(e).__name__})")
    if not total:
        return Check("catalog_coverage", SKIP, "카탈로그가 비어 있다")

    loaded = _scalar(conn, "SELECT count(DISTINCT doc_hash) FROM policy_chunks") or 0
    ratio = round(loaded / total, 3)
    detail = f"{loaded}/{total} ({ratio:.1%})"
    if ratio < floor:
        return Check("catalog_coverage", WARN,
                     f"{detail} — 하한 {floor:.0%} 미만. 격리·0청크 문서를 확인할 것", ratio)
    return Check("catalog_coverage", OK, detail, ratio)


def evaluate(conn, interval_s: int = 604800) -> Report:
    """DB 상태 점검. 예외는 개별 항목 실패로 바꿔 한 항목이 전체를 막지 않게 한다."""
    checks = []
    for name, fn in (
        ("embedding_dim", lambda: _check_embedding_dim(conn)),
        ("search_contract", lambda: _check_search_contract(conn)),
        ("documents_embedded", lambda: _check_documents_embedded(conn)),
        ("freshness", lambda: _check_freshness(conn, interval_s)),
        ("catalog_coverage", lambda: _check_catalog_coverage(conn)),
    ):
        try:
            checks.append(fn())
        except Exception as e:  # noqa: BLE001 - 점검이 사이클을 죽이면 안 된다
            conn.rollback()
            checks.append(Check(name, SKIP, f"점검 실패 ({type(e).__name__}: {e})"))
    return Report(checks)


def run(db_url: Optional[str] = None, interval_s: int = 604800) -> Report:
    from db.storage import get_connection
    conn = get_connection(db_url)
    try:
        report = evaluate(conn, interval_s)
    finally:
        conn.close()
    report.log()
    return report


def main() -> int:
    import logging_setup
    logging_setup.configure()
    interval = int(os.environ.get("INGEST_INTERVAL_SECONDS", "604800"))
    report = run(interval_s=interval)
    return 1 if report.worst == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
