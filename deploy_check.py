"""배포 전 점검 — 운영 DB 상태를 보고 재인덱싱 필요 여부를 판정한다.

실행(운영 환경에서):
    DATABASE_URL=postgres://... .venv/bin/python deploy_check.py

배경: 이번 사이클에서 검증된 개선 4건이 프로덕션에 미반영 상태다.
  1) v6 청킹        — 검색 R@5 0.851 → 0.894 (47문항, rerank 기준)
  2) SEARCH_RERANK  — v6은 rerank 전제로만 v3를 이긴다(embed 단독은 0.830 < 0.851)
  3) 질의 프리픽스   — eval은 instruct 프리픽스를 쓰는데 프로덕션은 안 썼다(벡터 recall 저평가)
  4) 임베딩 태그    — 기본값 qwen3:embedding 이 ollama에 없는 태그였다

이 중 3)은 질의 벡터만 바뀌므로 재인덱싱 불필요, 1)·4)는 재적재가 필요할 수 있다.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

EXPECT_DIM = 1024
EXPECT_MODEL = "qwen3-embedding:0.6b"


def _load_env_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DATABASE_URL"):
                return line.split("=", 1)[1].strip()
    return None


async def main() -> None:
    url = _load_env_url()
    if not url:
        print("✗ DATABASE_URL 미설정 — 운영 환경에서 실행하세요.")
        print("  예: DATABASE_URL=postgres://user:pw@host/db .venv/bin/python deploy_check.py")
        return

    import asyncpg
    conn = await asyncpg.connect(url, timeout=15)
    findings: list[tuple[str, str]] = []   # (상태, 메시지)

    n = await conn.fetchval("SELECT count(*) FROM policy_chunks")
    n_emb = await conn.fetchval("SELECT count(*) FROM policy_chunks WHERE embedding IS NOT NULL")
    print(f"policy_chunks: {n:,}행 (임베딩 {n_emb:,})")

    # 1) 임베딩 차원
    dim = await conn.fetchval(
        "SELECT vector_dims(embedding::vector) FROM policy_chunks "
        "WHERE embedding IS NOT NULL LIMIT 1")
    if dim == EXPECT_DIM:
        findings.append(("ok", f"임베딩 차원 {dim} — 정상"))
    else:
        findings.append(("재적재", f"임베딩 차원 {dim} ≠ 기대 {EXPECT_DIM} → 모델 불일치, 재적재 필요"))

    # 2) content_tsv 생성 컬럼 (파트너 레포 계약)
    has_tsv = await conn.fetchval(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name='policy_chunks' AND column_name='content_tsv'")
    findings.append(("ok" if has_tsv else "스키마",
                     "content_tsv 컬럼 있음" if has_tsv else
                     "content_tsv 없음 → db/schema.sql 마이그레이션 필요"))

    # 3) v6 청킹 반영 여부 — 호 단위 분할의 지문:
    #    같은 (doc_hash, section, article_number)에 청크가 여러 개면 호 분할된 것.
    split = await conn.fetchval("""
        SELECT count(*) FROM (
          SELECT doc_hash, section, article_number
          FROM policy_chunks
          WHERE article_number IS NOT NULL AND table_id IS NULL
          GROUP BY 1,2,3 HAVING count(*) > 1
        ) t""")
    if split and split > 0:
        findings.append(("ok", f"호 단위 분할 흔적 {split:,}건 — v5/v6 청킹으로 보임"))
    else:
        findings.append(("재적재", "호 분할 흔적 없음 → v3 이전 청킹. 재적재해야 R@5 0.894를 얻는다"))

    # 4) 적재 시점 — 이번 사이클(8/10) 이전이면 구버전
    last = await conn.fetchval("SELECT max(ingested_at) FROM policy_chunks")
    print(f"마지막 적재: {last}")
    if last and str(last) < "2026-08-10":
        findings.append(("재적재", f"마지막 적재 {last} — 이번 사이클 수정 이전"))

    # 5) 문서 목록 (보험사 커버리지)
    rows = await conn.fetch(
        "SELECT insurer, count(DISTINCT doc_hash) d, count(*) c "
        "FROM policy_chunks GROUP BY 1 ORDER BY c DESC")
    print("\n보험사별 적재 현황:")
    for r in rows:
        print(f"  {r['insurer'] or '(없음)':20} 문서 {r['d']:>4}  청크 {r['c']:>8,}")
    if any((r["insurer"] or "").startswith("KB") for r in rows):
        findings.append(("경고", "KB 문서가 적재돼 있음 — 경계 검출 붕괴 상태(섹션 7개). "
                                 "eval/KB_GENERALIZATION_TEST.md 참조, 검색 품질 신뢰 불가"))

    await conn.close()

    print("\n" + "=" * 62)
    for state, msg in findings:
        mark = {"ok": "✓", "재적재": "▲", "스키마": "▲", "경고": "⚠"}[state]
        print(f" {mark} [{state}] {msg}")

    need = [f for f in findings if f[0] in ("재적재", "스키마")]
    print("=" * 62)
    if need:
        print("\n판정: 재적재/마이그레이션 필요\n")
        print("  1) db/schema.sql 적용 (ALTER 구문 포함)")
        print("  2) EMBED_MODEL=qwen3-embedding:0.6b 확인 후 재적재:")
        print("     EMBED_MODEL=qwen3-embedding:0.6b .venv/bin/python ingest_many.py <manifest>")
        print("  3) .venv/bin/python rebuild_search_terms.py")
        print("  4) 서비스에 SEARCH_RERANK=1 설정")
    else:
        print("\n판정: 재적재 불필요. 서비스에 SEARCH_RERANK=1 만 설정하면 된다.")
        print("  (질의 프리픽스 수정은 질의 벡터에만 영향 — 문서 벡터 불변)")


if __name__ == "__main__":
    asyncio.run(main())
