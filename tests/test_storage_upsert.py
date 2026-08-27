"""upsert_chunks 계약 검증 — executemany → execute_values 전환이 안전한지 확인한다.

execute_values는 배치를 INSERT ... VALUES (..),(..) 한 문장으로 합쳐 왕복을 배치당 1회로
줄인다. 대신 executemany에 없던 두 가지 제약이 생기는데, 여기서 그 둘을 지킨다.

  1) SQL 안의 %s가 정확히 하나여야 한다(값 자리). 나중에 누가 LIKE '%...' 같은 걸 넣으면
     조용히 어긋나므로 분해가 되는지 검사한다.
  2) 한 문장 안에 같은 chunk_id가 두 번 있으면 Postgres가 ON CONFLICT DO UPDATE에서 죽는다
     (건별 실행이던 executemany는 조용히 넘어갔다). _dedupe가 그 동작을 지킨다.

DB는 띄우지 않는다 — 서버 없이 확인 가능한 성질만 검사한다.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import storage  # noqa: E402
from insurance_chunker.models import InsuranceChunk  # noqa: E402


def _chunk(chunk_id: str, content: str = "본문") -> InsuranceChunk:
    return InsuranceChunk(
        chunk_id=chunk_id, content=content, content_tokens="본문", token_count=2,
        section="암 진단특별약관", page_number=1, doc_type="policy_terms",
        chunk_type="coverage", source_pdf="약관.pdf", doc_hash="deadbeef",
        insurer="메리츠화재", product_name="단체안심생활보험",
        product_code=None, effective_date=None,
    )


def _upsert_sql() -> str:
    """upsert_chunks 본문에 박혀 있는 SQL 리터럴을 꺼낸다."""
    src = inspect.getsource(storage.upsert_chunks)
    return src.split('sql = """')[1].split('"""')[0]


def test_sql_has_exactly_one_placeholder():
    """값 자리 %s가 하나뿐이어야 한다.

    execute_values는 %s 하나를 기준으로 SQL을 앞뒤로 가른다. 나중에 누가 LIKE '%%...' 같은
    리터럴 %를 넣으면 엉뚱한 데서 갈려 깨진다 — psycopg2 버전과 무관하게 이걸로 막는다.
    """
    assert _upsert_sql().count("%") == 1
    assert "VALUES %s" in _upsert_sql()


def test_execute_values_splits_the_sql_as_intended():
    """psycopg2가 실제로 쓰는 분해기로 앞뒤가 의도대로 갈리는지 확인한다."""
    _split_sql = pytest.importorskip("psycopg2.extras")._split_sql
    pre, post = _split_sql(_upsert_sql().encode())
    # 버전에 따라 bytes 조각 리스트로 온다.
    assert b"INSERT INTO policy_chunks" in b"".join(pre)
    # 충돌 갱신 절이 값 뒤쪽에 남아야 재적재(--overwrite)가 UPDATE로 동작한다.
    assert b"ON CONFLICT (chunk_id) DO UPDATE" in b"".join(post)


def test_row_matches_the_insert_column_list():
    """_row의 값 개수와 INSERT 컬럼 수가 어긋나면 런타임에야 터진다 — 여기서 잡는다."""
    sql = _upsert_sql()
    cols = sql.split("policy_chunks (")[1].split(")")[0]
    n_cols = len([c for c in re.split(r",\s*", cols.strip()) if c.strip()])
    assert n_cols == len(storage._row(_chunk("a")))


def test_dedupe_keeps_the_last_value():
    """executemany 시절 동작(뒤엣것이 이긴다)을 그대로 유지해야 한다."""
    out = storage._dedupe([_chunk("a", "옛것"), _chunk("b"), _chunk("a", "새것")])
    assert [c.chunk_id for c in out] == ["a", "b"]
    assert next(c for c in out if c.chunk_id == "a").content == "새것"


def test_dedupe_is_a_noop_without_duplicates():
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    assert storage._dedupe(chunks) == chunks


def test_row_passes_none_embedding_through():
    """임베딩이 없는 청크(--no-embed·boilerplate)도 저장돼야 한다 — 원문 보관이 목적."""
    row = storage._row(_chunk("a"))
    assert row[0] == "a"
    assert row[3] is None
