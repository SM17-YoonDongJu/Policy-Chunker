"""청크 뷰어 — 화면이 데이터를 맞게 잇는지(쪽 ↔ 청크, 검색 결과 ↔ 문서)와 운영 검색 래퍼의 계약.

DB·S3·청커·임베딩 없이 돈다. 데이터 계층(queries·files·chunker_status·prod_search)을 가짜로
바꾸고 Streamlit AppTest로 화면을 실제로 실행한다. 로컬 DB에 붙여 보는 절차는 viewer/README.md.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP = str(ROOT / "viewer" / "app.py")

DOC = "a" * 64
META = {"doc_hash": DOC, "insurer": "테스트생명", "product_name": "실손의료보험",
        "product_code": None,
        "effective_date": None, "generation": None, "ingested_at": "2026-10-30T01:00:00+00:00",
        "chunks": 3, "last_page": 3}
CHUNKS = [
    {"chunk_id": "c1", "chunk_index": 1, "page_number": 1, "chunk_type": "definition",
     "section": "제1관", "article_number": "제1조", "article_title": "목적", "token_count": 40,
     "content": "제1조(목적) 이 계약은 골절을 보장합니다.", "table_id": None,
     "row_start": None, "row_end": None},
    {"chunk_id": "c2", "chunk_index": 2, "page_number": 1, "chunk_type": "coverage",
     "section": None, "article_number": "제2조", "article_title": None, "token_count": 50,
     "content": "제2조 보험금의 지급사유", "table_id": None, "row_start": None, "row_end": None},
    {"chunk_id": "c3", "chunk_index": 3, "page_number": 3, "chunk_type": "coverage",
     "section": None, "article_number": "제5조", "article_title": "지급기준표", "token_count": 70,
     "content": "골절진단비 30만원", "table_id": "t-1", "row_start": 1, "row_end": 2},
]
DOC_ROW = {"doc_hash": DOC, "insurer": "테스트생명", "product_name": "실손의료보험",
           "ingested_at": "2026-10-30T01:00:00+00:00", "chunks": 3, "pages_with_chunks": 2,
           "last_page": 3, "file_name": "실손약관.pdf", "part_order": 0, "part_total": 2}
STATUS = {
    "generated_at": "2026-10-30T01:10:00+00:00", "interval_s": 604800,
    "daemon": {"started_at": "2026-10-30T00:00:00+00:00", "last_run_ok": True,
               "last_run_at": "2026-10-23T00:00:00+00:00",
               "last_success_at": "2026-10-23T00:00:00+00:00"},
    "run": {"state": "running", "done": 1, "pending": 3, "total": 10, "skipped": 7,
            "quarantined": 0, "concurrency": 1, "started_at": "2026-10-30T01:00:00+00:00",
            "finished_at": None, "stopped": False},
    "processing": [{"sha256": "b" * 64, "name": "big.pdf",
                    "started_at": "2026-10-30T01:05:00+00:00", "attempts": 1}],
    "unfinished": [],
    "recent": [{"at": "2026-10-30T01:04:00+00:00", "sha256": DOC, "name": "실손약관.pdf",
                "status": "OK", "chunks": 3, "elapsed_s": 40, "insurer": "테스트생명"},
               {"at": "2026-10-30T01:02:00+00:00", "sha256": "c" * 64, "name": "broken.pdf",
                "status": "ERROR", "chunks": 0, "elapsed_s": 1, "error": "PDF 파싱 실패"}],
}


def _pdf(path: Path, pages: int) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    return path


@pytest.fixture
def fake(monkeypatch, tmp_path):
    from viewer import chunker_status, data, env, files, queries

    st.cache_data.clear()
    st.cache_resource.clear()
    # 로컬 .env가 무엇을 들고 있든 같은 조건에서 돈다
    monkeypatch.setattr(env, "S3_BUCKET", "")
    monkeypatch.setattr(env, "PDF_DIR", None)
    pdf = _pdf(tmp_path / "doc.pdf", 3)
    monkeypatch.setattr(queries, "catalog_access", lambda: True)
    monkeypatch.setattr(queries, "recent_documents", lambda **kw: [DOC_ROW])
    monkeypatch.setattr(queries, "insurers", lambda: ["테스트생명"])
    monkeypatch.setattr(queries, "chunk_types", lambda: ["coverage", "definition"])
    monkeypatch.setattr(queries, "document_meta", lambda h: META if h == DOC else None)
    monkeypatch.setattr(queries, "document_chunks", lambda h: CHUNKS if h == DOC else [])
    monkeypatch.setattr(queries, "source_file", lambda h: {
        "s3_key": f"corpus/terms/{h}.pdf", "file_name": "실손약관.pdf", "part_order": 0,
        "part_total": 2, "company": "테스트생명", "product_name": "실손의료보험"})
    monkeypatch.setattr(queries, "chunk_locations", lambda ids: {
        i: {"chunk_id": i, "doc_hash": DOC, "page_number": 3, "chunk_index": 3,
            "insurer": "테스트생명", "product_name": "실손의료보험"} for i in ids if i == "c3"})
    monkeypatch.setattr(queries, "find_phrase", lambda text, **kw: [
        {**CHUNKS[0], "doc_hash": DOC, "insurer": "테스트생명", "product_name": "실손의료보험",
         "ingested_at": None}])
    monkeypatch.setattr(files, "pdf_path", lambda h, key: pdf)
    monkeypatch.setattr(files, "table_markdown",
                        lambda tid: "| 구분 | 금액 |\n|---|---|\n| 골절 | 30만원 |")
    monkeypatch.setattr(chunker_status, "fetch", lambda url, **kw: STATUS)
    return types.SimpleNamespace(pdf=pdf, data=data, files=files, queries=queries,
                                 chunker_status=chunker_status)


def _run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _open_document(doc: str = DOC) -> AppTest:
    at = _run(AppTest.from_file(APP, default_timeout=30))
    at.switch_page("views/document.py")
    at.query_params["doc"] = doc
    return _run(at)


def _page_heading(at: AppTest) -> str:
    return next(m.value for m in at.markdown if "쪽에서 나온 청크" in m.value)


# ── 처리 현황 ─────────────────────────────────────────────────────────────────

def test_status_page_shows_the_running_cycle(fake):
    at = _run(AppTest.from_file(APP, default_timeout=30))
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["이번 실행"] == "처리 중" and metrics["처리"] == "1 / 3건"
    processing, recent, docs = (df.value for df in at.dataframe)
    assert list(processing["문서"]) == ["big.pdf"]
    # 적재된 결과만 문서 보기로 열 수 있다 — 실패 문서는 청크가 없다
    assert list(recent["열기"]) == [f"document?doc={DOC}", ""]
    assert list(docs["파트"]) == ["1/2"]  # part_order는 0부터 센다


def test_status_page_survives_without_the_chunker(fake, monkeypatch):
    def down(url, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(fake.chunker_status, "fetch", down)
    at = _run(AppTest.from_file(APP, default_timeout=30))
    assert any("청커 상태를 가져오지 못했어요" in i.value for i in at.info)
    assert len(at.dataframe) == 1  # 적재된 문서 목록은 그대로 보인다


# ── 문서 보기 ─────────────────────────────────────────────────────────────────

def test_document_page_pairs_the_page_with_its_chunks(fake):
    at = _open_document()
    assert at.title[0].value == "실손약관.pdf"
    assert _page_heading(at) == "**1쪽에서 나온 청크 2개**"
    assert len(at.code) == 2 and len(at.get("image")) == 1
    assert "파트 1/2" in at.caption[0].value
    assert any("청크가 0개인 쪽 1개" in m.value for m in at.markdown)

    next_button = next(b for b in at.button if b.label == "다음 ▶")
    next_button.click()
    _run(at)
    assert _page_heading(at) == "**2쪽에서 나온 청크 0개**"
    assert any("이 쪽에서 나온 청크가 없어요" in x.value for x in at.warning)


def test_only_chunked_pages_skips_empty_ones_and_shows_tables(fake):
    at = _open_document()
    at.toggle[0].set_value(True)
    next(b for b in at.button if b.label == "다음 ▶").click()
    _run(at)
    assert _page_heading(at) == "**3쪽에서 나온 청크 1개**"
    assert "표 원문 — 1~2행" in [e.label for e in at.expander]
    assert any(m.value.startswith("| 구분 | 금액 |") for m in at.markdown)


def test_document_page_without_a_pdf_still_lists_chunks(fake, monkeypatch):
    monkeypatch.setattr(fake.files, "pdf_path", lambda h, key: None)
    at = _open_document()
    assert any("원본을 받을" in i.value for i in at.info)  # S3 버킷 미설정 안내
    assert len(at.get("image")) == 0 and len(at.code) == 2


def test_unknown_document_is_reported(fake):
    at = _open_document("f" * 64)
    assert any("청크가 없어요" in e.value for e in at.error)


# ── 검색 ──────────────────────────────────────────────────────────────────────

class _FakeProdSearch:
    def search(self, query, **kw):
        from viewer.prod_search import ProdSearchResult

        return ProdSearchResult(
            chunks=[{"text": "골절진단비 30만원", "namespace": "terms", "score": 0.0331,
                     "source_ref": "c3", "article_number": "제5조", "chunk_type": "coverage"},
                    {"text": "대법원 판례", "namespace": "case", "score": 0.0164,
                     "source_ref": "case-1", "article_number": None, "chunk_type": None}],
            citations=[{"clause_no": "제5조", "exhibit": None}], namespaces=["terms", "case"],
            in_scope=True, reason=None, elapsed_ms=12, keyword_query="골절 보험금",
            embed_text=query, corrections=[{"original": "골적", "canonical": "골절"}],
            vector_used=True)


def test_prod_search_results_link_back_to_their_pages(fake, monkeypatch):
    monkeypatch.setattr(fake.data, "prod_search", lambda: _FakeProdSearch())
    at = _run(AppTest.from_file(APP, default_timeout=30))
    at.switch_page("views/search.py")
    _run(at)
    at.text_input[0].input("골적 보험금")
    next(b for b in at.button if b.label == "검색").click()
    _run(at)
    ranks = [m.value for m in at.markdown if "RRF" in m.value]
    assert len(ranks) == 2 and ranks[0].startswith("**1** · RRF 0.0331")
    # 약관 결과만 문서·쪽으로 이어진다(판례는 policy_chunks가 아니다)
    assert len(at.get("page_link")) == 1
    captions = " ".join(c.value for c in at.caption)
    assert "벡터+키워드" in captions and "골적 → 골절" in captions


def test_phrase_search_highlights_the_match(fake):
    at = _run(AppTest.from_file(APP, default_timeout=30))
    at.switch_page("views/search.py")
    _run(at)
    phrase = next(t for t in at.text_input if t.label == "문구")
    phrase.input("골절")
    next(b for b in at.button if b.label == "찾기").click()
    _run(at)
    assert any("**골절**" in m.value for m in at.markdown)


# ── 운영 검색 래퍼 ────────────────────────────────────────────────────────────

def test_prod_dsn_forces_read_only_and_the_corpus_schema():
    from viewer.prod_search import prod_dsn

    assert prod_dsn("postgresql://u@h/db") == \
        "postgresql://u@h/db?default_transaction_read_only=on&search_path=corpus,public"
    # 이미 정한 search_path는 존중한다
    assert prod_dsn("postgresql://u@h/db?sslmode=require&search_path=x") == \
        "postgresql://u@h/db?sslmode=require&search_path=x&default_transaction_read_only=on"
    assert prod_dsn("") == ""


_FAKE_RAG = '''
import asyncio

async def correct_query(pool, query):
    return types.SimpleNamespace(keyword_query=query + "!", embed_text=query, corrections=[])

async def _embed_query(text):
    await asyncio.sleep(0)
    return None if "fail" in text else [0.1]

async def search(query):
    corrected = await correct_query(None, query)
    task = asyncio.create_task(_embed_query(corrected.embed_text))  # 운영처럼 별도 태스크
    await asyncio.sleep(0)
    return await task
'''


def test_observers_keep_concurrent_requests_apart():
    """운영 함수를 감싸 엿본 값이 요청끼리 섞이면 화면이 남의 결과를 설명한다."""
    from viewer import prod_search as ps

    mod = types.ModuleType("fake_rag_search")
    mod.types = types
    exec(_FAKE_RAG, mod.__dict__)
    ps._install_observers(mod)
    ps._install_observers(mod)  # 두 번 감싸지 않는다

    async def one(query: str) -> dict:
        box: dict = {}
        ps._observed.set(box)
        await mod.search(query)
        return box

    async def main() -> list[dict]:
        return await asyncio.gather(one("ok"), one("fail"))

    ok, failed = asyncio.run(main())
    assert ok["vector_used"] is True and failed["vector_used"] is False
    assert ok["corrected"].keyword_query == "ok!" and failed["corrected"].keyword_query == "fail!"


def test_ai_engine_ref_reports_the_pinned_commit(monkeypatch):
    from viewer import prod_search as ps

    sha = "d4e3742dda0954a536263df610d1df736615ede2"

    class Dist:
        def read_text(self, name):
            return json.dumps({"url": f"https://github.com/x/AI/archive/{sha}.tar.gz"})

    monkeypatch.setattr(ps.metadata, "distribution", lambda name: Dist())
    assert ps.ai_engine_ref() == "d4e3742"

    def missing(name):
        raise ps.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ps.metadata, "distribution", missing)
    assert ps.ai_engine_ref() == "설치 안 됨"


# ── 원본 파일·조회 헬퍼 ───────────────────────────────────────────────────────

def test_local_pdf_is_found_by_its_hash(tmp_path, monkeypatch):
    from viewer import env, files

    pdf = _pdf(tmp_path / "아무_이름.pdf", 2)
    monkeypatch.setattr(env, "PDF_DIR", str(tmp_path))
    monkeypatch.setattr(files, "_pdf_index", None)
    assert files.pdf_path(files.sha256_of(pdf), None) == pdf
    assert files.pdf_path("0" * 64, None) is None
    assert files.render_page(pdf, 2, 50).startswith(b"\x89PNG")
    assert files.page_count(pdf) == 2


def test_pdf_cache_keeps_only_the_newest(tmp_path, monkeypatch):
    import os

    from viewer import env, files

    monkeypatch.setattr(env, "CACHE_MAX_PDFS", 2)
    for i in range(4):
        p = tmp_path / f"{i}.pdf"
        p.write_bytes(b"%PDF")
        os.utime(p, (i, i))
    files._prune(tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["2.pdf", "3.pdf"]


def test_non_pdf_names_are_not_opened():
    from viewer.files import is_pdf_name

    assert is_pdf_name("약관.PDF") and is_pdf_name(None)
    assert not is_pdf_name("약관.hwp.zip")


def test_like_pattern_matches_wildcards_literally():
    from viewer.queries import like_pattern

    assert like_pattern("100%_a\\b") == "%100\\%\\_a\\\\b%"


def test_connection_is_read_only_and_verifies_rds_when_a_ca_is_given(monkeypatch):
    from viewer import env, queries

    monkeypatch.setattr(env, "RDS_CA_PATH", None)
    kw = queries.connect_kwargs()
    assert "default_transaction_read_only=on" in kw["options"] and "sslmode" not in kw
    monkeypatch.setattr(env, "RDS_CA_PATH", "/etc/ssl/rds/global-bundle.pem")
    kw = queries.connect_kwargs()
    assert kw["sslmode"] == "verify-full" and kw["sslrootcert"].endswith("global-bundle.pem")


def test_snippet_bolds_the_match_and_escapes_markdown():
    from viewer.widgets import md_escape, snippet

    assert md_escape("제1조(목적)*") == "제1조\\(목적\\)\\*"
    out = snippet("앞부분 " * 30 + "골절진단비 지급" + " 뒷부분" * 30, "골절진단비", width=10)
    assert "**골절진단비**" in out and out.startswith("…") and out.endswith("…")
