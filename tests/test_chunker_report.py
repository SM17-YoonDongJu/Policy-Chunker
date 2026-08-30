"""경계 신뢰도가 청킹 밖으로 나오는 배선 — 이 dict 하나가 지표·SLO의 유일한 입력이다.

chunk_document은 신뢰도를 반환값이 아니라 out-param dict(`report`)로 올린다. 그래서
배선이 끊겨도 청킹 결과는 멀쩡해 보이고, 지표만 조용히 비거나 문서가 통째로 죽는다.
실제로 그랬다 — rechunk.report(통계 함수)와 out-param 이름이 겹쳐 `report[...] = level`이
함수에 대입을 시도했고, policy_terms PDF가 100% TypeError로 죽었다. import 스모크는
통과하는 종류의 버그라 여기서 잡는다.

값 자체도 계약이다. exporter.py는 boundary_confidence == "weak"를 세고,
ingest_many.py는 이 dict에서 그대로 꺼내 runlog에 적는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from insurance_chunker import boundaries, rechunk, toc  # noqa: E402
from insurance_chunker.chunker import chunk_document  # noqa: E402
from insurance_chunker.models import DocMeta, PageResult  # noqa: E402


class _FakeDoc:
    """fitz.open()의 컨텍스트 매니저 자리만 채운다 — 내용은 find/assess를 스텁해 안 쓴다."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _meta() -> DocMeta:
    return DocMeta(source_pdf="terms.pdf", doc_hash="a" * 16, doc_type="policy_terms",
                   insurer="삼성화재", product_name="다이렉트 실손의료비보험")


def _pages() -> list[PageResult]:
    p1 = "제1조(목적) 이 약관은 보험계약에 관한 사항을 정한다.\n" * 6
    p2 = "제2조(용어의 정의) 이 약관에서 쓰는 용어의 뜻은 다음과 같다.\n" * 6
    return [PageResult(page_num=1, text=p1), PageResult(page_num=2, text=p2)]


@pytest.fixture
def font_detection(monkeypatch):
    """폰트 경계 감지를 스텁한다. 반환할 신뢰도는 테스트가 set()으로 정한다.

    PDF를 실제로 열지 않는 이유: 여기서 보려는 건 조판 판독이 아니라 그 결과가 report로
    나가는 배선이다. 진짜 약관 조판을 합성하면 boundaries의 휴리스틱에 테스트가 묶인다.
    """
    bounds = [boundaries.Boundary(page=1, label="보통약관", kind="base")]
    det = boundaries.Detection(body_size=10.0, title_size=13.0, title_lo=12.0, title_hi=14.0,
                               base_start_page=1, front_end_page=1, product="실손")
    state: dict = {"level": "ok", "reasons": ["제1조 p1, 경계 1개"]}

    monkeypatch.setattr(pymupdf, "open", lambda path: _FakeDoc())
    monkeypatch.setattr(boundaries, "find", lambda doc, d=None: (bounds, det))
    monkeypatch.setattr(boundaries, "assess",
                        lambda doc, d=None, b=None: (state["level"], state["reasons"]))
    monkeypatch.setattr(toc, "extract_toc_titles_ordered", lambda doc, page: [])

    state["bounds"] = bounds
    state["set"] = lambda level, reasons: state.update(level=level, reasons=reasons)
    return state


def test_confidence_reaches_the_caller(font_detection):
    """성공 경로에서 report가 채워진다 — 이게 안 되면 지표가 전부 None이다."""
    report: dict = {}
    chunks, _ = chunk_document(_pages(), _meta(), pdf_path="terms.pdf", report=report)

    assert chunks
    assert report["boundary_confidence"] == "ok"
    assert report["boundaries"] == 1
    assert report["boundary_reasons"] == ["제1조 p1, 경계 1개"]


def test_weak_is_reported_verbatim(font_detection):
    """exporter가 세는 문자열은 정확히 "weak"다 — 다른 표기로 바뀌면 카운터가 0이 된다."""
    font_detection["set"]("weak", ["제목 폰트를 못 찾음", "제1조를 못 찾음"])
    report: dict = {}
    chunks, _ = chunk_document(_pages(), _meta(), pdf_path="terms.pdf", report=report)

    # 낮은 신뢰도는 실패가 아니다 — 적재는 계속하고 사람이 보게 지표로만 올린다.
    assert chunks
    assert report["boundary_confidence"] == "weak"
    assert len(report["boundary_reasons"]) == 2


def test_detection_failure_is_recorded_not_raised(font_detection, monkeypatch):
    """감지가 터져도 문서는 폴백 적재된다. 예외를 밖으로 흘리면 배치 1건이 통째로 ERROR다."""
    monkeypatch.setattr(boundaries, "find",
                        lambda doc, d=None: (_ for _ in ()).throw(RuntimeError("폰트 테이블 손상")))
    report: dict = {}
    chunks, _ = chunk_document(_pages(), _meta(), pdf_path="terms.pdf", report=report)

    assert chunks, "경계를 못 잡아도 단순 텍스트 청킹으로 적재는 돼야 한다"
    assert report["boundary_confidence"] == "error"
    assert "RuntimeError" in report["boundary_reasons"][0]


def test_report_is_optional(font_detection):
    """report를 안 넘기는 호출부(ingest.py, eval/*)도 그대로 돌아야 한다."""
    chunks, _ = chunk_document(_pages(), _meta(), pdf_path="terms.pdf")
    assert chunks


def test_stats_call_is_not_shadowed_by_the_report_dict(font_detection, monkeypatch):
    """청킹 통계는 rechunk.report(함수)로 가야 한다.

    out-param과 이름이 겹치면 둘 중 하나가 반드시 틀린 타입으로 불린다 — dict를 호출하거나
    함수에 대입하거나. 통계 함수가 (chunks, bounds)로 실제 호출됐는지로 갈라졌음을 확인한다.
    """
    seen: list[tuple] = []
    real = rechunk.report

    def spy(chunks, bounds):
        seen.append((chunks, bounds))
        return real(chunks, bounds)

    monkeypatch.setattr(rechunk, "report", spy)
    report: dict = {}
    chunk_document(_pages(), _meta(), pdf_path="terms.pdf", report=report)

    assert len(seen) == 1
    got_chunks, got_bounds = seen[0]
    assert got_chunks and got_bounds == font_detection["bounds"]
    assert report["boundary_confidence"] == "ok"
