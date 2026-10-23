"""find_tables 테두리 오탐 — 테두리 안 본문이 통째로 유실되는 문제(이슈 #73).

손해보험 약관은 "보장종목/보상하는 사항" 류를 전폭 테두리에 수십 쪽 이어 쓴다.
find_tables가 그 테두리를 표로 잡으면 parse_text가 박스 안 줄을 전부 본문에서 빼서,
그 쪽은 조항 구조 없는 거대한 한 셀 표로만 남는다 — BM25·임베딩 대상에서 본문이 빠진다.

판정은 extractor.is_border_box 하나다. parse_text(본문 제외)와 extract_pymupdf(표 생성)가
같은 판정을 써야 본문·표 어느 쪽도 중복되거나 함께 사라지지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 내장 helv 폰트는 한글 글리프가 없어 추출 텍스트가 깨진다 — 검증 대상은 기하라 ASCII로 쓴다.
BODY = "the company pays the insured amount when an injury occurs"


@pytest.fixture
def bordered_body_pdf(tmp_path):
    """전폭 2열 테두리(보장종목 | 보상하는 사항) 안에 본문 문단 — 메리츠 p64 모양.

    find_tables가 이 테두리를 2x2(4셀) 표로 잡는 것을 테스트 안에서 전제로 확인한다 —
    안 잡히면 오탐 자체가 없어 검증이 공허해진다.
    """
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page()
    p.draw_rect(pymupdf.Rect(55, 57, 540, 785), color=(0, 0, 0), width=1)
    p.draw_line(pymupdf.Point(180, 57), pymupdf.Point(180, 785))
    p.draw_line(pymupdf.Point(55, 100), pymupdf.Point(540, 100))
    p.insert_text((70, 80), "coverage", fontsize=10)
    p.insert_text((200, 80), "what we cover", fontsize=10)
    for i in range(25):
        p.insert_text((200, 120 + i * 24), f"{BODY} ({i}).", fontsize=10)
    path = tmp_path / "bordered.pdf"
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture
def full_page_table_pdf(tmp_path):
    """전면을 덮지만 진짜인 22행 2열 분류표 — 면적 기준만으로 자르면 이게 죽는다."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page()
    for r in range(23):
        p.draw_line(pymupdf.Point(55, 60 + r * 31), pymupdf.Point(540, 60 + r * 31))
    for x in (55, 300, 540):
        p.draw_line(pymupdf.Point(x, 60), pymupdf.Point(x, 60 + 22 * 31))
    for r in range(22):
        p.insert_text((60, 80 + r * 31), f"I{r:02d}", fontsize=9)
        p.insert_text((305, 80 + r * 31), f"circulatory disease {r}", fontsize=9)
    path = tmp_path / "classification.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def _tables(pdf_path):
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        page = doc[0]
        return list(page.find_tables().tables), page.rect


# ── 판정 ──────────────────────────────────────────────────────────────────────

def test_a_full_width_border_with_few_cells_is_not_a_table(bordered_body_pdf):
    from insurance_chunker.extractor import is_border_box

    tables, rect = _tables(bordered_body_pdf)
    assert tables, "find_tables가 테두리를 잡아야 오탐 검증이 성립한다"
    assert all(is_border_box(t, rect) for t in tables)


def test_a_real_full_page_table_is_still_a_table(full_page_table_pdf):
    """분류표는 쪽을 덮어도 셀이 수십 개다 — 면적으로 자르면 진짜 표가 사라진다."""
    from insurance_chunker.extractor import is_border_box

    tables, rect = _tables(full_page_table_pdf)
    assert tables and len(tables[0].cells) >= 40
    assert not any(is_border_box(t, rect) for t in tables)


def test_one_or_two_cells_is_a_border_regardless_of_size(tmp_path):
    """작은 콜아웃 박스([유의사항]/[용어해설], 현대 p31)는 면적이 작아도 셀 1~2개다."""
    from insurance_chunker.extractor import is_border_box

    class _Rect:
        width, height = 595.0, 842.0

    class _Fake:
        def __init__(self, cells, bbox):
            self.cells, self.bbox = cells, bbox

    small_box = _Fake([(100, 100, 300, 200)], (100, 100, 300, 200))
    assert is_border_box(small_box, _Rect())
    two_cols = _Fake([(55, 57, 180, 785), (180, 57, 540, 785)], (55, 57, 540, 785))
    assert is_border_box(two_cols, _Rect())
    dense = _Fake([(0, 0, 10, 10)] * 12, (60, 80, 480, 200))
    assert not is_border_box(dense, _Rect())


# ── 본문·표 양쪽의 결과 ───────────────────────────────────────────────────────

def test_bordered_body_text_survives_in_parse_text(bordered_body_pdf):
    """고치기 전에는 박스 안 줄 중심점이 전부 테두리 안이라 본문이 통째로 빠졌다."""
    from insurance_chunker.pdf_parser import parse_text

    pages, _ = parse_text(bordered_body_pdf)
    assert BODY in pages[0].text
    assert "coverage" in pages[0].text


def test_bordered_body_does_not_become_a_table(bordered_body_pdf):
    """본문을 살렸으면 같은 내용의 한 셀 표는 나오면 안 된다 — 중복 적재가 된다."""
    import pymupdf

    from insurance_chunker.extractor import extract_pymupdf

    with pymupdf.open(bordered_body_pdf) as doc:
        assert extract_pymupdf(doc[0]) is None


def test_real_full_page_table_is_extracted_and_excluded_from_text(full_page_table_pdf):
    """진짜 표는 지금처럼: 표로 나오고, 그 영역 텍스트는 본문에서 빠진다."""
    import pymupdf

    from insurance_chunker.extractor import extract_pymupdf
    from insurance_chunker.pdf_parser import parse_text

    with pymupdf.open(full_page_table_pdf) as doc:
        md = extract_pymupdf(doc[0])
    assert md is not None and "circulatory disease 5" in md
    pages, _ = parse_text(full_page_table_pdf)
    assert "circulatory disease 5" not in pages[0].text
