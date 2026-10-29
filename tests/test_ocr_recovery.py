"""텍스트 레이어 없는 쪽의 본문 복구(이슈 #74) 계약.

스캔·이미지 쪽은 parse_text가 빈 문자열을 내고, 예전 경로의 VLM은 그 쪽을 gap 1순위로
받아 '표만' 물었다 — 본문은 어디에서도 복구되지 않은 채 적재는 OK로 집계됐다. 레이아웃
경로에서는 그 쪽을 가려내(needs_ocr) 쪽 전체를 OCR 프롬프트로 보내 본문을 채운다.

복구된 쪽은 폰트 신호가 없어 경계 검출이 아무 경계도 못 만든다 — 라벨은 label_for가
앞 경계를 그대로 잇고, is_ocr=True로 그 사실을 하류에 남긴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_layout_crop import fake_layout  # noqa: E402,F401 - 픽스처 공유
from test_stages import ex  # noqa: E402,F401 - 픽스처 공유

RECOVERED = "제3조(보험금의 지급) 회사는 피보험자가 상해를 입은 경우 보험금을 지급한다."


@pytest.fixture
def scan_pdf(tmp_path):
    """p1 본문 / p2 전면 이미지(텍스트 레이어 없음) / p3 본문 — 운영 스캔본의 축소판."""
    import pymupdf
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((70, 100), "page one body", fontsize=11)

    src = pymupdf.open()
    sp = src.new_page()
    sp.insert_text((70, 100), "scanned body", fontsize=20)
    full = src[0].get_pixmap(dpi=72).tobytes("png")
    src.close()
    p2 = doc.new_page()
    p2.insert_image(p2.rect, stream=full)  # 쪽 전체를 덮는 이미지, 텍스트 없음

    p3 = doc.new_page()
    p3.insert_text((70, 100), "page three body", fontsize=11)
    path = tmp_path / "scan.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def _ocr_server(monkeypatch, body=RECOVERED):
    """프롬프트별로 응답을 가른다 — OCR 프롬프트에만 본문을 준다."""
    import requests

    seen = {"prompts": []}

    class _Resp:
        status_code = 200

        def __init__(self, content):
            self._content = content

        def json(self):
            return {"choices": [{"message": {"content": self._content},
                                 "finish_reason": "stop"}]}

        def raise_for_status(self):
            pass

    def _post(url, json=None, timeout=None):
        prompt = json["messages"][0]["content"][1]["text"]
        seen["prompts"].append(prompt)
        from insurance_chunker import extractor
        if prompt == extractor.VLM_OCR_PROMPT:
            return _Resp(body)
        return _Resp("| a | b |\n|---|---|\n| 1 | 2 |")

    monkeypatch.setattr(requests, "post", _post)
    return seen


def _parse(pdf):
    from insurance_chunker import stages
    pages, results = stages.parse(pdf)
    return pages, {r.name: r for r in results}


# ── 가려내기 ──────────────────────────────────────────────────────────────────

def test_needs_ocr_only_for_page_covering_images_without_text(scan_pdf):
    import pymupdf

    from insurance_chunker.layout import needs_ocr
    with pymupdf.open(scan_pdf) as doc:
        assert not needs_ocr(doc[0]), "텍스트가 있는 쪽은 복구 대상이 아니다"
        assert needs_ocr(doc[1])
        assert not needs_ocr(doc[2])


def test_blank_page_is_not_an_ocr_target(tmp_path):
    """커버 이미지 없이 그냥 빈 쪽은 복구할 것도 없다 — 호출만 는다."""
    import pymupdf

    from insurance_chunker.layout import needs_ocr
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(path)
    doc.close()
    with pymupdf.open(str(path)) as d:
        assert not needs_ocr(d[0])


# ── 복구 배선 ─────────────────────────────────────────────────────────────────

def test_scan_page_body_is_recovered_and_marked(ex, fake_layout, scan_pdf, monkeypatch):  # noqa: F811
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({})
    seen = _ocr_server(monkeypatch)
    pages, by = _parse(scan_pdf)

    from insurance_chunker import extractor
    assert seen["prompts"] == [extractor.VLM_OCR_PROMPT], "스캔 쪽 하나 = OCR 호출 하나"
    assert pages[1].text == RECOVERED
    assert pages[1].is_ocr is True
    assert not pages[1].tables, "복구 본문이 표로 들어가면 표 row 청크로 쪼개진다"
    assert pages[0].is_ocr is False and pages[2].is_ocr is False
    assert by["layout"].metrics["ocr_pages"] == 1
    assert by["vlm"].metrics["ocr_targets"] == 1
    assert by["vlm"].metrics["ocr_recovered"] == 1
    assert by["combine"].metrics["ocr_recovered_pages"] == 1


def test_table_boxes_on_a_scan_page_still_use_the_table_prompt(ex, fake_layout, scan_pdf,  # noqa: F811
                                                               monkeypatch):
    """같은 쪽에서 표 영역은 표 프롬프트, 쪽 전체는 OCR 프롬프트 — 둘은 딴 호출이다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({2: [[60.0, 80.0, 300.0, 200.0]]})
    seen = _ocr_server(monkeypatch)
    pages, by = _parse(scan_pdf)

    from insurance_chunker import extractor
    assert sorted(seen["prompts"]) == sorted([extractor.VLM_OCR_PROMPT, extractor.VLM_PROMPT])
    assert pages[1].text == RECOVERED and pages[1].is_ocr
    assert pages[1].tables and pages[1].tables[0]["source"] == "vlm"


def test_recovery_consumes_the_crop_budget_first(ex, fake_layout, scan_pdf, monkeypatch):  # noqa: F811
    """표에는 PyMuPDF 폴백이 있지만 이 쪽의 본문은 이것 말고 어디에도 없다 — 예산 1순위."""
    ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_CROPS="1")
    fake_layout({1: [[60.0, 80.0, 300.0, 200.0]]})  # 표 박스도 하나 있지만 예산은 1
    seen = _ocr_server(monkeypatch)
    pages, _ = _parse(scan_pdf)

    from insurance_chunker import extractor
    assert seen["prompts"] == [extractor.VLM_OCR_PROMPT]
    assert pages[1].text == RECOVERED


def test_truncated_ocr_output_is_discarded(ex, fake_layout, scan_pdf, monkeypatch):  # noqa: F811
    """잘린 본문을 절반만 싣면 조항이 중간에서 끊긴 채 색인된다 — 버리고 빈 쪽으로 둔다."""
    import requests

    class _Cut:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "제1조 " * 50},
                                 "finish_reason": "length"}]}

        def raise_for_status(self):
            pass

    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({})
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Cut())
    pages, by = _parse(scan_pdf)
    assert pages[1].text == ""
    assert by["vlm"].metrics["ocr_recovered"] == 0
    assert by["vlm"].metrics["outcomes"].get("truncated") == 1


def test_pages_with_a_text_layer_are_never_overwritten(ex, fake_layout, scan_pdf, monkeypatch):  # noqa: F811
    """복구는 빈 쪽에만 — 멀쩡한 텍스트 레이어를 VLM 출력으로 덮으면 오독이 본문을 오염시킨다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({})
    _ocr_server(monkeypatch)
    pages, _ = _parse(scan_pdf)
    assert "page one body" in pages[0].text
    assert "page three body" in pages[2].text


# ── 프롬프트 선택 ─────────────────────────────────────────────────────────────

def test_paddle_model_uses_the_ocr_task_token(ex, monkeypatch):  # noqa: F811
    monkeypatch.delenv("VLM_OCR_PROMPT", raising=False)
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B")
    assert extractor.VLM_OCR_PROMPT == "OCR:"


def test_general_vlm_gets_an_instruct_ocr_prompt(ex, monkeypatch):  # noqa: F811
    monkeypatch.delenv("VLM_OCR_PROMPT", raising=False)
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="general-vlm:8b-instruct")
    assert "원문 그대로" in extractor.VLM_OCR_PROMPT


# ── 라벨 잇기 ─────────────────────────────────────────────────────────────────

def test_recovered_page_inherits_the_previous_section_label():
    """복구된 쪽은 폰트 신호가 없어 경계가 안 생긴다 — label_for가 앞 경계를 그대로 잇는다.

    경계 검출은 PDF 폰트 스팬을 읽으므로(PageResult.text가 아니라) 복구 본문이 경계를
    만들거나 깨뜨릴 수 없다. 그 쪽의 조항은 직전 특약의 라벨로 들어간다.
    """
    from insurance_chunker.boundaries import Boundary, label_for

    bounds = [Boundary(1, "암 진단특별약관", "yak"), Boundary(5, "다음 특약", "yak")]
    assert label_for(bounds, 3) == ("암 진단특별약관", "yak")
