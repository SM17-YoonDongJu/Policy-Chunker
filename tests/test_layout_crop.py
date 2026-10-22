"""레이아웃 crop 경로(이슈 #72) 계약.

페이지 전체를 VLM에 넣는 기존 경로는 백지에서 반복 루프가 돌고 본문이 표로 둔갑했다.
새 경로는 레이아웃이 잡은 표 박스만 잘라 넣는다: 텍스트의 표 영역 제외 기준도, VLM 대상도,
combine의 선택 규칙도 박스가 기준이다.

paddleocr는 테스트 환경에 없다 — 박스 탐지(page_table_boxes)를 가짜로 끼워 단계 배선을
검증하고, 모델이 없을 때 기존 경로로 떨어지는 것 자체도 계약으로 고정한다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_stages import _as_tuples, ex, fake_vlm, mixed_pdf  # noqa: E402,F401 - 픽스처 공유

# p1 괘선 표의 위치 — 픽스처와 박스가 같은 좌표를 봐야 한다.
TABLE_RECT = [55.0, 70.0, 490.0, 210.0]
BODY_LINE = "ordinary policy text outside the table"


@pytest.fixture
def crop_pdf(tmp_path):
    """p1: 괘선 표(상단) + 본문 한 줄(하단). p2: 괘선 표만(레이아웃이 놓친 쪽 역할)."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page()
    for r in range(4):
        for c in range(3):
            rect = pymupdf.Rect(60 + c * 140, 80 + r * 28, 60 + (c + 1) * 140, 80 + (r + 1) * 28)
            p.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p.insert_text((rect.x0 + 5, rect.y0 + 18), f"r{r}c{c}", fontsize=9)
    p.insert_text((70, 300), BODY_LINE, fontsize=10)

    p2 = doc.new_page()
    for r in range(3):
        for c in range(2):
            rect = pymupdf.Rect(60 + c * 140, 80 + r * 28, 60 + (c + 1) * 140, 80 + (r + 1) * 28)
            p2.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p2.insert_text((rect.x0 + 5, rect.y0 + 18), f"q{r}c{c}", fontsize=9)
    path = tmp_path / "crop.pdf"
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture
def fake_layout(monkeypatch):
    """page_table_boxes를 가짜 박스로 바꾼다. 모델 로드는 일어나지 않는다."""
    from insurance_chunker import layout as layout_mod

    def _install(boxes_by_page: dict[int, list[list[float]]]):
        layout_mod.reset_for_tests()
        monkeypatch.setattr(layout_mod, "LAYOUT_BACKEND", "paddle")
        monkeypatch.setattr(
            layout_mod, "page_table_boxes",
            lambda page: [{"bbox_pdf": list(b),
                           "bbox_px": [v * layout_mod.LAYOUT_DPI / 72 for v in b],
                           "score": 0.9}
                          for b in boxes_by_page.get(page.number + 1, [])])
        return layout_mod
    yield _install
    from insurance_chunker import layout as layout_mod2
    layout_mod2.reset_for_tests()


def _parse(pdf):
    from insurance_chunker import stages
    pages, results = stages.parse(pdf)
    by = {r.name: r for r in results}
    return pages, by


# ── 배선: 박스가 텍스트 제외·VLM 대상·combine을 전부 끌고 간다 ─────────────────

def test_text_excludes_by_layout_box_not_find_tables(ex, fake_vlm, fake_layout, crop_pdf):  # noqa: F811
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({1: [TABLE_RECT]})
    pages, by = _parse(crop_pdf)

    assert by["layout"].metrics["enabled"] is True
    assert "r0c0" not in pages[0].text, "박스 안 셀 텍스트는 본문에서 빠져야 한다"
    assert BODY_LINE in pages[0].text, "박스 밖 본문은 남아야 한다"
    # p2는 박스가 없다 — find_tables로 돌아가지 않으므로 표 셀 텍스트가 본문에 남는다
    assert "q0c0" in pages[1].text


def test_vlm_gets_one_call_per_box_with_clip(ex, fake_vlm, fake_layout, crop_pdf, monkeypatch):  # noqa: F811
    from insurance_chunker import extractor
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({1: [TABLE_RECT]})
    clips = []
    orig = extractor.render_page

    def _spy(page, pno, clip=None):
        clips.append(clip)
        return orig(page, pno, clip)

    monkeypatch.setattr(extractor, "render_page", _spy)
    _, by = _parse(crop_pdf)
    assert fake_vlm["n"] == 1, "박스 하나 = 호출 하나"
    assert clips == [TABLE_RECT]
    assert by["vlm"].metrics["targets"] == 1


def test_clip_render_is_smaller_than_the_page(ex, fake_layout, crop_pdf):  # noqa: F811
    """crop은 쪽 전체가 아니라 영역만 렌더해야 입력 토큰이 준다 — 전환의 핵심 전제."""
    import pymupdf

    from insurance_chunker import extractor
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    with pymupdf.open(crop_pdf) as doc:
        full = extractor.render_page(doc[0], 1)
        cropped = extractor.render_page(doc[0], 1, TABLE_RECT)
    assert cropped is not None and full is not None
    assert cropped[1] < full[1] * 0.5, "crop 렌더가 쪽 전체의 절반보다 커선 안 된다(이 픽스처 기준)"


def test_combine_takes_vlm_on_box_pages_and_pymupdf_elsewhere(ex, fake_vlm,  # noqa: F811
                                                              fake_layout, crop_pdf):
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({1: [TABLE_RECT]})
    pages, by = _parse(crop_pdf)

    assert pages[0].tables and pages[0].tables[0]["source"] == "vlm"
    assert pages[1].tables and pages[1].tables[0]["source"] == "pymupdf", \
        "박스가 없는 쪽은 PyMuPDF 폴백이어야 한다 — 레이아웃이 놓친 표가 사라지면 안 된다"


def test_empty_vlm_result_falls_back_to_pymupdf(ex, fake_layout, crop_pdf, monkeypatch):  # noqa: F811
    """퇴행 폐기·차단기로 VLM이 빈 쪽은 PyMuPDF 표가 남아야 한다 — 전부 잃는 게 아니다."""
    import requests

    class _Empty:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

        def raise_for_status(self):
            pass

    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({1: [TABLE_RECT]})
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Empty())
    pages, _ = _parse(crop_pdf)
    assert pages[0].tables and pages[0].tables[0]["source"] == "pymupdf"


def test_two_boxes_on_a_page_join_in_reading_order(ex, fake_layout, crop_pdf, monkeypatch):  # noqa: F811
    """좌우 두 표는 두 호출 → 한 쪽 안에서 빈 줄로 이어진다. 순서는 박스 순서다."""
    import requests

    n = {"i": 0}

    class _Seq:
        status_code = 200

        def json(self):
            n["i"] += 1
            return {"choices": [{"message": {"content": f"| t{n['i']} |\n|---|\n| v |"},
                                 "finish_reason": "stop"}]}

        def raise_for_status(self):
            pass

    ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_CONCURRENCY="1")
    left = [55.0, 70.0, 260.0, 210.0]
    right = [270.0, 70.0, 490.0, 210.0]
    fake_layout({1: [left, right]})
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Seq())
    pages, _ = _parse(crop_pdf)
    md = pages[0].tables[0]["markdown"]
    assert md.index("t1") < md.index("t2")
    assert "\n\n" in md


def test_crop_budget_is_vision_max_crops(ex, fake_vlm, fake_layout, crop_pdf):  # noqa: F811
    """상한의 단위가 쪽이 아니라 박스다 — VISION_MAX_PAGES는 crop 경로에 관여하지 않는다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_CROPS="2", VISION_MAX_PAGES="50")
    boxes = [[55.0, 70.0 + i * 20, 490.0, 90.0 + i * 20] for i in range(5)]
    fake_layout({1: boxes})
    _, by = _parse(crop_pdf)
    assert fake_vlm["n"] == 2
    assert by["tables_cpu"].metrics["vlm_targets"] == 5


def test_gap_heuristic_is_not_used_when_layout_is_on(ex, fake_vlm, fake_layout, mixed_pdf):  # noqa: F811
    """이미지만 있는 쪽(기존 gap 1순위)도 박스가 없으면 VLM에 가지 않는다 — 백지 루프의 뿌리."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    fake_layout({})  # 박스 0개
    _, by = _parse(mixed_pdf)
    assert fake_vlm["n"] == 0
    assert by["tables_cpu"].metrics["gap_pages"] == 0


# ── 폴백: 레이아웃이 없으면 기존 경로 그대로 ──────────────────────────────────

def test_layout_off_env_runs_the_legacy_path(ex, fake_vlm, mixed_pdf, monkeypatch):  # noqa: F811
    from insurance_chunker import layout as layout_mod
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    monkeypatch.setattr(layout_mod, "LAYOUT_BACKEND", "off")
    _, by = _parse(mixed_pdf)
    assert by["layout"].metrics.get("skipped") is True
    assert by["tables_cpu"].metrics["gap_pages"] > 0, "off면 gap 휴리스틱이 살아 있어야 한다"
    assert fake_vlm["n"] > 0


@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is not None,
                    reason="paddleocr가 설치된 환경에서는 미설치 폴백을 재현할 수 없다")
def test_missing_paddleocr_falls_back_to_the_legacy_path(ex, fake_vlm, mixed_pdf):  # noqa: F811
    """의존이 빠진 이미지가 'VLM이 조용히 꺼진 상태'가 되면 안 된다 — 기존 경로로 돈다."""
    from insurance_chunker import layout as layout_mod
    layout_mod.reset_for_tests()
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    assert layout_mod.LAYOUT_BACKEND == "paddle", "기본값은 레이아웃 켜짐이다"
    pages, by = _parse(mixed_pdf)
    assert by["layout"].metrics["enabled"] is False
    assert fake_vlm["n"] > 0, "레이아웃이 없어도 VLM(gap 경로)은 돌아야 한다"


# ── 레이아웃 모듈 단위 ────────────────────────────────────────────────────────

class _FakeRes:
    def __init__(self, boxes):
        self.json = {"res": {"boxes": boxes}}


class _FakeModel:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def predict(self, img, batch_size=1):
        self.calls += 1
        return [_FakeRes(self.boxes)]


def test_page_table_boxes_filters_and_scales(crop_pdf, monkeypatch):
    """label=table·score>=임계만 남고, 픽셀 좌표가 PDF 포인트로 환산되고, 읽기 순서로 정렬."""
    import pymupdf

    from insurance_chunker import layout as layout_mod
    layout_mod.reset_for_tests()
    fake = _FakeModel([
        {"label": "table", "score": 0.9, "coordinate": [100, 400, 200, 500]},   # 아래 박스
        {"label": "table", "score": 0.9, "coordinate": [100, 100, 200, 200]},   # 위 박스
        {"label": "text", "score": 0.99, "coordinate": [0, 0, 50, 50]},         # 표 아님
        {"label": "table", "score": 0.2, "coordinate": [0, 0, 50, 50]},         # 점수 미달
    ])
    monkeypatch.setattr(layout_mod, "_load_model", lambda: fake)
    with pymupdf.open(crop_pdf) as doc:
        boxes = layout_mod.page_table_boxes(doc[0])
    assert len(boxes) == 2
    assert boxes[0]["bbox_pdf"][1] < boxes[1]["bbox_pdf"][1], "위 박스가 먼저다"
    scale = 72.0 / layout_mod.LAYOUT_DPI
    assert boxes[0]["bbox_pdf"] == pytest.approx([v * scale for v in [100, 100, 200, 200]])


def test_blank_pages_skip_the_model(tmp_path, monkeypatch):
    """백지는 레이아웃을 부르지 않는다 — 쪽당 수 초짜리 모델이라 선필터가 분 단위를 아낀다."""
    import pymupdf

    from insurance_chunker import layout as layout_mod
    layout_mod.reset_for_tests()
    doc = pymupdf.open()
    doc.new_page()                                    # 백지
    p = doc.new_page()
    p.insert_text((70, 100), "text page", fontsize=10)
    path = tmp_path / "blank.pdf"
    doc.save(path)
    doc.close()

    fake = _FakeModel([])
    monkeypatch.setattr(layout_mod, "_load_model", lambda: fake)
    boxes, metrics = layout_mod.detect_tables(str(path))
    assert fake.calls == 1
    assert metrics["blank_skipped"] == 1
    assert boxes == {}
