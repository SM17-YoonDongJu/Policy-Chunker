"""텍스트 레이어 스냅(VLM_SNAP_TEXT) 계약 — 이슈 #72 PR 3.

영역 crop 입력에서도 남는 오독(실측: 관혈→관협)은 입력 토큰이 영역당 1,100~1,300에서 막혀
해상도로는 못 고친다. 같은 쪽 텍스트 레이어가 정답을 들고 있으므로, 짧은 셀을 공백 제거
ratio >= VLM_SNAP_RATIO로 대조해 일치하는 레이어 줄로 바꾼다. 켜고 끈 차이는 벤치의
cell_match(텍스트 레이어 일치율)로 재서 기본값을 정한다 — 그때까지 기본 off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_layout_crop import fake_layout  # noqa: E402,F401 - 픽스처 공유
from test_stages import ex  # noqa: E402,F401 - 픽스처 공유

LAYER_1 = "insured coverage amount 123456"
LAYER_2 = "deductible 5000 won"


# ── 단위: snap_cells_to_text ──────────────────────────────────────────────────

def _snap(md, lines):
    from insurance_chunker.extractor import snap_cells_to_text
    return snap_cells_to_text(md, lines)


def test_near_match_cell_is_replaced_by_the_layer_line():
    md, n = _snap(f"| {LAYER_1[:-1]}5 |\n|---|", [LAYER_1])
    assert n == 1 and LAYER_1 in md


def test_exact_cells_are_left_alone():
    """정확히 일치하는 셀을 다시 쓰면 교정 수가 부풀어 효과 측정이 틀어진다."""
    md, n = _snap(f"| {LAYER_1} |\n|---|", [LAYER_1])
    assert n == 0 and LAYER_1 in md


def test_distant_cells_are_not_forced():
    """비슷한 줄이 없으면 그대로 둔다 — 억지로 붙이면 멀쩡한 셀이 엉뚱한 줄로 바뀐다."""
    md, n = _snap("| 전혀 다른 내용 |\n|---|", [LAYER_1])
    assert n == 0 and "전혀 다른 내용" in md


def test_long_and_multiline_cells_are_excluded():
    """긴 문단 셀과 <br> 셀은 레이어 여러 줄에 걸쳐 있어 줄 단위 대조가 성립하지 않는다."""
    long_cell = "a" * 41
    md, n = _snap(f"| {long_cell} | 가<br>나 |\n|---|---|", [long_cell[:-1] + "b", "가 나"])
    assert n == 0


def test_snapped_text_keeps_the_table_shape():
    """교정된 줄에 파이프가 있으면 마크다운 행이 깨진다 — 전각으로 치환해 넣는다."""
    md, n = _snap("| limit A 100 or B |\n|---|", ["limit A 100 | B"])
    assert n == 1
    row = md.splitlines()[0]
    assert row.count("|") == 2 and "／" in row


# ── 배선: crop 경로에서 켜질 때만 동작한다 ────────────────────────────────────

@pytest.fixture
def snap_pdf(tmp_path):
    """괘선 박스 안에 레이어 줄 두 개 — VLM이 오독한 셀의 정답이 레이어에 있다."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page()
    p.draw_rect(pymupdf.Rect(60, 80, 480, 160), color=(0, 0, 0), width=0.8)
    p.insert_text((70, 100), LAYER_1, fontsize=10)
    p.insert_text((70, 130), LAYER_2, fontsize=10)
    path = tmp_path / "snap.pdf"
    doc.save(path)
    doc.close()
    return str(path)


BOX = [55.0, 70.0, 490.0, 170.0]
# 두 셀 모두 레이어와 한 글자 차이(ratio 0.9 이상), 오른쪽 열은 레이어에 없는 값.
_VLM_MD = (f"| {LAYER_1[:-1]}5 | x |\n|---|---|\n| {LAYER_2[:-1]}m | y |")


def _run_with_vlm(ex, fake_layout, snap_pdf, monkeypatch, **env):  # noqa: F811
    import requests

    from insurance_chunker import stages

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": _VLM_MD},
                                 "finish_reason": "stop"}]}

        def raise_for_status(self):
            pass

    ex(VLM_BACKEND="local", VLM_MODEL="m", **env)
    fake_layout({1: [BOX]})
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    pages, results = stages.parse(snap_pdf)
    return pages, {r.name: r for r in results}


def test_snap_fixes_misread_cells_in_the_crop_path(ex, fake_layout, snap_pdf, monkeypatch):  # noqa: F811
    pages, by = _run_with_vlm(ex, fake_layout, snap_pdf, monkeypatch, VLM_SNAP_TEXT="1")
    md = pages[0].tables[0]["markdown"]
    assert LAYER_1 in md and LAYER_2 in md, "오독 셀이 레이어 원문으로 바뀌어야 한다"
    assert "| x |" in md.replace("  ", " "), "레이어에 없는 셀은 건드리지 않는다"
    assert by["vlm"].metrics["snapped_cells"] == 2


def test_snap_is_off_by_default(ex, fake_layout, snap_pdf, monkeypatch):  # noqa: F811
    """기본값은 벤치(cell_match)로 켜고 끈 차이를 잰 뒤에 정한다 — 그때까지 off."""
    pages, by = _run_with_vlm(ex, fake_layout, snap_pdf, monkeypatch)
    md = pages[0].tables[0]["markdown"]
    assert LAYER_1 not in md, "플래그 없이 교정되면 기본값이 off가 아니다"
    assert by["vlm"].metrics["snapped_cells"] == 0
