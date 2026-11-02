"""레이아웃·VLM 겹치기(PIPELINE_OVERLAP) 계약.

Paddle 전환으로 VLM이 빨라져 CPU 단계(레이아웃·PyMuPDF 표)와 GPU 단계(VLM)가 같은 규모가 됐다.
겹치기 경로는 레이아웃이 쪽마다 찾은 박스를 바로 VLM에 넘긴다. 시간만 바꿔야 한다 — VLM 대상의
내용·순서·상한 적용이 순차 경로와 같아서 결과가 같아야 하고, 실제로 겹쳐야 하며, 레이아웃을 못
쓰면 순차 경로로 물러나야 한다.

paddleocr는 테스트 환경에 없다 — 박스 탐지(page_table_boxes)를 가짜로 끼운다
(test_layout_crop와 같은 방식).
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_stages import _as_tuples, ex  # noqa: E402,F401 - 픽스처 공유
from test_vlm_stage import _Server  # noqa: E402

# 쪽 → 표 박스(PDF 포인트). p3은 박스 2개, p4는 박스 없음, p5는 이미지만(OCR 대상), p6은 백지
BOXES = {1: [[55.0, 70.0, 490.0, 210.0]],
         2: [[55.0, 70.0, 350.0, 160.0]],
         3: [[55.0, 70.0, 490.0, 150.0], [55.0, 300.0, 490.0, 420.0]],
         7: [[55.0, 70.0, 490.0, 210.0]]}


@pytest.fixture
def doc_pdf(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    for k in range(7):
        p = doc.new_page()
        if k == 4:   # 이미지만 — 텍스트 레이어 없이 쪽을 덮는다(OCR 대상)
            src = pymupdf.open()
            sp = src.new_page(width=200, height=200)
            sp.draw_rect(pymupdf.Rect(0, 0, 200, 200), color=(0, 0, 0), fill=(0.6, 0.6, 0.6))
            sp.insert_text((20, 100), f"scan{k}", fontsize=20)
            p.insert_image(p.rect, stream=sp.get_pixmap(dpi=72).tobytes("png"))
            src.close()
            continue
        if k == 5:   # 백지
            continue
        p.insert_text((70, 50), f"Article {k + 1} body text", fontsize=11)
        for r in range(4):
            for c in range(3):
                rect = pymupdf.Rect(60 + c * 140, 80 + r * 28, 60 + (c + 1) * 140,
                                    80 + (r + 1) * 28)
                p.draw_rect(rect, color=(0, 0, 0), width=0.8)
                p.insert_text((rect.x0 + 5, rect.y0 + 18), f"p{k}r{r}c{c}", fontsize=9)
    path = tmp_path / "overlap.pdf"
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture
def fake_layout(monkeypatch):
    """page_table_boxes를 가짜로. delay를 주면 쪽마다 그만큼 걸린다(CPU 레이아웃 흉내)."""
    from insurance_chunker import layout as layout_mod

    def _install(boxes=BOXES, delay: float = 0.0, fail_on_page: int | None = None):
        layout_mod.reset_for_tests()
        monkeypatch.setattr(layout_mod, "LAYOUT_BACKEND", "paddle")
        calls = {"n": 0, "threads": set()}

        def _boxes(page):
            calls["n"] += 1
            calls["threads"].add(threading.get_ident())
            if fail_on_page is not None and page.number + 1 >= fail_on_page:
                layout_mod._load_failed = True
                return []
            if delay:
                time.sleep(delay)
            return [{"bbox_pdf": list(b), "bbox_px": [v * layout_mod.LAYOUT_DPI / 72 for v in b],
                     "score": 0.9} for b in boxes.get(page.number + 1, [])]
        monkeypatch.setattr(layout_mod, "page_table_boxes", _boxes)
        return calls
    yield _install
    from insurance_chunker import layout as layout_mod2
    layout_mod2.reset_for_tests()


def _server(monkeypatch, delay=0.0):
    import requests
    srv = _Server(delay=delay)
    order: list[str] = []
    orig = srv.post

    def _post(url, json=None, timeout=None):
        order.append(json["messages"][0]["content"][0]["image_url"]["url"][-40:])
        return orig(url, json=json, timeout=timeout)
    monkeypatch.setattr(requests, "post", _post)
    srv.order = order
    return srv


def _parse(pdf, overlap: bool, monkeypatch):
    from insurance_chunker import stages
    monkeypatch.setattr(stages, "PIPELINE_OVERLAP", overlap)
    t0 = time.perf_counter()
    pages, results = stages.parse(pdf)
    return pages, {r.name: r for r in results}, time.perf_counter() - t0


# ── 결과 동일성 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("concurrency", ["1", "3"])
def test_overlap_gives_the_same_pages(ex, monkeypatch, doc_pdf, fake_layout, concurrency):  # noqa: F811
    """표·본문·OCR 복구까지 순차와 같아야 한다.

    가짜 서버는 이미지 해시로 답하므로 완료 순서가 섞여도 비교된다.
    """
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY=concurrency)
    fake_layout()
    _server(monkeypatch)
    seq_pages, seq, _ = _parse(doc_pdf, False, monkeypatch)
    fake_layout()
    _server(monkeypatch)
    ovl_pages, ovl, _ = _parse(doc_pdf, True, monkeypatch)

    assert seq["vlm"].metrics["targets"] == 6, "OCR 1 + 박스 5 — 의미 있는 비교여야 한다"
    assert ovl["vlm"].metrics.get("overlap") is True
    assert _as_tuples(ovl_pages) == _as_tuples(seq_pages)
    assert ovl["vlm"].output.tables == seq["vlm"].output.tables
    assert ovl["vlm"].output.ocr_text == seq["vlm"].output.ocr_text
    assert ovl["layout"].output.boxes == seq["layout"].output.boxes
    assert ovl["tables_cpu"].output.sources == seq["tables_cpu"].output.sources
    assert ovl["tables_cpu"].output.targets == seq["tables_cpu"].output.targets


def test_calls_go_out_in_the_same_order(ex, monkeypatch, doc_pdf, fake_layout):  # noqa: F811
    """동시도 1이면 호출 순서까지 같다 — OCR 쪽 먼저, 그다음 쪽 순서대로 박스."""
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="1")
    fake_layout()
    a = _server(monkeypatch)
    _parse(doc_pdf, False, monkeypatch)
    fake_layout()
    b = _server(monkeypatch)
    _parse(doc_pdf, True, monkeypatch)
    assert a.order == b.order and len(a.order) == 6


def test_budget_picks_the_same_targets(ex, monkeypatch, doc_pdf, fake_layout):  # noqa: F811
    """상한은 대상 순서대로 깎인다. 흘려 받아도 같은 대상이 남아야 한다."""
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="2",
       VISION_MAX_CROPS="3")
    fake_layout()
    a = _server(monkeypatch)
    seq_pages, seq, _ = _parse(doc_pdf, False, monkeypatch)
    fake_layout()
    b = _server(monkeypatch)
    ovl_pages, ovl, _ = _parse(doc_pdf, True, monkeypatch)
    assert a.calls == b.calls == 3
    assert ovl["vlm"].output.tables == seq["vlm"].output.tables
    assert _as_tuples(ovl_pages) == _as_tuples(seq_pages)


# ── 실제로 겹치는가 ───────────────────────────────────────────────────────────

def test_layout_and_vlm_really_overlap(ex, monkeypatch, doc_pdf, fake_layout):  # noqa: F811
    """레이아웃 쪽당 0.15초, VLM 호출당 0.2초면 순차는 합, 겹치기는 큰 쪽에 가깝다."""
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="1")
    fake_layout(delay=0.15)
    _server(monkeypatch, delay=0.2)
    _, seq, seq_wall = _parse(doc_pdf, False, monkeypatch)
    fake_layout(delay=0.15)
    _server(monkeypatch, delay=0.2)
    _, ovl, ovl_wall = _parse(doc_pdf, True, monkeypatch)

    lay, vlm = ovl["layout"].seconds, ovl["vlm"].seconds
    assert lay == pytest.approx(6 * 0.15, abs=0.15), "레이아웃은 백지 빼고 6쪽"
    assert vlm >= 6 * 0.2 * 0.9, "VLM 단계 시간은 호출이 진행 중이던 구간의 합이다"
    assert ovl_wall < seq_wall - 0.4, f"겹치지 않았다: 순차 {seq_wall:.2f}s 대 {ovl_wall:.2f}s"
    # 묶음 벽시계에는 레이아웃·VLM 말고도 PyMuPDF 표·렌더 같은 CPU 구간이 들어 있다. 느린
    # 러너에서는 그 구간만 1초를 넘어 lay + vlm과 비교하면 겹쳐도 깨진다(CI 2026-11-02:
    # 2.41초 대 0.9 + 1.21초). 같은 CPU 구간을 담은 순차 경로의 묶음 합과 비교한다.
    seq_group = sum(seq[n].seconds for n in ("layout", "tables_cpu", "vlm"))
    assert ovl["vlm"].metrics["overlap_wall_s"] < seq_group - 0.4, \
        f"겹친 만큼 묶음 벽시계가 순차 묶음 합보다 짧다: {seq_group:.2f}s"


def test_layout_runs_on_the_calling_thread(ex, monkeypatch, doc_pdf, fake_layout):  # noqa: F811
    """레이아웃(PyMuPDF 객체)은 문서를 연 스레드에서만 — VLM 작업 스레드가 만지면 안 된다."""
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="3")
    calls = fake_layout()
    _server(monkeypatch, delay=0.05)
    _parse(doc_pdf, True, monkeypatch)
    assert calls["threads"] == {threading.get_ident()}


# ── 물러나기 ──────────────────────────────────────────────────────────────────

def test_model_failure_falls_back_to_the_sequential_path(ex, monkeypatch, doc_pdf,  # noqa: F811
                                                         fake_layout):
    """레이아웃 모델을 못 열면 겹치기 없이 순차로 — 거기서 기존 경로(gap + 쪽 전체)로 떨어진다."""
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="1")
    fake_layout(fail_on_page=1)
    srv = _server(monkeypatch)
    _, by, _ = _parse(doc_pdf, True, monkeypatch)
    assert by["layout"].metrics["enabled"] is False
    assert by["vlm"].metrics.get("overlap") is None
    assert srv.calls > 0, "기존 경로로라도 VLM은 돌아야 한다"


def test_overlap_is_off_by_default(monkeypatch):
    import importlib

    from insurance_chunker import stages
    monkeypatch.delenv("PIPELINE_OVERLAP", raising=False)
    importlib.reload(stages)
    assert stages.PIPELINE_OVERLAP is False


@pytest.mark.parametrize("overlap", [False, True])
def test_layout_on_gpu_is_counted_as_a_gpu_stage(ex, monkeypatch, doc_pdf, fake_layout,  # noqa: F811
                                                 overlap):
    """레이아웃을 GPU에 올리면 하한 계산에서도 GPU 단계다 — CPU로 남으면 병목을 잘못 고른다."""
    from insurance_chunker import layout as layout_mod
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="2")
    fake_layout()
    _server(monkeypatch)
    _, cpu, _ = _parse(doc_pdf, overlap, monkeypatch)
    monkeypatch.setattr(layout_mod, "LAYOUT_DEVICE", "gpu:0")
    fake_layout()
    _server(monkeypatch)
    _, gpu, _ = _parse(doc_pdf, overlap, monkeypatch)
    assert cpu["layout"].resource == "cpu"
    assert gpu["layout"].resource == "gpu"
    assert gpu["layout"].metrics["device"] == "gpu:0"


# ── 단계 실행 CLI ─────────────────────────────────────────────────────────────

def test_stage_runner_reports_wall_time_and_keeps_caches_apart(ex, monkeypatch, doc_pdf,  # noqa: F811
                                                               fake_layout, tmp_path):
    """리포트의 처리 시간은 겹친 구간을 벽시계로 센다.

    순차 실행의 산출물(과 그때 잰 시간)은 재사용하지 않는다.
    """
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="1")
    from insurance_chunker import stage_runner, stages
    from insurance_chunker.models import DocMeta, compute_doc_hash
    meta = DocMeta(source_pdf="overlap.pdf", doc_hash=compute_doc_hash(doc_pdf),
                   doc_type="plain", insurer="t", product_name="p")

    fake_layout(delay=0.1)
    _server(monkeypatch, delay=0.15)
    monkeypatch.setattr(stages, "PIPELINE_OVERLAP", False)
    seq = stage_runner.run(doc_pdf, tmp_path, meta, to_stage="combine")
    fake_layout(delay=0.1)
    srv = _server(monkeypatch, delay=0.15)
    monkeypatch.setattr(stages, "PIPELINE_OVERLAP", True)
    ovl = stage_runner.run(doc_pdf, tmp_path, meta, to_stage="combine")

    by = {r["stage"]: r for r in ovl}
    assert not by["vlm"]["cached"] and srv.calls == 6, \
        "순차 산출물을 재사용하면 겹치기를 잰 게 아니다"
    assert by["layout"]["key"] != {r["stage"]: r for r in seq}["layout"]["key"]
    wall = by["vlm"]["metrics"]["overlap_wall_s"]
    assert stage_runner._elapsed(ovl) == pytest.approx(
        wall + by["text"]["seconds"] + by["combine"]["seconds"], abs=0.01)
    assert stage_runner._elapsed(ovl) < sum(r["seconds"] for r in ovl)


def test_stage_runner_reruns_the_whole_group_when_only_vlm_changes(ex, monkeypatch, doc_pdf,  # noqa: F811
                                                                   fake_layout, tmp_path):
    """동시도만 바꾼 겹치기 변형은 vlm 키만 다르다. 레이아웃·PyMuPDF 표를 캐시에서 읽고 VLM만
    순차로 돌리면 '겹치기'라는 이름으로 순차 시간을 잰다 — 셋 다 다시 돌아야 한다."""
    from insurance_chunker import stage_runner, stages
    from insurance_chunker.models import DocMeta, compute_doc_hash
    meta = DocMeta(source_pdf="overlap.pdf", doc_hash=compute_doc_hash(doc_pdf),
                   doc_type="plain", insurer="t", product_name="p")
    monkeypatch.setattr(stages, "PIPELINE_OVERLAP", True)

    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="1")
    fake_layout()
    _server(monkeypatch)
    stage_runner.run(doc_pdf, tmp_path, meta, to_stage="combine")

    ex(VLM_CONCURRENCY="4")
    calls = fake_layout()
    _server(monkeypatch)
    recs = {r["stage"]: r for r in stage_runner.run(doc_pdf, tmp_path, meta, to_stage="combine")}
    assert not recs["layout"]["cached"] and not recs["tables_cpu"]["cached"]
    assert calls["n"] > 0, "레이아웃이 다시 돌아야 한다"
    assert recs["vlm"]["metrics"].get("overlap_wall_s") is not None
