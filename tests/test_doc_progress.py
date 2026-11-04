"""쪽 진행 보고 — 파이프라인이 어느 단계·어느 쪽을 지나는지 처리 중에 밖으로 알리는 훅.

reporter가 없으면 전부 no-op이어야 하고(단독 CLI·벤치·테스트), 보고 실패가 처리를
죽여서도 안 된다. 단계 모듈들이 실제로 보고를 부르는지는 작은 PDF로 parse를 돌려 본다 —
순차 경로(본문·CPU 표 루프)와 겹치기 경로(스윕 루프, detect_tables가 안 불린다) 둘 다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_overlap import _parse, _server, doc_pdf, fake_layout  # noqa: E402,F401 - 픽스처 공유
from test_stages import ex  # noqa: E402,F401 - 픽스처 공유

from insurance_chunker import progress  # noqa: E402


@pytest.fixture(autouse=True)
def reset_reporter():
    yield
    progress.set_reporter(None)


def test_report_without_a_reporter_is_a_noop():
    progress.report("text", 1, 2)  # 예외 없이 지나가면 된다


def test_reports_are_throttled_by_time_not_by_stage(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock["t"])
    calls: list = []
    progress.set_reporter(lambda s, p, n: calls.append((s, p, n)))

    progress.report("layout", 1, 10)  # 첫 보고는 바로
    progress.report("vlm", 1, 10)     # 겹치기처럼 단계가 번갈아 와도 간격 안이면 묶인다
    clock["t"] += 1.5
    progress.report("vlm", 3, 10)     # 간격이 지났다
    progress.report("chunk", 10, 10, force=True)  # 쪽 루프 없는 전환은 간격과 무관하게 바로
    assert calls == [("layout", 1, 10), ("vlm", 3, 10), ("chunk", 10, 10)]


def test_reporter_errors_do_not_escape():
    def boom(stage, page, pages):
        raise OSError("디스크가 가득 참")

    progress.set_reporter(boom)
    progress.report("layout", 1, 10)  # 기록 실패가 파이프라인을 죽이면 안 된다


def test_sequential_parse_reports_text_and_cpu_table_pages(tmp_path, monkeypatch):
    """레이아웃·VLM이 꺼진 순차 경로에서도 본문·CPU 표 단계가 쪽마다 진행을 알린다."""
    import pymupdf

    from insurance_chunker import extractor, layout, stages

    pdf = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for i in range(2):
        doc.new_page().insert_text((72, 72), f"제{i + 1}조(목적) 내용", fontsize=11)
    doc.save(pdf)
    doc.close()

    monkeypatch.setattr(layout, "LAYOUT_BACKEND", "off")
    monkeypatch.setattr(extractor, "VLM_BACKEND", "off")
    monkeypatch.setattr(progress, "_MIN_INTERVAL_S", 0.0)
    calls: list = []
    progress.set_reporter(lambda s, p, n: calls.append((s, p, n)))

    stages.parse(str(pdf), use_vision=False)
    assert calls == [("text", 1, 2), ("text", 2, 2),
                     ("tables_cpu", 1, 2), ("tables_cpu", 2, 2)]


def test_overlapped_parse_reports_the_sweep(ex, monkeypatch, doc_pdf, fake_layout):  # noqa: F811
    """겹치기에선 detect_tables·extract_cpu_tables가 안 불린다 — 스윕 루프가 쪽 진행을
    알리고, VLM 제출도 같은 스레드에서 섞여 흐른다."""
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="1")
    fake_layout()
    _server(monkeypatch)
    monkeypatch.setattr(progress, "_MIN_INTERVAL_S", 0.0)
    calls: list = []
    progress.set_reporter(lambda s, p, n: calls.append((s, p, n)))

    _parse(doc_pdf, True, monkeypatch)

    assert [p for s, p, n in calls if s == "layout"] == list(range(1, 8))  # 쪽마다, 순서대로
    assert any(s == "vlm" for s, p, n in calls)
    assert not any(s == "tables_cpu" for s, p, n in calls)  # 겹치기에선 그 루프가 없다
    assert all(n == 7 for s, p, n in calls)  # 분모는 늘 전체 쪽 수
