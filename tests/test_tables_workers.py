"""PyMuPDF 표 작업 프로세스(TABLES_CPU_WORKERS) 계약.

레이아웃을 GPU로 옮긴 뒤 겹치기 경로의 가장 큰 몫이 PyMuPDF 표였다 — 레이아웃과 같은 스레드에서
쪽마다 번갈아 돌았다. 작업 프로세스로 떼어도 같은 함수를 같은 쪽에 부르므로 결과는 같아야 하고,
프로세스를 못 띄우거나 도중에 죽어도 문서는 끝까지 처리돼야 한다.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_overlap import _parse, _server, doc_pdf, fake_layout  # noqa: E402,F401
from test_stages import ex  # noqa: E402,F401
from test_vlm_stage import table_pdf  # noqa: E402,F401


def _tables(extractor, pdf, pages, workers):
    t = extractor.PymupdfTables(pdf, pages, workers=workers).start()
    return t.result(), t


def test_workers_give_the_same_tables_as_in_process(ex, monkeypatch, table_pdf):  # noqa: F811
    extractor = ex(TABLES_CPU_WORKERS="0")
    monkeypatch.setattr(extractor, "_TABLES_CHUNK", 2)   # 8쪽 → 묶음 4개, 작업자 2개가 나눈다
    pages = list(range(1, 9))
    seq, _ = _tables(extractor, table_pdf, pages, 0)
    par, t = _tables(extractor, table_pdf, pages, 2)
    assert len(seq) == 8, "표 8쪽이 잡혀야 의미 있는 비교다"
    assert par == seq
    assert os.getpid() not in t.pids and len(t.pids) >= 2, "다른 프로세스들에서 뽑아야 한다"
    assert t.seconds > 0


def test_cpu_stage_matches_with_workers(ex, table_pdf):  # noqa: F811
    from insurance_chunker import stages
    pages = list(range(1, 9))
    ex(TABLES_CPU_WORKERS="0", VLM_BACKEND="off")
    seq = stages.tables_cpu(table_pdf, pages, use_vision=False)
    ex(TABLES_CPU_WORKERS="2", VLM_BACKEND="off")
    par = stages.tables_cpu(table_pdf, pages, use_vision=False)
    assert par.output.sources == seq.output.sources
    assert par.metrics["workers"] == 2 and seq.metrics["workers"] == 0


@pytest.mark.parametrize("concurrency", ["1", "3"])
def test_overlap_with_workers_gives_the_same_pages(ex, monkeypatch, doc_pdf,  # noqa: F811
                                                   fake_layout, concurrency):  # noqa: F811
    from test_stages import _as_tuples
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY=concurrency,
       TABLES_CPU_WORKERS="0")
    fake_layout()
    _server(monkeypatch)
    seq_pages, seq, _ = _parse(doc_pdf, True, monkeypatch)
    ex(TABLES_CPU_WORKERS="2")
    fake_layout()
    _server(monkeypatch)
    par_pages, par, _ = _parse(doc_pdf, True, monkeypatch)
    assert _as_tuples(par_pages) == _as_tuples(seq_pages)
    assert par["tables_cpu"].output.sources == seq["tables_cpu"].output.sources
    assert par["tables_cpu"].output.sources["pymupdf"], "PyMuPDF 표가 있어야 의미 있는 비교다"
    assert par["vlm"].output.tables == seq["vlm"].output.tables
    assert par["tables_cpu"].metrics["workers"] == 2


def test_failed_worker_chunks_are_redone_in_process(ex, monkeypatch, table_pdf, caplog):  # noqa: F811
    """작업 프로세스가 실패한 묶음(기동 실패·비정상 종료·시간 초과)은 여기서 마저 뽑는다."""
    import subprocess
    extractor = ex(TABLES_CPU_WORKERS="0")
    monkeypatch.setattr(extractor, "_TABLES_CHUNK", 2)
    pages = list(range(1, 9))
    seq, _ = _tables(extractor, table_pdf, pages, 0)

    real = extractor._pymupdf_chunk

    def _flaky(pdf, chunk):
        if chunk[0] in (3, 7):
            raise subprocess.CalledProcessError(1, "pymupdf_worker", stderr="killed")
        return real(pdf, chunk)
    monkeypatch.setattr(extractor, "_pymupdf_chunk", _flaky)
    with caplog.at_level(logging.WARNING, logger="insurance_chunker.extractor"):
        par, t = _tables(extractor, table_pdf, pages, 2)
    assert par == seq
    assert os.getpid() in t.pids and len(t.pids) >= 2, "일부는 작업자, 나머지는 여기서"
    warned = [r for r in caplog.records if getattr(r, "event", "") == "tables_workers_failed"]
    assert len(warned) == 1, "실패 묶음이 여럿이어도 경고는 한 번"


def test_worker_module_runs_standalone(table_pdf):  # noqa: F811
    """작업 프로세스는 부모의 __main__을 다시 import하지 않고 이 모듈만 돈다."""
    import json
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "insurance_chunker.pymupdf_worker", table_pdf,
                           "1,2"], capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    assert sorted(data["tables"]) == ["1", "2"] and data["pid"] != os.getpid()


def test_overlap_fallback_leaves_no_worker_processes(ex, monkeypatch, doc_pdf,  # noqa: F811
                                                     fake_layout):  # noqa: F811
    """레이아웃 모델이 못 열려 순차 경로로 물러날 때 작업 스레드·프로세스를 정리한다."""
    ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B", VLM_CONCURRENCY="1",
       TABLES_CPU_WORKERS="2")
    fake_layout(fail_on_page=1)
    _server(monkeypatch)
    _, by, _ = _parse(doc_pdf, True, monkeypatch)
    assert by["layout"].metrics["enabled"] is False
    import threading
    assert not [t for t in threading.enumerate() if t.name.startswith("pymupdf")]
