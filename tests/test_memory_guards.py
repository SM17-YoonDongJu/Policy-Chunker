"""메모리 방어 — OOM 111회 재시작(2026-10-15~20)의 직접 원인과 계측.

434쪽 약관 한 건에서 pdfplumber가 표·gap 356쪽의 글자 캐시를 한 번에 쌓아 2GB를 넘겼고,
파드 한도 4Gi에서 죽었다. 죽는 순간은 VLM 단계여서 로그만으로는 원인을 못 짚었다 — 그래서
단계별 상주 메모리도 같이 남긴다.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fake_pdfplumber(monkeypatch, pages):
    class _PDF:
        def __init__(self):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake = types.ModuleType("pdfplumber")
    fake.open = lambda path: _PDF()
    monkeypatch.setitem(sys.modules, "pdfplumber", fake)


def test_pdfplumber_page_cache_is_released_per_page(monkeypatch):
    """쪽마다 close()로 캐시를 풀어야 최고점이 한 쪽 분량이 된다."""
    closed = []

    class _Page:
        def __init__(self, i):
            self.i = i

        def extract_tables(self):
            return [[["a", "b"], ["1", "2"]]] if self.i % 2 else []

        def close(self):
            closed.append(self.i)

    _fake_pdfplumber(monkeypatch, [_Page(i) for i in range(5)])
    from insurance_chunker import extractor

    out = extractor.extract_pdfplumber_tables("x.pdf", [1, 2, 3, 4, 5])
    assert closed == [0, 1, 2, 3, 4]
    assert set(out) == {2, 4}  # 표가 있는 쪽의 결과는 그대로


def test_page_is_released_even_when_extraction_fails(monkeypatch):
    closed = []

    class _Page:
        def extract_tables(self):
            raise ValueError("broken page")

        def close(self):
            closed.append(True)

    _fake_pdfplumber(monkeypatch, [_Page()])
    from insurance_chunker import extractor

    assert extractor.extract_pdfplumber_tables("x.pdf", [1]) == {}
    assert closed == [True]


def test_rss_is_measured_and_lands_in_every_stage_summary():
    from insurance_chunker import stages

    assert stages.rss_mb() > 0
    assert stages.peak_rss_mb() > 0
    r = stages.StageResult("text", None, 1.0, stages.CPU, {"pages": 3})
    s = r.summary()
    assert s["pages"] == 3 and s["seconds"] == 1.0
    assert s["rss_mb"] > 0
