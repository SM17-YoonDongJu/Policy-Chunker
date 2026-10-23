"""메모리 계측 — OOM 111회 재시작(2026-10-15~20)의 재발을 보는 눈.

원인이던 pdfplumber(쪽 글자 캐시)는 #72에서 경로째 제거됐다 — 쪽마다 close()하는
핫픽스(#71)와 그 테스트도 함께 걷어냈다. 남는 방어는 단계별 상주 메모리 계측이다:
죽는 순간의 단계가 아니라 쌓이는 단계를 지목할 수 있어야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rss_is_measured_and_lands_in_every_stage_summary():
    from insurance_chunker import stages

    assert stages.rss_mb() > 0
    assert stages.peak_rss_mb() > 0
    r = stages.StageResult("text", None, 1.0, stages.CPU, {"pages": 3})
    s = r.summary()
    assert s["pages"] == 3 and s["seconds"] == 1.0
    assert s["rss_mb"] > 0


def test_pdfplumber_path_is_gone():
    """OOM 벡터가 코드째 사라졌는지 고정한다 — 다시 들어오면 쪽 캐시 문제도 같이 돌아온다."""
    from insurance_chunker import extractor

    assert not hasattr(extractor, "extract_pdfplumber_tables")
    assert not hasattr(extractor, "extract_camelot_tables")
