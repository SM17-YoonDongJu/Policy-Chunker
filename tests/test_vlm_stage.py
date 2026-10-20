"""VLM 단계 동시 호출 계약.

동시도를 올리는 건 시간만 바꿔야 한다. 같은 입력이면 같은 표가 나와야 하고(벤치마크의
expect_identical_to가 그걸 본다), 호출 상한·차단기가 병렬에서도 지켜져야 한다. 그리고 실제로
동시에 나가야 한다 — 겉으로만 스레드를 쓰고 직렬로 돌면 의미가 없다.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_stages import _as_tuples, mixed_pdf  # noqa: E402,F401 - 픽스처 공유


class _Resp:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}, "finish_reason": "stop"}]}

    def raise_for_status(self):
        pass


@pytest.fixture
def load(monkeypatch):
    def _load(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from insurance_chunker import extractor
        importlib.reload(extractor)
        from insurance_chunker import vlm_stage
        importlib.reload(vlm_stage)
        return extractor, vlm_stage
    return _load


@pytest.fixture
def table_pdf(tmp_path):
    """괘선 표 8쪽. 페이지마다 내용이 달라 렌더 이미지도 다르다."""
    import pymupdf
    doc = pymupdf.open()
    for k in range(8):
        p = doc.new_page()
        for r in range(4):
            for c in range(3):
                rect = pymupdf.Rect(60 + c * 140, 80 + r * 28, 60 + (c + 1) * 140,
                                    80 + (r + 1) * 28)
                p.draw_rect(rect, color=(0, 0, 0), width=0.8)
                p.insert_text((rect.x0 + 5, rect.y0 + 18), f"p{k}r{r}c{c}", fontsize=9)
    path = tmp_path / "t.pdf"
    doc.save(path)
    doc.close()
    return str(path)


class _Server:
    """가짜 VLM 서버. 이미지 해시로 응답을 정해(같은 페이지 → 같은 표) 완료 순서와 무관하게
    결과를 비교할 수 있게 하고, 동시에 처리 중인 요청 수의 최댓값을 잰다."""

    def __init__(self, delay=0.0, fail=False):
        self.delay, self.fail = delay, fail
        self.calls = 0
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def post(self, url, json=None, timeout=None):
        import requests
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if self.fail:
                raise requests.ConnectionError("refused")
            img = json["messages"][0]["content"][0]["image_url"]["url"]
            h = hashlib.sha256(img.encode()).hexdigest()[:8]
            return _Resp(f"| h |\n|---|\n| {h} |")
        finally:
            with self._lock:
                self.active -= 1


def _install(monkeypatch, server):
    import requests
    monkeypatch.setattr(requests, "post", server.post)
    return server


def _run(extractor, pdf):
    from insurance_chunker import stages
    pages, results = stages.parse(pdf)
    return pages, results[2]


# ── 결과 동일성 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("concurrency", ["2", "3", "8"])
def test_concurrency_does_not_change_results(load, monkeypatch, table_pdf, concurrency):
    """동시도는 시간만 바꾼다. 완료 순서가 뒤섞여도 페이지별 표가 순차와 같아야 한다."""
    extractor, _ = load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables",
                        VLM_CONCURRENCY="1")
    srv = _install(monkeypatch, _Server(delay=0.01))
    seq_pages, seq_vlm = _run(extractor, table_pdf)

    extractor, _ = load(VLM_CONCURRENCY=concurrency)
    srv = _install(monkeypatch, _Server(delay=0.01))
    par_pages, par_vlm = _run(extractor, table_pdf)

    assert seq_vlm.metrics["tables"] == 8
    assert _as_tuples(par_pages) == _as_tuples(seq_pages)
    assert par_vlm.output == seq_vlm.output
    assert srv.calls == 8


def test_staged_parse_still_matches_legacy_with_concurrency(load, monkeypatch, mixed_pdf):  # noqa: F811
    """단계 분리 전 경로(extract_tables_for_doc)와도 같아야 한다 — 동시도 3에서."""
    extractor, _ = load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="both",
                        VLM_CONCURRENCY="3")
    _install(monkeypatch, _Server())
    from test_stages import _legacy_parse
    legacy = _legacy_parse(mixed_pdf, use_vision=True)
    staged, _ = _run(extractor, mixed_pdf)
    assert _as_tuples(staged) == _as_tuples(legacy)


# ── 실제로 동시에 나가는가 ────────────────────────────────────────────────────

def test_calls_really_overlap_up_to_the_limit(load, monkeypatch, table_pdf):
    """동시도 3이면 8건이 3·3·2 세 차례로 나가 0.6초 안팎이다. 직렬이면 1.6초다.

    VLM 단계의 시간만 재고 렌더링은 가짜로 둔다. 전체 파싱 벽시계로 재면 텍스트 추출·pdfplumber·
    렌더 같은 CPU 구간이 1.12초 예산에 섞여, 느린 러너에서는 동시 호출과 무관하게 깨진다 —
    ubuntu-latest가 Ubuntu 26으로 바뀐 2026-10-19 이후 CI의 pytest가 13초에서 120초로 늘자
    이 테스트만 5.35초로 떨어졌다. 로컬에서도 예산의 0.84초를 이미 쓰고 있었다.
    """
    extractor, _ = load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables",
                        VLM_CONCURRENCY="3")
    monkeypatch.setattr(extractor, "render_page", lambda page, pno: (f"img{pno}", 1))
    srv = _install(monkeypatch, _Server(delay=0.2))
    _, vlm = _run(extractor, table_pdf)
    assert srv.peak == 3, f"동시 처리 최댓값이 {srv.peak} — 동시도 3이어야 한다"
    assert vlm.seconds < 8 * 0.2 * 0.7, f"직렬(1.6초)과 다를 바 없다: {vlm.seconds:.2f}초"


def test_concurrency_one_is_strictly_sequential(load, monkeypatch, table_pdf):
    extractor, _ = load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables",
                        VLM_CONCURRENCY="1")
    srv = _install(monkeypatch, _Server(delay=0.02))
    _run(extractor, table_pdf)
    assert srv.peak == 1


# ── 상한·차단기 ───────────────────────────────────────────────────────────────

def test_budget_holds_under_concurrency(load, monkeypatch, table_pdf):
    extractor, _ = load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables",
                        VLM_CONCURRENCY="4", VISION_MAX_PAGES="5")
    srv = _install(monkeypatch, _Server(delay=0.01))
    _, vlm = _run(extractor, table_pdf)
    assert srv.calls == 5 and vlm.metrics["tables"] == 5


def test_circuit_bounds_wasted_calls_under_concurrency(load, monkeypatch, table_pdf):
    """동시도 N이면 차단 순간 이미 날아간 N-1건까지만 더 끝난다. 나머지는 부르지 않는다."""
    extractor, _ = load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables",
                        VLM_CONCURRENCY="3", VLM_FAIL_STREAK="2")
    srv = _install(monkeypatch, _Server(delay=0.05, fail=True))
    _, vlm = _run(extractor, table_pdf)
    assert 2 <= srv.calls <= 2 + 3 - 1, f"호출 {srv.calls}회"
    assert vlm.metrics["circuit_open"] is True
    assert vlm.metrics["calls"] == srv.calls


def test_circuit_logs_the_skipped_count_once(load, monkeypatch, table_pdf, caplog):
    import logging
    extractor, _ = load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables",
                        VLM_CONCURRENCY="3", VLM_FAIL_STREAK="2")
    srv = _install(monkeypatch, _Server(delay=0.05, fail=True))
    with caplog.at_level(logging.INFO, logger="insurance_chunker.extractor"):
        _run(extractor, table_pdf)
    opened = [r for r in caplog.records if getattr(r, "event", "") == "vlm_circuit_open"]
    skipped = [r for r in caplog.records if getattr(r, "event", "") == "vlm_pages_skipped"]
    assert len(opened) == 1 and len(skipped) == 1
    assert skipped[0].skipped == 8 - srv.calls


def test_stats_match_calls(load, monkeypatch, table_pdf):
    """호출 기록은 계측의 근거다 — 동시 호출에서 빠지거나 겹치면 지연 분포가 틀린다."""
    extractor, _ = load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables",
                        VLM_CONCURRENCY="4")
    srv = _install(monkeypatch, _Server(delay=0.01))
    _, vlm = _run(extractor, table_pdf)
    assert len(vlm.metrics["latencies"]) == srv.calls == 8
    assert vlm.metrics["concurrency"] == 4
    assert sorted(s["page"] for s in extractor.vision_page_stats()) == list(range(1, 9))
