"""VLM 동시도 자동 조정 계약.

vLLM을 OCR worker와 나눠 쓴다. 대기열이 생기면 이쪽이 물러서고(OCR worker 우선), 비면 천천히
상한(VLM_CONCURRENCY)까지 돌아간다. 조정은 시간만 바꾼다 — 같은 입력이면 같은 표다.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_vlm_stage import _install, _run, _Server, load, table_pdf  # noqa: E402,F401 - 픽스처 공유

METRICS = """# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_running{{engine="0",model_name="PaddleOCR-VL-1.6-0.9B"}} 3.0
vllm:num_requests_waiting{{engine="0",model_name="PaddleOCR-VL-1.6-0.9B"}} {w}
"""


class _Metrics:
    def __init__(self, waiting=0, fail=False):
        self.waiting, self.fail, self.polls = waiting, fail, 0

    def get(self, url, timeout=None):
        import requests
        self.polls += 1
        if self.fail:
            raise requests.ConnectionError("refused")
        assert url.endswith("/metrics")
        w = self.waiting

        class R:
            text = METRICS.format(w=float(w))

            def raise_for_status(self):
                pass
        return R()


@pytest.fixture
def adaptive(monkeypatch, load):  # noqa: F811
    from insurance_chunker import vlm_limiter

    def _setup(metrics: _Metrics, **env):
        env.setdefault("VLM_ADAPTIVE", "1")
        env.setdefault("VLM_ADAPT_INTERVAL", "60")   # 기본은 폴러가 끼어들지 않게
        env.setdefault("VLM_ADAPT_COOLDOWN", "3")
        import requests
        monkeypatch.setattr(requests, "get", metrics.get)
        vlm_limiter.reset_for_tests()
        return load(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables", **env)
    yield _setup
    vlm_limiter.reset_for_tests()


# ── 판정 규칙 ─────────────────────────────────────────────────────────────────

def test_parse_waiting_sums_label_sets():
    from insurance_chunker.vlm_limiter import parse_waiting
    two = METRICS.format(w=2.0) + 'vllm:num_requests_waiting{engine="1"} 3.0\n'
    assert parse_waiting(two) == 5
    assert parse_waiting("# nothing\nother_metric 1\n") is None


def test_backs_off_by_half_while_the_server_queues():
    from insurance_chunker.vlm_limiter import AdaptiveLimit
    lim = AdaptiveLimit(16, floor=2, cooldown=3)
    assert lim.observe(4) == (16, 8)
    assert lim.observe(1) == (8, 4)
    assert lim.observe(1) == (4, 2)
    assert lim.observe(9) is None and lim.limit == 2, "바닥 밑으로는 안 내려간다"


def test_raises_one_step_only_after_cooldown_and_only_when_blocked():
    from insurance_chunker.vlm_limiter import AdaptiveLimit
    lim = AdaptiveLimit(4, cooldown=3)
    lim.observe(5)                       # 4 → 2
    lim._blocked = True
    assert lim.observe(0) is None        # 조용한 1회
    lim._blocked = True
    assert lim.observe(0) is None        # 2회
    assert lim.observe(0) is None, "막힌 적이 없으면 올리지 않는다"
    lim._blocked = True
    assert lim.observe(0) == (2, 3)
    lim._blocked = True
    assert lim.observe(0) == (3, 4)
    lim._blocked = True
    assert lim.observe(0) is None and lim.limit == 4, "상한(VLM_CONCURRENCY)을 넘지 않는다"


def test_unreadable_metrics_hold_the_limit():
    from insurance_chunker.vlm_limiter import AdaptiveLimit
    lim = AdaptiveLimit(8)
    assert lim.observe(None) is None and lim.limit == 8


# ── VlmStage에 걸렸을 때 ──────────────────────────────────────────────────────

def test_lowered_limit_caps_in_flight_calls_and_keeps_results(adaptive, monkeypatch, table_pdf):  # noqa: F811
    """스레드는 8개여도 상한이 2면 서버에 동시에 걸리는 건 2건이다. 결과는 순차와 같다."""
    from insurance_chunker import vlm_limiter
    extractor = adaptive(_Metrics(), VLM_ADAPTIVE="0", VLM_CONCURRENCY="1")
    _install(monkeypatch, _Server(delay=0.01))
    seq_pages, seq_vlm = _run(extractor, table_pdf)

    extractor = adaptive(_Metrics(), VLM_CONCURRENCY="8")
    vlm_limiter.get().limit = 2
    srv = _install(monkeypatch, _Server(delay=0.05))
    pages, vlm = _run(extractor, table_pdf)
    assert srv.peak == 2
    assert vlm.output == seq_vlm.output
    assert vlm.metrics["adaptive"] is True and vlm.metrics["concurrency_range"] == [2, 2]


def test_server_queue_drives_the_limit_down_during_a_document(adaptive, monkeypatch,
                                                               table_pdf, caplog):  # noqa: F811
    metrics = _Metrics(waiting=5)
    extractor = adaptive(metrics, VLM_CONCURRENCY="8", VLM_ADAPT_INTERVAL="0.05")
    srv = _install(monkeypatch, _Server(delay=0.15))
    with caplog.at_level(logging.INFO, logger="insurance_chunker.extractor"):
        _, vlm = _run(extractor, table_pdf)
    # 첫 호출이 언제 자리를 얻느냐는 러너 속도에 달렸다 — 느린 CI에선 첫 호출 전에 폴러가 이미
    # 8 → 4로 내려 범위가 1~4로 찍혔다(2026-11-03). 그래서 범위의 위쪽 대신 조정 기록으로 본다.
    lo, _ = vlm.metrics["concurrency_range"]
    assert lo == 1, f"대기열을 보고 바닥까지 물러서야 한다: {vlm.metrics['concurrency_range']}"
    assert srv.calls == 8 and vlm.metrics["tables"] == 8, "물러서도 대상은 다 부른다"
    changes = [(r.old, r.new) for r in caplog.records
               if getattr(r, "event", "") == "vlm_concurrency_change"]
    assert changes[:3] == [(8, 4), (4, 2), (2, 1)]


def test_limit_recovers_when_the_queue_clears(adaptive, monkeypatch, table_pdf):  # noqa: F811
    from insurance_chunker import vlm_limiter
    metrics = _Metrics(waiting=0)
    extractor = adaptive(metrics, VLM_CONCURRENCY="4", VLM_ADAPT_INTERVAL="0.02",
                         VLM_ADAPT_COOLDOWN="1")
    vlm_limiter.get().limit = 1
    _install(monkeypatch, _Server(delay=0.1))
    _, vlm = _run(extractor, table_pdf)
    assert vlm.metrics["concurrency_range"][1] > 1, "막혀 있고 대기열이 비면 다시 올라간다"


def test_unreachable_metrics_keep_fixed_concurrency_and_warn_once(adaptive, monkeypatch,
                                                                  table_pdf, caplog):  # noqa: F811
    extractor = adaptive(_Metrics(fail=True), VLM_CONCURRENCY="3", VLM_ADAPT_INTERVAL="0.02")
    srv = _install(monkeypatch, _Server(delay=0.1))
    with caplog.at_level(logging.WARNING, logger="insurance_chunker.extractor"):
        _, vlm = _run(extractor, table_pdf)
        time.sleep(0.1)
    assert srv.peak == 3 and vlm.metrics["concurrency_range"] == [3, 3]
    warned = [r for r in caplog.records if getattr(r, "event", "") == "vlm_adapt_unavailable"]
    assert len(warned) == 1


def test_circuit_still_bounds_wasted_calls_under_a_lowered_limit(adaptive, monkeypatch,
                                                                  table_pdf):  # noqa: F811
    """스레드 8개가 상한 2 앞에서 기다리다 차단기가 열리면, 기다리던 것은 부르지 않는다."""
    from insurance_chunker import vlm_limiter
    extractor = adaptive(_Metrics(), VLM_CONCURRENCY="8", VLM_FAIL_STREAK="2")
    vlm_limiter.get().limit = 2
    srv = _install(monkeypatch, _Server(delay=0.05, fail=True))
    _, vlm = _run(extractor, table_pdf)
    assert 2 <= srv.calls <= 2 + 2 - 1, f"호출 {srv.calls}회"
    assert vlm.metrics["circuit_open"] is True


@pytest.mark.parametrize("env", [{"VLM_ADAPTIVE": "0", "VLM_CONCURRENCY": "8"},
                                 {"VLM_ADAPTIVE": "1", "VLM_CONCURRENCY": "1"}])
def test_no_limiter_when_off_or_sequential(adaptive, monkeypatch, table_pdf, env):  # noqa: F811
    from insurance_chunker import vlm_limiter
    metrics = _Metrics()
    extractor = adaptive(metrics, **env)
    assert vlm_limiter.get() is None
    _install(monkeypatch, _Server())
    _, vlm = _run(extractor, table_pdf)
    assert vlm.metrics["adaptive"] is False and metrics.polls == 0


def test_adaptive_is_on_by_default(monkeypatch):
    from insurance_chunker import vlm_limiter
    monkeypatch.delenv("VLM_ADAPTIVE", raising=False)
    assert vlm_limiter.settings()["adaptive"] is True
