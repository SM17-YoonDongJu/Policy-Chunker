"""/metrics 계약 — Prometheus 스크랩이 기대하는 형태인지 확인한다.

배치 잡이라 backend와 지표 성격이 다르다. 주기가 7일이면 대부분의 시간 동안 카운터는
안 움직이므로, 주 신호는 rate가 아니라 "마지막 실행 상태" 게이지다. 여기서는 그 게이지가
runlog 상태를 정확히 반영하는지, 그리고 패널에 구멍을 내지 않는지를 본다.
"""
from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def exp(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_STATE_DIR", str(tmp_path / "state"))
    import runlog
    importlib.reload(runlog)
    import metrics
    importlib.reload(metrics)
    import exporter
    importlib.reload(exporter)
    monkeypatch.setattr(exporter, "runlog", runlog)
    monkeypatch.setattr(exporter, "metrics_mod", metrics)
    exporter._rl = runlog
    return exporter


def _samples(exp) -> dict:
    """이름(+라벨) → 값으로 평탄화한다."""
    out = {}
    for family in exp.RunlogCollector().collect():
        for s in family.samples:
            key = s.name + (f"|{sorted(s.labels.items())}" if s.labels else "")
            out[key] = s.value
    return out


def test_freshness_reflects_the_last_success(exp):
    """신선도가 이 서비스의 유일한 SLI다 — 알림이 이 값 하나에 걸린다."""
    exp._rl.record_cycle(True, "완료")
    ts = _samples(exp)["insurance_chunker_last_success_timestamp_seconds"]
    assert abs(datetime.now(UTC).timestamp() - ts) < 5


def test_no_success_means_no_freshness_metric(exp):
    """한 번도 성공한 적 없으면 값을 지어내지 않는다 — 0을 내면 1970년으로 읽힌다."""
    exp._rl.record_start()
    assert "insurance_chunker_last_success_timestamp_seconds" not in _samples(exp)


def test_failed_cycle_reports_zero_success(exp):
    exp._rl.record_cycle(False, "ingest_catalog 실패")
    assert _samples(exp)["insurance_chunker_last_cycle_success"] == 0.0


def test_all_status_labels_are_always_present(exp):
    """0인 상태도 내보내야 대시보드 패널에 구멍이 안 생긴다."""
    exp._rl.record_run({"ok": 3, "skipped": 17})
    s = _samples(exp)
    for st in ("ok", "empty", "skipped", "quarantined", "error"):
        assert f"insurance_chunker_last_cycle_documents|[('status', '{st}')]" in s
    assert s["insurance_chunker_last_cycle_documents|[('status', 'skipped')]"] == 17.0
    assert s["insurance_chunker_last_cycle_documents|[('status', 'empty')]"] == 0.0


def test_quarantined_count_comes_from_attempt_history(exp):
    """격리 문서 수는 사이클 요약이 아니라 현재 상태(attempts.json)에서 세야 한다."""
    for _ in range(3):
        exp._rl.record_item(sha256="bad", name="스캔본.pdf", status="EMPTY")
    exp._rl.record_item(sha256="ok", name="약관.pdf", status="OK", chunks=10)
    assert _samples(exp)["insurance_chunker_quarantined_documents"] == 1.0


def test_cumulative_totals_span_multiple_cycles(exp):
    """누적 카운터는 사이클 경계를 넘어 쌓여야 rate()로 장기 추세를 볼 수 있다."""
    exp._rl.record_item(sha256="a", name="a.pdf", status="OK", chunks=100,
                        phases={"parse": 10.0, "embed": 5.0})
    exp._rl.record_run({"ok": 1, "total_chunks": 100})
    exp._rl.record_item(sha256="b", name="b.pdf", status="OK", chunks=200,
                        phases={"parse": 20.0, "embed": 7.0})
    exp._rl.record_run({"ok": 1, "total_chunks": 200})

    s = _samples(exp)
    assert s["insurance_chunker_chunks_indexed_total"] == 300.0
    assert s["insurance_chunker_phase_duration_seconds_total|[('phase', 'parse')]"] == 30.0
    # 마지막 사이클 게이지는 누적이 아니라 그 사이클 값만
    assert s["insurance_chunker_last_cycle_chunks_indexed"] == 200.0


def test_empty_state_does_not_crash(exp):
    """첫 배포 직후 상태다 — 스크랩이 500을 내면 TargetDown이 오발화한다."""
    s = _samples(exp)
    assert s["insurance_chunker_last_cycle_success"] == 0.0
    assert s["insurance_chunker_chunks_indexed_total"] == 0.0


def test_stale_success_is_still_reported(exp):
    """오래된 성공도 값으로 내야 한다 — 알림은 Prometheus가 time()과 비교해 판단한다."""
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    exp._rl._write_json("daemon.json", {"last_success_at": old, "last_run_ok": True})
    ts = _samples(exp)["insurance_chunker_last_success_timestamp_seconds"]
    assert ts == datetime.fromisoformat(old).timestamp()


def test_port_zero_disables_exposure(exp, monkeypatch):
    monkeypatch.setenv("METRICS_PORT", "0")
    assert exp.start() is False


def test_start_failure_does_not_raise(exp, monkeypatch):
    """포트 충돌 같은 실패가 데몬을 죽이면 안 된다."""
    monkeypatch.setenv("METRICS_PORT", "9101")
    monkeypatch.setattr(exp, "_max_retry", lambda: 3)
    import prometheus_client
    monkeypatch.setattr(prometheus_client, "start_http_server",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("address in use")))
    assert exp.start() is False
