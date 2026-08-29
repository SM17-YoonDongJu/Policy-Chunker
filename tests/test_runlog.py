"""runlog / healthcheck 계약 검증 — 격리 판정과 좀비 감지가 의도대로 도는지 확인한다.

이 두 가지는 운영에서만 드러나는 로직이라(주기가 7일) 실수해도 한참 뒤에야 안다.
그래서 파일 상태를 직접 만들어 판정만 떼어 검사한다 — DB도 컨테이너도 필요 없다.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def rl(tmp_path, monkeypatch):
    """상태 디렉터리를 tmp로 돌린 runlog 모듈. 모듈 캐시(_dir_cache)를 끊으려고 새로 읽는다."""
    monkeypatch.setenv("INGEST_STATE_DIR", str(tmp_path / "state"))
    import runlog
    importlib.reload(runlog)
    return runlog


def test_state_dir_falls_back_when_unwritable(tmp_path, monkeypatch):
    """마운트가 없거나 권한이 없으면 로컬로 떨어져야 한다 — 이력 때문에 데몬이 죽으면 안 된다."""
    monkeypatch.setenv("INGEST_STATE_DIR", "/proc/nonexistent/state")
    monkeypatch.chdir(tmp_path)
    import runlog
    importlib.reload(runlog)
    assert runlog.state_dir() == Path(".state")


def test_ok_resets_the_retry_counter(rl):
    """실패가 쌓였어도 한 번 성공하면 카운터가 0으로 돌아가야 한다.

    안 그러면 파일이 고쳐져 다시 적재된 문서가 예전 실패 횟수 때문에 계속 격리된다.
    """
    for _ in range(2):
        rl.record_item(sha256="abc", name="x.pdf", status="EMPTY")
    assert rl.attempt("abc")["attempts"] == 2

    rl.record_item(sha256="abc", name="x.pdf", status="OK", chunks=10)
    assert rl.attempt("abc")["attempts"] == 0


def test_skipped_is_not_an_attempt(rl):
    """이미 적재돼 건너뛴 건 시도가 아니다 — 이걸 세면 멀쩡한 문서가 격리된다."""
    for _ in range(5):
        rl.record_item(sha256="abc", name="x.pdf", status="SKIPPED")
    assert rl.attempt("abc") is None


def test_skipped_still_lands_in_the_item_log(rl):
    """카운터에는 안 세더라도 이력에는 남아야 한다 — 멱등 스킵률의 유일한 근거다."""
    rl.record_item(sha256="abc", name="x.pdf", status="SKIPPED", source="catalog")
    lines = (rl.state_dir() / "items.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "SKIPPED"


def test_quarantined_does_not_touch_the_counter(rl):
    """격리는 새 시도가 아니다.

    여기서 카운터를 올리면 주기마다 늘어나기만 하고, status가 QUARANTINED로 덮여
    원래 실패 사유(EMPTY인지 ERROR인지)를 잃는다.
    """
    for _ in range(3):
        rl.record_item(sha256="abc", name="x.pdf", status="EMPTY")
    before = rl.attempt("abc")

    for _ in range(4):  # 이후 주기마다 격리로 건너뛴다
        rl.record_item(sha256="abc", name="x.pdf", status="QUARANTINED")

    after = rl.attempt("abc")
    assert after["attempts"] == before["attempts"] == 3
    assert after["status"] == "EMPTY"          # 원래 사유가 보존된다
    assert after["last_at"] == before["last_at"]


def test_quarantined_still_lands_in_the_item_log(rl):
    """무엇을 건너뛰었는지는 남아야 지표·감사에서 보인다."""
    rl.record_item(sha256="abc", name="x.pdf", status="QUARANTINED", error="EMPTY 3회 연속")
    rec = json.loads((rl.state_dir() / "items.jsonl").read_text(encoding="utf-8").strip())
    assert rec["status"] == "QUARANTINED"
    assert rec["error"] == "EMPTY 3회 연속"


def test_empty_and_error_accumulate_toward_quarantine(rl):
    """0청크와 오류는 같은 카운터를 쓴다 — 원인이 무엇이든 진도가 안 나가는 건 같다."""
    rl.record_item(sha256="abc", name="x.pdf", status="EMPTY")
    rl.record_item(sha256="abc", name="x.pdf", status="ERROR", error="boom")
    hist = rl.attempt("abc")
    assert hist["attempts"] == 2
    assert hist["status"] == "ERROR"


def test_phase_records_time_even_when_the_body_raises(rl):
    """실패한 문서가 어느 단계에서 시간을 썼는지도 병목 판단에 필요하다."""
    timings: dict[str, float] = {}
    with pytest.raises(ValueError):
        with rl.phase(timings, "parse"):
            raise ValueError("boom")
    assert "parse" in timings


def test_cycle_success_updates_last_success(rl):
    rl.record_cycle(False, "실패")
    assert "last_success_at" not in rl.daemon_state()
    rl.record_cycle(True, "완료")
    assert rl.daemon_state()["last_success_at"]


def test_last_run_reads_the_most_recent_line(rl):
    rl.record_run({"source": "catalog", "ok": 1})
    rl.record_run({"source": "catalog", "ok": 7})
    assert rl.last_run()["ok"] == 7


# ── healthcheck ───────────────────────────────────────────────────────────────

def _health(rl, monkeypatch, interval="100"):
    monkeypatch.setenv("INGEST_INTERVAL_SECONDS", interval)
    import healthcheck
    importlib.reload(healthcheck)
    monkeypatch.setattr(healthcheck, "runlog", rl)
    return healthcheck.main()


def test_unhealthy_without_any_state(rl, monkeypatch):
    """상태 파일이 아예 없으면 데몬이 안 떴다는 뜻이다."""
    assert _health(rl, monkeypatch) == 1


def test_healthy_during_the_first_cycle(rl, monkeypatch):
    """첫 기동 직후엔 성공 이력이 없다 — 유예 안이면 healthy여야 한다."""
    rl.record_start()
    assert _health(rl, monkeypatch) == 0


def test_unhealthy_when_the_first_cycle_never_succeeds(rl, monkeypatch):
    """기동만 하고 계속 실패하면 유예를 넘긴 시점에 unhealthy로 드러나야 한다."""
    stale = (datetime.now(UTC) - timedelta(seconds=1000)).isoformat()
    rl._write_json("daemon.json", {"started_at": stale})
    assert _health(rl, monkeypatch) == 1


def test_unhealthy_when_the_last_success_is_too_old(rl, monkeypatch):
    """프로세스는 살아 있는데 인덱싱만 계속 실패하는 좀비 — 이게 잡고 싶은 상태다."""
    stale = (datetime.now(UTC) - timedelta(seconds=1000)).isoformat()
    rl._write_json("daemon.json", {"started_at": stale, "last_success_at": stale})
    assert _health(rl, monkeypatch) == 1


def test_healthy_when_a_recent_cycle_succeeded(rl, monkeypatch):
    rl.record_cycle(True, "완료")
    assert _health(rl, monkeypatch) == 0


# ── metrics ───────────────────────────────────────────────────────────────────

def _metrics(rl, monkeypatch):
    import metrics
    importlib.reload(metrics)
    monkeypatch.setattr(metrics, "runlog", rl)
    return metrics


def test_metrics_on_empty_history(rl, monkeypatch):
    """이력이 하나도 없어도 죽지 않아야 한다 — 첫 배포 직후가 이 상태다."""
    m = _metrics(rl, monkeypatch).collect()
    assert m["documents"]["considered"] == 0
    assert m["per_document_seconds"]["p50"] == 0


def test_metrics_counts_skips_separately_from_attempts(rl, monkeypatch):
    """멱등 스킵은 '실패'가 아니라 '아낀 일'이다 — 성공률 분모에서 빠져야 한다."""
    rl.record_item(sha256="a", name="a.pdf", status="OK", chunks=10, elapsed_s=4.0,
                   phases={"parse": 3.0, "embed": 1.0})
    rl.record_item(sha256="b", name="b.pdf", status="EMPTY", elapsed_s=1.0)
    for s in ("c", "d"):
        rl.record_item(sha256=s, name=f"{s}.pdf", status="SKIPPED")

    m = _metrics(rl, monkeypatch).collect()
    assert m["documents"]["considered"] == 4
    assert m["documents"]["success_rate"] == 0.5          # OK 1 / 시도 2
    assert m["documents"]["idempotent_skip_rate"] == 0.5  # SKIPPED 2 / 전체 4


def test_metrics_ranks_phases_by_total_time(rl, monkeypatch):
    """무엇을 먼저 고칠지 정하는 근거 — 가장 오래 걸린 단계가 맨 앞에 와야 한다."""
    rl.record_item(sha256="a", name="a.pdf", status="OK", chunks=1, elapsed_s=10.0,
                   phases={"parse": 2.0, "embed": 8.0})
    m = _metrics(rl, monkeypatch).collect()
    assert list(m["phase_share"]) == ["embed", "parse"]
    assert m["phase_share"]["embed"]["share"] == 0.8


# ── 경계 검출 신뢰도 (품질 신호) ──────────────────────────────────────────────

def test_weak_boundary_is_recorded_alongside_success(rl):
    """경계를 못 잡아도 적재는 성공한다 — status만 보면 품질 저하가 안 보인다.

    섹션이 안 갈리면 조번호가 전부 어긋나는데(IMPROVEMENT_LOG C-1), 그 문서도
    status=OK로 남는다. 그래서 신뢰도를 따로 들고 간다.
    """
    rl.record_item(sha256="a", name="KB약관.pdf", status="OK", chunks=300,
                   boundary_confidence="weak")
    rec = json.loads((rl.state_dir() / "items.jsonl").read_text(encoding="utf-8").strip())
    assert rec["status"] == "OK"
    assert rec["boundary_confidence"] == "weak"


def test_weak_boundary_does_not_affect_retry_counter(rl):
    """품질 저하지 실패가 아니다 — 격리하면 그 보험사 문서가 영영 안 들어온다."""
    rl.record_item(sha256="a", name="x.pdf", status="OK", chunks=300,
                   boundary_confidence="weak")
    assert rl.attempt("a")["attempts"] == 0


def test_weak_boundary_counter_in_metrics(rl, monkeypatch):
    import metrics
    importlib.reload(metrics)
    import exporter
    importlib.reload(exporter)
    monkeypatch.setattr(exporter, "runlog", rl)
    monkeypatch.setattr(exporter, "metrics_mod", metrics)

    for name, conf in (("a.pdf", "weak"), ("b.pdf", "ok"), ("c.pdf", "weak")):
        rl.record_item(sha256=name, name=name, status="OK", chunks=10,
                       boundary_confidence=conf)

    samples = {s.name: s.value for f in exporter.RunlogCollector().collect()
               for s in f.samples}
    assert samples["insurance_chunker_weak_boundary_documents"] == 2.0
