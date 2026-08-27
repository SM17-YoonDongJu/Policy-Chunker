"""격리 판정과 데몬 사이클 판정 — 운영에서만 드러나는 두 결정을 떼어 검사한다.

둘 다 주기가 7일이라 잘못돼도 한참 뒤에 안다. 격리가 과하면 멀쩡한 문서가 영영 안 들어오고,
모자라면 못 읽는 문서를 매 주기 다시 받는다. 사이클 판정이 틀리면 healthcheck가 거짓말한다.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ingest_catalog  # noqa: E402

# ── 격리 판정 ─────────────────────────────────────────────────────────────────

def test_no_history_is_not_quarantined():
    """처음 보는 문서는 당연히 처리 대상이다."""
    assert ingest_catalog._should_quarantine(None, retry_ok=False) is False


def test_below_the_limit_still_retries():
    assert ingest_catalog._should_quarantine({"attempts": 1}, retry_ok=False) is False


def test_at_the_limit_is_quarantined():
    """상한에 도달하면 다운로드조차 하지 않는다 — 이게 무한 재처리를 끊는 지점이다."""
    n = ingest_catalog._MAX_RETRY
    assert ingest_catalog._should_quarantine({"attempts": n}, retry_ok=False) is True


def test_retry_flag_overrides_quarantine():
    """--retry-quarantined / --overwrite로 사람이 강제로 다시 돌릴 수 있어야 한다."""
    n = ingest_catalog._MAX_RETRY
    assert ingest_catalog._should_quarantine({"attempts": n}, retry_ok=True) is False


# ── 데몬 사이클 판정 ──────────────────────────────────────────────────────────

@pytest.fixture
def wk(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("INGEST_SOURCE", "catalog")
    import runlog
    importlib.reload(runlog)
    import worker
    importlib.reload(worker)
    # 알림은 웹훅 미설정이면 조용히 건너뛴다. 여기선 호출 여부만 본다.
    sent: list[str] = []
    monkeypatch.setattr(worker.notify, "notify",
                        lambda status, title, fields: sent.append(status) or True)
    worker._sent = sent
    return worker


def test_failed_ingest_does_not_mark_success(wk, monkeypatch):
    """적재가 실패하면 마지막 성공 시각이 갱신되면 안 된다 — healthcheck의 근거가 썩는다."""
    monkeypatch.setattr(wk, "_run", lambda label, argv: False)
    wk._cycle()
    assert "last_success_at" not in wk.runlog.daemon_state()
    assert wk._sent == ["failure"]


def test_failed_ingest_skips_search_terms(wk, monkeypatch):
    """부분 상태로 덮지 않기 위해 적재 실패 시 rebuild_search_terms는 돌지 않아야 한다."""
    calls: list[str] = []

    def _fake_run(label, argv):
        calls.append(label)
        return False

    monkeypatch.setattr(wk, "_run", _fake_run)
    wk._cycle()
    assert calls == ["ingest_catalog"]


def test_successful_cycle_marks_success(wk, monkeypatch):
    monkeypatch.setattr(wk, "_run", lambda label, argv: True)
    wk._cycle()
    assert wk.runlog.daemon_state()["last_success_at"]
    assert wk._sent == ["success"]


def test_search_terms_failure_is_a_warning_not_a_failure(wk, monkeypatch):
    """색인은 갱신됐고 BM25 용어만 낡은 상태다 — 사이클을 실패로 치면 과하게 운다."""
    monkeypatch.setattr(wk, "_run", lambda label, argv: label == "ingest_catalog")
    wk._cycle()
    assert wk.runlog.daemon_state()["last_success_at"]
    assert wk._sent == ["warning"]
