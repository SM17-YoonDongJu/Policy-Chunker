"""정지 신호가 문서 경계까지 닿는지 — 배포가 진행 중인 인덱싱을 자를 때의 동작.

배포는 컨테이너를 재생성한다. SIGTERM은 PID 1(worker)에만 오고 유예가 끝나면 SIGKILL이
전부 끊는다. 그 사이에 "시작한 문서는 끝내고, 안 시작한 문서는 손대지 않고, 그 사실을
이력에 남기고" 빠지는 것이 여기서 검사하는 계약이다.

문서를 중간에 끊지 않는 건 의도다 — 저장 트랜잭션의 단위가 문서라, 끊어도 다음 주기에
처음부터 다시 해야 한다.
"""
from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ingest_catalog  # noqa: E402
import shutdown  # noqa: E402


@pytest.fixture(autouse=True)
def clean_stop_state():
    """정지 플래그는 프로세스 전역이라 테스트끼리 샌다."""
    shutdown.reset()
    prev = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    shutdown.reset()
    for s, h in prev.items():
        signal.signal(s, h)


# ── 스위치 ────────────────────────────────────────────────────────────────────

def test_sigterm_becomes_a_stop_request():
    """죽는 게 아니라 정지 예약이어야 한다 — 기본 동작이면 진행 중인 문서가 통째로 날아간다."""
    called: list[str] = []
    shutdown.install(on_stop=lambda: called.append("forwarded"))
    assert not shutdown.stopping()

    signal.raise_signal(signal.SIGTERM)

    assert shutdown.stopping()
    assert called == ["forwarded"], "자식에게 전달할 훅이 신호 즉시 불려야 한다"


def test_wait_wakes_up_on_the_signal():
    """주기 대기(기본 7일)가 신호에 즉시 깨어나야 종료가 지연되지 않는다."""
    shutdown.install()
    signal.raise_signal(signal.SIGTERM)
    assert shutdown.event().wait(0.01) is True


def test_install_without_hook_is_fine():
    """CLI는 전달할 자식이 없어 훅 없이 설치한다."""
    shutdown.install()
    signal.raise_signal(signal.SIGINT)
    assert shutdown.stopping()


# ── 카탈로그 적재: 문서 경계에서 멈춘다 ──────────────────────────────────────

def _tasks(*names: str) -> list[dict]:
    return [{"name": n, "row": {"sha256": n, "company": "삼성화재", "product_name": "실손"}}
            for n in names]


def test_stops_between_documents(monkeypatch):
    """정지 신호 뒤의 문서는 시작조차 하지 않는다."""
    seen: list[str] = []

    def fake(task):
        seen.append(task["name"])
        shutdown.request_stop()  # 첫 문서를 처리하는 도중에 배포가 났다
        return {"status": "OK", "chunks": 1}

    monkeypatch.setattr(ingest_catalog, "_process_one", fake)
    out = list(ingest_catalog._run_tasks(_tasks("a", "b", "c"), concurrency=1))

    assert seen == ["a"], "두 번째 문서는 손도 대지 않아야 한다"
    assert len(out) == 1, "끝낸 문서의 결과는 그대로 나와야 한다(이력에 남는다)"


def test_finishes_the_document_it_started(monkeypatch):
    """신호가 왔다고 진행 중인 문서를 버리지 않는다 — 버려도 다음 주기에 처음부터 다시다."""
    def fake(task):
        shutdown.request_stop()
        return {"status": "OK", "chunks": 7}

    monkeypatch.setattr(ingest_catalog, "_process_one", fake)
    out = list(ingest_catalog._run_tasks(_tasks("a", "b"), concurrency=1))

    assert [r["chunks"] for _, r in out] == [7]


def test_runs_everything_when_not_stopping(monkeypatch):
    """평상시에는 하나도 빠뜨리지 않는다."""
    monkeypatch.setattr(ingest_catalog, "_process_one",
                        lambda task: {"status": "OK", "chunks": 1})
    out = list(ingest_catalog._run_tasks(_tasks("a", "b", "c"), concurrency=1))
    assert [t["name"] for t, _ in out] == ["a", "b", "c"]
