"""/status 계약 — 청크 뷰어의 "처리 중" 패널이 기대하는 형태인지 확인한다.

핵심은 attempts.json의 STARTED를 둘로 가르는 것이다. 지금 처리 중인 문서와, 지난 시도가
끝을 못 본 문서(OOM·강제 종료)는 같은 STARTED로 남는다. 이번 실행이 살아 있고 그 시작 뒤에
시작한 것만 '처리 중'이어야 한다 — 틀리면 뷰어가 죽은 문서를 처리 중이라고 보여 준다.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def st(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_STATE_DIR", str(tmp_path / "state"))
    import runlog
    importlib.reload(runlog)
    import metrics
    importlib.reload(metrics)
    import exporter
    importlib.reload(exporter)
    monkeypatch.setattr(exporter, "runlog", runlog)
    monkeypatch.setattr(exporter, "metrics_mod", metrics)
    # 기록마다 1초씩 흐르는 시계 — 같은 마이크로초에 찍혀 순서가 뒤섞이는 일을 없앤다.
    clock = {"now": datetime(2026, 10, 30, 1, 0, tzinfo=UTC)}

    def now_iso() -> str:
        clock["now"] += timedelta(seconds=1)
        return clock["now"].isoformat()

    monkeypatch.setattr(runlog, "now_iso", now_iso)
    exporter._rl = runlog
    return exporter


def _shas(rows: list[dict]) -> list[str]:
    return [r["sha256"] for r in rows]


def test_processing_and_unfinished_are_told_apart(st):
    rl = st._rl
    rl.record_start()
    rl.record_attempt_start(sha256="old", name="old.pdf")  # 지난 실행에서 끝을 못 봤다
    rl.record_progress_start(total=3, pending=2, skipped=1, quarantined=0, concurrency=1)
    rl.record_attempt_start(sha256="a", name="a.pdf")

    snap = st.status_snapshot()
    assert snap["run"]["state"] == "running"
    assert _shas(snap["processing"]) == ["a"]
    assert _shas(snap["unfinished"]) == ["old"]


def test_progress_counts_only_this_run(st):
    """진행률의 분자는 이번 실행 안에서 실제로 처리한 건수다. 생략·지난 실행은 빼야 한다."""
    rl = st._rl
    rl.record_start()
    rl.record_item(sha256="prev", name="prev.pdf", status="OK", chunks=5)
    rl.record_item(sha256="s", name="s.pdf", status="SKIPPED")
    rl.record_progress_start(total=4, pending=2, skipped=1, quarantined=0, concurrency=1)
    for sha, status in (("a", "OK"), ("b", "EMPTY")):
        rl.record_attempt_start(sha256=sha, name=f"{sha}.pdf")
        rl.record_item(sha256=sha, name=f"{sha}.pdf", status=status, chunks=3)

    snap = st.status_snapshot()
    assert snap["run"]["done"] == 2 and snap["run"]["pending"] == 2
    assert _shas(snap["recent"]) == ["b", "a", "prev"]  # 최신부터, SKIPPED 제외
    assert snap["processing"] == [] and snap["unfinished"] == []


def test_dead_process_means_interrupted(st):
    """끝 기록 없이 프로세스가 사라졌으면 처리 중이 아니다(OOM으로 죽은 실행)."""
    rl = st._rl
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    rl.record_start()
    rl.record_progress_start(total=1, pending=1, skipped=0, quarantined=0, concurrency=1)
    prog = rl.progress()
    prog["pid"] = gone.pid
    rl._write_json("progress.json", prog)
    rl.record_attempt_start(sha256="a", name="a.pdf")

    snap = st.status_snapshot()
    assert snap["run"]["state"] == "interrupted"
    assert snap["processing"] == [] and _shas(snap["unfinished"]) == ["a"]


def test_daemon_restart_invalidates_the_old_run(st):
    """데몬이 다시 떴으면 그 전 실행의 pid는 다른 수명의 것이다 — 살아 있어 보여도 무효."""
    rl = st._rl
    rl.record_progress_start(total=1, pending=1, skipped=0, quarantined=0, concurrency=1)
    rl.record_start()
    assert st.status_snapshot()["run"]["state"] == "interrupted"


@pytest.mark.parametrize("stopped,state", [(False, "finished"), (True, "stopped")])
def test_finished_runs_have_nothing_in_progress(st, stopped, state):
    rl = st._rl
    rl.record_start()
    rl.record_progress_start(total=1, pending=1, skipped=0, quarantined=0, concurrency=1)
    rl.record_progress_end(stopped=stopped)
    snap = st.status_snapshot()
    assert snap["run"]["state"] == state and snap["run"]["finished_at"]


def test_empty_state_does_not_crash(st):
    snap = st.status_snapshot()
    assert snap["run"] is None
    assert snap["processing"] == [] and snap["unfinished"] == [] and snap["recent"] == []


def test_recent_respects_the_limit(st):
    for i in range(5):
        st._rl.record_item(sha256=f"s{i}", name=f"{i}.pdf", status="OK")
    assert _shas(st.status_snapshot(limit=2)["recent"]) == ["s4", "s3"]


def test_endpoint_serves_status_next_to_metrics(st):
    from prometheus_client import CollectorRegistry

    reg = CollectorRegistry()
    reg.register(st.RunlogCollector())
    st._rl.record_item(sha256="a", name="a.pdf", status="OK")
    server = st._serve("127.0.0.1", 0, reg)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/status?limit=1") as r:
            assert r.headers["Content-Type"].startswith("application/json")
            body = json.load(r)
        assert _shas(body["recent"]) == ["a"]
        assert {"daemon", "run", "processing", "unfinished", "interval_s"} <= set(body)

        with urllib.request.urlopen(f"{base}/metrics") as r:
            assert b"insurance_chunker_last_cycle_documents" in r.read()

        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{base}/status?limit=abc")
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_catalog_run_leaves_progress_for_the_viewer(st, monkeypatch):
    """ingest_catalog가 2단계 시작과 끝을 실제로 남기는지 — 배선이 빠지면 진행률이 안 뜬다."""
    import db.storage as storage
    import ingest_catalog
    import shutdown

    rows = [{"company": "A생명", "product_name": "P", "product_code": None,
             "effective_date": None, "category": "terms", "s3_key": f"corpus/terms/{sha}.pdf",
             "sha256": sha, "notion_file_name": f"{sha}.pdf", "part_order": 1}
            for sha in ("s1", "s2", "s3")]

    class FakeConn:
        def close(self) -> None:
            pass

    def fake_run_tasks(tasks, concurrency, on_start=None):
        for t in tasks:
            on_start(t)
            yield t, {"pdf": t["name"], "status": "OK", "chunks": 3, "warnings": 0,
                      "elapsed_s": 1.0, "phases": {}}

    monkeypatch.setattr(storage, "get_connection", lambda url=None: FakeConn())
    monkeypatch.setattr(storage, "init_schema", lambda conn, skip=False: None)
    monkeypatch.setattr(storage, "doc_already_ingested", lambda conn, sha: sha == "s1")
    monkeypatch.setattr(ingest_catalog, "_fetch_catalog", lambda conn, cat, limit: rows)
    monkeypatch.setattr(ingest_catalog, "_run_tasks", fake_run_tasks)
    monkeypatch.setattr(sys, "argv", ["ingest_catalog.py", "--bucket", "b"])
    shutdown.reset()
    try:
        ingest_catalog.main()
    finally:
        shutdown.reset()

    prog = st._rl.progress()
    assert (prog["total"], prog["pending"], prog["skipped"]) == (3, 2, 1)
    snap = st.status_snapshot()
    assert snap["run"]["state"] == "finished" and snap["run"]["done"] == 2
    assert snap["processing"] == []
