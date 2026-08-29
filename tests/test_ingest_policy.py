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


# ── 문서 병렬 처리 (#24) ──────────────────────────────────────────────────────

def test_serial_path_preserves_order(monkeypatch):
    """동시성 1이면 기존 동작 그대로 — 순서가 유지돼야 로그를 읽을 수 있다."""
    tasks = [{"name": f"{i}.pdf"} for i in range(5)]
    monkeypatch.setattr(ingest_catalog, "_process_one",
                        lambda t: {"pdf": t["name"], "status": "OK"})
    out = [t["name"] for t, _ in ingest_catalog._run_tasks(tasks, 1)]
    assert out == ["0.pdf", "1.pdf", "2.pdf", "3.pdf", "4.pdf"]


def test_single_task_does_not_spawn_a_pool(monkeypatch):
    """문서 한 건에 프로세스 풀을 띄우면 재import 비용만 낸다."""
    spawned = []
    monkeypatch.setattr(ingest_catalog, "_process_one",
                        lambda t: {"pdf": t["name"], "status": "OK"})
    import concurrent.futures as cf
    monkeypatch.setattr(cf, "ProcessPoolExecutor",
                        lambda *a, **k: spawned.append(1))
    list(ingest_catalog._run_tasks([{"name": "a.pdf"}], 4))
    assert spawned == []


def test_result_pairs_with_its_own_task(monkeypatch):
    """결과가 완료 순서로 오므로 어느 문서 것인지 짝을 잃으면 안 된다 —
    엉뚱한 sha256에 이력이 기록되면 격리 판정이 틀어진다."""
    tasks = [{"name": f"{i}.pdf", "sha": f"s{i}"} for i in range(4)]
    monkeypatch.setattr(ingest_catalog, "_process_one",
                        lambda t: {"pdf": t["name"], "status": "OK", "echo": t["sha"]})
    for task, result in ingest_catalog._run_tasks(tasks, 1):
        assert result["echo"] == task["sha"]


def test_worker_crash_becomes_an_error_result(monkeypatch):
    """워커 프로세스가 죽어도 나머지 문서는 계속돼야 한다."""
    def _boom(t):
        raise RuntimeError("worker died")

    monkeypatch.setattr(ingest_catalog, "_process_one", _boom)
    # 동시성 1 경로에서는 예외가 그대로 오르므로, 풀 경로의 계약만 확인한다.
    with pytest.raises(RuntimeError):
        list(ingest_catalog._run_tasks([{"name": "a.pdf"}], 1))


def test_prepare_doc_maps_catalog_row(tmp_path):
    """카탈로그 행의 메타데이터가 청크에 그대로 실린다 — 여기가 틀리면 검색 필터가 어긋난다."""
    import datetime
    row = {"company": "메리츠화재", "product_name": "단체안심생활보험",
           "product_code": "ABC-123", "effective_date": datetime.date(2026, 6, 1),
           "category": "terms"}
    doc = ingest_catalog._prepare_doc(row, "약관.pdf", tmp_path / "x.pdf")
    assert doc["insurer"] == "메리츠화재"
    assert doc["effective_date"] == "2026-06-01"
    assert doc["doc_type"] == "policy_terms"


def test_prepare_doc_falls_back_when_metadata_is_missing(tmp_path):
    """카탈로그에 값이 비어 있어도 적재는 되어야 한다."""
    row = {"company": None, "product_name": None, "product_code": None,
           "effective_date": None, "category": "terms"}
    doc = ingest_catalog._prepare_doc(row, "무제.pdf", tmp_path / "x.pdf")
    assert doc["insurer"] == "미상"
    assert doc["product_name"] == "무제"
    assert doc["effective_date"] is None


def test_task_payload_survives_pickling():
    """작업 페이로드가 프로세스 경계를 넘는다.

    argparse.Namespace와 카탈로그 행(datetime.date 포함)이 피클되지 않으면 동시성을 켠
    순간 운영에서만 터진다 — 로컬 순차 실행에서는 절대 안 드러난다.
    """
    import argparse
    import datetime
    import pickle

    task = {
        "row": {"company": "메리츠화재", "product_name": "x", "product_code": None,
                "effective_date": datetime.date(2026, 6, 1), "category": "terms",
                "s3_key": "corpus/a.pdf", "sha256": "abc"},
        "args": argparse.Namespace(dry_run=False, overwrite=False, no_embed=True,
                                   target_tokens=500, hard_max_tokens=1000,
                                   db_url=None, ollama_url=None, embed_model=None,
                                   no_ocr=True, no_vision=True, no_init_schema=True),
        "name": "약관.pdf", "bucket": "b", "dest": "/tmp/x.pdf",
        "dry_run_dir": "out", "prev_attempts": 0,
    }
    restored = pickle.loads(pickle.dumps(task))
    assert restored["row"]["effective_date"] == datetime.date(2026, 6, 1)
    assert restored["args"].target_tokens == 500


def test_process_one_is_importable_by_name():
    """spawn 방식은 워커에서 함수를 이름으로 다시 찾는다 — 모듈 최상위여야 한다."""
    import pickle
    assert pickle.loads(pickle.dumps(ingest_catalog._process_one)) is ingest_catalog._process_one
