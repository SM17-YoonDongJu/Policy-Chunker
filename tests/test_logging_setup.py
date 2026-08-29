"""구조화 로깅 계약 — Alloy·Loki 파이프라인이 기대하는 형태인지 확인한다.

Alloy가 컨테이너 stdout을 JSON으로 파싱해 level을 라벨로 승격한다. 한 줄이라도 JSON이
아니거나 level 필드가 없으면 그 줄은 라벨 없이 들어가 검색·알림에서 빠진다.
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging_setup  # noqa: E402


@pytest.fixture
def log_lines(monkeypatch, capsys):
    """LOG_FORMAT을 걸고 로깅한 뒤 stderr 줄을 돌려주는 헬퍼를 만든다."""
    def _run(emit, **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        importlib.reload(logging_setup)
        logging_setup.configure()
        emit(logging.getLogger("test"))
        return [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    return _run


def test_json_mode_emits_one_object_per_line(log_lines):
    lines = log_lines(lambda log: log.info("인덱싱 데몬 시작"), LOG_FORMAT="json")
    rec = json.loads(lines[0])
    assert rec["level"] == "INFO"
    assert rec["service"] == "insurance-chunker"
    assert rec["message"] == "인덱싱 데몬 시작"
    assert rec["logger"] == "test"


def test_extra_context_is_carried_into_the_payload(log_lines):
    """문서 결과를 Loki에서 집계하려면 extra가 필드로 실려야 한다."""
    lines = log_lines(
        lambda log: log.info("문서 처리 완료", extra={
            "event": "document_done", "status": "OK", "chunks": 687,
            "phases": {"parse": 12.4, "embed": 22.8},
        }),
        LOG_FORMAT="json")
    rec = json.loads(lines[0])
    assert rec["event"] == "document_done"
    assert rec["chunks"] == 687
    assert rec["phases"]["embed"] == 22.8


def test_exception_is_captured_as_a_field(log_lines):
    def _emit(log):
        try:
            raise RuntimeError("cannot open broken document")
        except RuntimeError:
            log.exception("처리 실패")

    rec = json.loads(log_lines(_emit, LOG_FORMAT="json")[0])
    assert rec["level"] == "ERROR"
    assert "cannot open broken document" in rec["exception"]


def test_unserializable_value_does_not_lose_the_line(log_lines):
    """직렬화 못 하는 값 하나 때문에 로그 줄이 통째로 사라지면 안 된다."""
    lines = log_lines(lambda log: log.info("x", extra={"obj": object()}), LOG_FORMAT="json")
    assert json.loads(lines[0])["message"] == "x"


def test_korean_is_not_escaped(log_lines):
    """ensure_ascii=False — Loki에서 사람이 그대로 읽을 수 있어야 한다."""
    lines = log_lines(lambda log: log.info("0청크 — 적재 없음"), LOG_FORMAT="json")
    assert "0청크" in lines[0]


def test_text_mode_stays_human_readable(log_lines):
    """로컬 기본값은 평문이다 — extra는 본문을 어지럽히지 않는다."""
    lines = log_lines(lambda log: log.info("완료", extra={"event": "cycle_done"}),
                      LOG_FORMAT="text")
    assert "완료" in lines[0]
    assert not lines[0].startswith("{")


def test_configure_is_idempotent(log_lines):
    """진입점이 여러 번 불려도 핸들러가 쌓여 줄이 중복되면 안 된다."""
    def _emit(log):
        logging_setup.configure()
        logging_setup.configure()
        log.info("한 번만")

    assert len(log_lines(_emit, LOG_FORMAT="json")) == 1


def test_log_level_is_configurable(log_lines):
    lines = log_lines(lambda log: log.debug("보이면 안 됨"), LOG_FORMAT="json", LOG_LEVEL="INFO")
    assert lines == []
