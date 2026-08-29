"""임베딩 경로 계약 — 실패를 드러내는지, 평가와 같은 전처리를 하는지 확인한다.

두 가지가 조용히 어긋날 수 있어서 테스트로 못 박는다.
  1) 배치가 깨져 건별 폴백으로 떨어지면 크게 느려지는데, 예전엔 로그에 흔적이 없었다.
  2) 장문 절단 상한이 eval과 다르면 색인 벡터와 평가 벡터가 달라져 R@5 수치가 재현되지 않는다.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def emb(monkeypatch):
    monkeypatch.setenv("EMBED_BACKEND", "ollama")
    monkeypatch.setenv("EMBED_RETRY_DELAY", "0")  # 테스트가 재시도 대기로 늘어지지 않게
    from insurance_chunker import embedder
    importlib.reload(embedder)
    return embedder


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload or {}, text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _vec(n=1024):
    return [0.0] * n


# ── 배치 실패가 로그에 남는가 ─────────────────────────────────────────────────

@pytest.mark.parametrize("resp,expected", [
    (_Resp(500, text="internal error"), "HTTP 500"),
    (_Resp(200, {"embeddings": []}), "embeddings 없음"),
    (_Resp(200, {"embeddings": [_vec()]}), "개수 불일치"),
])
def test_batch_failure_reason_is_logged(emb, monkeypatch, caplog, resp, expected):
    """사유별로 구분돼야 한다 — 타임아웃인지 500인지 개수 불일치인지 모르면 못 고친다."""
    monkeypatch.setattr(emb.requests, "post", lambda *a, **k: resp)
    with caplog.at_level("WARNING"):
        assert emb._ollama_batch(["a", "b"], "http://x", "m") is None
    assert expected in caplog.text


def test_batch_exception_reason_is_logged(emb, monkeypatch, caplog):
    def _boom(*a, **k):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(emb.requests, "post", _boom)
    with caplog.at_level("WARNING"):
        assert emb._ollama_batch(["a"], "http://x", "m") is None
    assert "TimeoutError" in caplog.text


def test_fallback_count_is_reported(emb, monkeypatch, caplog):
    """embed 시간이 실제 추론인지 폴백 대기인지 구분할 유일한 단서다."""
    monkeypatch.setattr(emb, "_BATCH_SIZE", 2)
    monkeypatch.setattr(emb.requests, "post", lambda *a, **k: _Resp(503, text="busy"))
    monkeypatch.setattr(emb, "_ollama_single", lambda t, u, m: _vec())

    with caplog.at_level("WARNING"):
        out = emb._embed_ollama(["a", "b", "c", "d"], "http://x", "m")
    assert len(out) == 4
    assert "2배치가 건별 폴백" in caplog.text


def test_successful_batch_does_not_warn(emb, monkeypatch, caplog):
    monkeypatch.setattr(emb, "_BATCH_SIZE", 2)
    monkeypatch.setattr(emb.requests, "post",
                        lambda *a, **k: _Resp(200, {"embeddings": [_vec(), _vec()]}))
    with caplog.at_level("WARNING"):
        assert len(emb._embed_ollama(["a", "b"], "http://x", "m")) == 2
    assert "폴백" not in caplog.text


# ── 장문 절단 ─────────────────────────────────────────────────────────────────

def test_long_text_is_truncated_to_the_eval_limit(emb):
    """eval/retrieval_eval.py:150과 같은 1800자 — 다르면 평가 수치가 재현되지 않는다."""
    assert emb._MAX_CHARS == 1800
    assert len(emb._truncate(["가" * 5000])[0]) == 1800


def test_empty_text_becomes_a_space(emb):
    """Ollama는 빈 입력에 오류를 낸다 — 청크 하나 때문에 배치 전체가 깨지면 안 된다."""
    assert emb._truncate([""]) == [" "]


def test_short_text_is_untouched(emb):
    assert emb._truncate(["짧은 조항"]) == ["짧은 조항"]


def test_truncation_applies_to_queries_too(emb, monkeypatch):
    """문서만 자르고 질의를 안 자르면 비대칭이 생겨 검색 품질이 어긋난다."""
    seen = {}

    def _capture(url, json=None, timeout=None):
        seen["input"] = json["input"]
        return _Resp(200, {"embeddings": [_vec()]})

    monkeypatch.setattr(emb.requests, "post", _capture)
    emb.embed_query("가" * 5000)
    assert all(len(t) <= emb._MAX_CHARS for t in seen["input"])


# ── 설정 ──────────────────────────────────────────────────────────────────────

def test_batch_size_is_configurable(monkeypatch):
    """GPU를 연속으로 먹이려면 배치 크기를 실측으로 찾아야 한다 — 재빌드 없이 조정."""
    monkeypatch.setenv("EMBED_BATCH_SIZE", "128")
    from insurance_chunker import embedder
    importlib.reload(embedder)
    assert embedder._BATCH_SIZE == 128
