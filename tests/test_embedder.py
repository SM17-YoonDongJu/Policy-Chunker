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


# ── voyage ────────────────────────────────────────────────────────────────────

@pytest.fixture
def voy(monkeypatch):
    monkeypatch.setenv("EMBED_BACKEND", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.setenv("EMBED_RETRY_DELAY", "0")
    from insurance_chunker import embedder
    importlib.reload(embedder)
    yield embedder
    # 다른 테스트 모듈이 voyage 설정이 남은 모듈을 보지 않게 되돌린다
    monkeypatch.undo()
    importlib.reload(embedder)


def _voyage_ok(texts, tokens=3):
    """Voyage 응답 형식 — 항목 순서가 보장되지 않아 일부러 뒤집어 보낸다."""
    data = [{"object": "embedding", "embedding": [float(i)] * 1024, "index": i}
            for i in range(len(texts))]
    return _Resp(200, {"object": "list", "data": data[::-1], "model": "voyage-4-large",
                       "usage": {"total_tokens": tokens}})


def test_voyage_request_carries_contract(voy, monkeypatch):
    """검색 쪽(AI rag.search)과 같은 모델·차원이고, 문서는 document로 보낸다."""
    seen = {}

    def _capture(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json)
        return _voyage_ok(json["input"])

    monkeypatch.setattr(voy.requests, "post", _capture)
    out = voy.embed_texts(["가", "나"])
    assert seen["url"] == "https://ai.mongodb.com/v1/embeddings"
    assert seen["headers"] == {"Authorization": "Bearer test-key"}
    assert seen["json"] == {"model": "voyage-4-large", "input": ["가", "나"],
                            "input_type": "document", "output_dimension": 1024}
    assert [v[0] for v in out] == [0.0, 1.0]  # index 순서로 되돌린다


def test_voyage_query_has_no_instruct_prefix(voy, monkeypatch):
    """Voyage는 input_type=query에 자체 프롬프트를 붙인다 — qwen3 프리픽스를 또 붙이면 안 된다."""
    seen = {}

    def _capture(url, headers=None, json=None, timeout=None):
        seen.update(json)
        return _voyage_ok(json["input"])

    monkeypatch.setattr(voy.requests, "post", _capture)
    voy.embed_query("암 진단금")
    assert seen["input"] == ["암 진단금"]
    assert seen["input_type"] == "query"


def test_voyage_does_not_truncate_but_guards_empty(voy, monkeypatch):
    """1800자 상한은 qwen3 전용이다. 빈 문자열은 Voyage도 400이라 공백으로 바꾼다."""
    seen = {}

    def _capture(url, headers=None, json=None, timeout=None):
        seen["input"] = json["input"]
        return _voyage_ok(json["input"])

    monkeypatch.setattr(voy.requests, "post", _capture)
    voy.embed_texts(["가" * 5000, ""])
    assert seen["input"] == ["가" * 5000, " "]


def test_voyage_splits_batch_over_token_limit(voy, monkeypatch, caplog):
    """요청당 토큰 한도를 넘으면 나눠서 다시 보낸다 — 표가 긴 문서 하나가 실패로 끝나지 않게."""
    sizes = []

    def _post(url, headers=None, json=None, timeout=None):
        sizes.append(len(json["input"]))
        if len(json["input"]) > 2:
            return _Resp(400, {"detail": "The max allowed tokens per submitted batch is 120000.",
                               "error_code": "TOO_MANY_TOKENS_IN_BATCH"})
        return _voyage_ok(json["input"])

    monkeypatch.setattr(voy.requests, "post", _post)
    with caplog.at_level("INFO"):
        out = voy.embed_texts([f"t{i}" for i in range(8)])
    assert len(out) == 8
    assert sizes == [8, 4, 2, 2, 4, 2, 2]
    done = [r for r in caplog.records if getattr(r, "event", None) == "embed_done"]
    assert done[-1].split_batches == 3 and done[-1].tokens == 12


def test_voyage_retries_transient_errors(voy, monkeypatch):
    calls = []

    def _post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return _Resp(503, text="busy") if len(calls) == 1 else _voyage_ok(json["input"])

    monkeypatch.setattr(voy.requests, "post", _post)
    assert len(voy.embed_texts(["가"])) == 1
    assert len(calls) == 2


def test_voyage_gives_up_after_retry_max(voy, monkeypatch):
    def _boom(*a, **k):
        raise voy.requests.ConnectionError("connection reset")

    monkeypatch.setattr(voy.requests, "post", _boom)
    with pytest.raises(RuntimeError, match="3회"):
        voy.embed_texts(["가"])


@pytest.mark.parametrize("status", [401, 403, 404])
def test_voyage_config_errors_are_not_retried(voy, monkeypatch, status):
    """키·모델이 틀리면 기다려도 낫지 않는다 — 재시도로 문서마다 시간을 버리지 않는다."""
    calls = []

    def _post(*a, **k):
        calls.append(1)
        return _Resp(status, text='{"detail":"denied"}')

    monkeypatch.setattr(voy.requests, "post", _post)
    with pytest.raises(voy.EmbedConfigError, match=f"HTTP {status}"):
        voy.embed_texts(["가"])
    assert len(calls) == 1


def test_voyage_without_key_fails_before_request(voy, monkeypatch):
    monkeypatch.setattr(voy, "VOYAGE_API_KEY", "")
    monkeypatch.setattr(voy.requests, "post", lambda *a, **k: pytest.fail("요청을 보내면 안 된다"))
    with pytest.raises(voy.EmbedConfigError, match="VOYAGE_API_KEY"):
        voy.embed_texts(["가"])


def test_voyage_dimension_mismatch_is_an_error(voy, monkeypatch):
    bad = _Resp(200, {"data": [{"embedding": [0.0] * 512, "index": 0}], "usage": {}})
    monkeypatch.setattr(voy.requests, "post", lambda *a, **k: bad)
    with pytest.raises(RuntimeError, match="차원 불일치"):
        voy.embed_texts(["가"])


def test_voyage_model_is_the_default_unless_overridden(voy, monkeypatch):
    assert voy.model_name() == "voyage-4-large"
    assert voy.model_name("voyage-4") == "voyage-4"
    monkeypatch.setenv("EMBED_MODEL", "voyage-4-lite")
    importlib.reload(voy)
    assert voy.VOYAGE_MODEL == "voyage-4-lite"


# ── 적재 전 사전 확인 ─────────────────────────────────────────────────────────

def test_preflight_stops_on_config_error(voy, monkeypatch):
    """키가 틀린 채 사이클을 돌면 문서마다 VLM까지 돌고 실패해 격리 카운터만 쌓인다."""
    monkeypatch.setattr(voy.requests, "post", lambda *a, **k: _Resp(401, text="bad key"))
    with pytest.raises(voy.EmbedConfigError):
        voy.preflight()


def test_preflight_lets_transient_errors_through(voy, monkeypatch, caplog):
    monkeypatch.setattr(voy.requests, "post", lambda *a, **k: _Resp(503, text="busy"))
    with caplog.at_level("WARNING"):
        voy.preflight()
    assert "일시 오류로 보고 계속" in caplog.text


def test_preflight_skips_ollama(emb, monkeypatch):
    """ollama는 청커보다 늦게 뜰 수 있다 — 사전 확인으로 사이클을 막지 않는다(기존 동작)."""
    monkeypatch.setattr(emb.requests, "post", lambda *a, **k: pytest.fail("요청하면 안 된다"))
    emb.preflight()


def test_ingest_exits_before_any_document_on_config_error(voy, monkeypatch):
    import argparse

    import ingest_many
    monkeypatch.setattr(voy.requests, "post", lambda *a, **k: _Resp(401, text="bad key"))
    with pytest.raises(SystemExit):
        ingest_many._check_embed(argparse.Namespace(no_embed=False))
    ingest_many._check_embed(argparse.Namespace(no_embed=True))  # 임베딩을 안 하면 보지 않는다
