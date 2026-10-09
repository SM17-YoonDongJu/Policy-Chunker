"""VLM 표 추출 백엔드 계약.

호스트의 Ollama에 qwen3-vl:8b-instruct가 받아져 있는데 파이프라인이 한 번도 부르지 않고
있었다(로그에 "surya_ocr 미설치" 273회). 원인은 셋이었고 그중 코드 문제는 하나다 —
페이로드에 model이 없어 Ollama가 받지 못했다. llama-server는 모델 하나만 서빙해서
생략해도 됐지만 Ollama는 필수다.

두 서버를 같은 코드로 상대하므로, 그 분기가 어긋나지 않게 여기서 고정한다.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakePix:
    def tobytes(self, fmt):
        return b"\x89PNG-fake"


class _FakePage:
    def get_pixmap(self, dpi=None):
        return _FakePix()


class _Resp:
    def __init__(self, content="", status=200):
        self._content, self.status_code = content, status

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def ex(monkeypatch):
    def _load(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from insurance_chunker import extractor
        importlib.reload(extractor)
        return extractor
    return _load


def _capture(monkeypatch, content="| a | b |\n|---|---|\n| 1 | 2 |"):
    """요청 페이로드를 가로채고 가짜 응답을 돌려준다.

    extract_vision_local이 함수 안에서 `import requests`를 하므로 모듈 속성이 아니다 —
    전역 requests.post를 갈아끼운다.
    """
    import requests
    seen = {}

    def _post(url, json=None, timeout=None):
        seen["url"], seen["payload"], seen["timeout"] = url, json, timeout
        return _Resp(content)

    monkeypatch.setattr(requests, "post", _post)
    return seen


# ── Ollama 호환 ───────────────────────────────────────────────────────────────

def test_model_is_sent_when_configured(ex, monkeypatch):
    """Ollama는 model이 없으면 400을 낸다 — 이게 빠져서 여태 안 붙었다."""
    extractor = ex(VLM_BACKEND="local", VLM_URL="http://localhost:11434",
                   VLM_MODEL="qwen3-vl:8b-instruct")
    seen = _capture(monkeypatch)
    extractor.extract_vision_local(_FakePage(), 1)
    assert seen["payload"]["model"] == "qwen3-vl:8b-instruct"
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"


def test_model_is_omitted_when_unset(ex, monkeypatch):
    """llama-server는 모델 하나만 서빙해 model을 받지 않는다 — 하위 호환 유지."""
    extractor = ex(VLM_BACKEND="local", VLM_URL="http://localhost:8090", VLM_MODEL="")
    seen = _capture(monkeypatch)
    extractor.extract_vision_local(_FakePage(), 1)
    assert "model" not in seen["payload"]


def test_image_is_sent_as_a_data_uri(ex, monkeypatch):
    """OpenAI 호환 규격 — Ollama도 llama-server도 이 형태로 이미지를 받는다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    seen = _capture(monkeypatch)
    extractor.extract_vision_local(_FakePage(), 1)
    content = seen["payload"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "text"


# ── 프롬프트 ──────────────────────────────────────────────────────────────────

def test_default_prompt_targets_a_general_vlm(ex, monkeypatch):
    """PaddleOCR-VL의 'Table Recognition:'은 태스크 토큰이라 qwen3-vl에는 안 먹는다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="qwen3-vl:8b-instruct")
    seen = _capture(monkeypatch)
    extractor.extract_vision_local(_FakePage(), 1)
    prompt = seen["payload"]["messages"][0]["content"][1]["text"]
    assert "마크다운" in prompt
    assert "표가 전혀 없으면" in prompt
    # 약관 표는 숫자·한자가 많아 의역이 곧 오답이다.
    assert "의역" in prompt


def test_prompt_is_overridable(ex, monkeypatch):
    """PaddleOCR-VL로 되돌릴 수 있어야 한다."""
    extractor = ex(VLM_BACKEND="local", VLM_PROMPT="Table Recognition:")
    seen = _capture(monkeypatch)
    extractor.extract_vision_local(_FakePage(), 1)
    assert seen["payload"]["messages"][0]["content"][1]["text"] == "Table Recognition:"


# ── 출력 처리 ─────────────────────────────────────────────────────────────────

def test_markdown_table_passes_through(ex, monkeypatch):
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    _capture(monkeypatch, "| 담보 | 금액 |\n|---|---|\n| 암 | 1000 |")
    out = extractor.extract_vision_local(_FakePage(), 1)
    assert "담보" in out


def test_no_table_returns_none(ex, monkeypatch):
    """표가 없는 페이지에 설명문을 받아 표로 착각하면 안 된다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    _capture(monkeypatch, "이 페이지에는 표가 없습니다.")
    assert extractor.extract_vision_local(_FakePage(), 1) is None


def test_empty_response_returns_none(ex, monkeypatch):
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    _capture(monkeypatch, "")
    assert extractor.extract_vision_local(_FakePage(), 1) is None


def test_code_fence_is_stripped(ex, monkeypatch):
    """모델이 지시를 어기고 펜스를 붙이는 경우가 흔하다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    _capture(monkeypatch, "```markdown\n| a |\n|---|\n| 1 |\n```")
    out = extractor.extract_vision_local(_FakePage(), 1)
    assert out is not None and "```" not in out


def test_server_error_returns_none_not_raise(ex, monkeypatch):
    """VLM 실패가 문서 전체를 죽이면 안 된다 — 표 없이라도 적재는 되어야 한다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")

    import requests

    def _boom(*a, **k):
        raise ConnectionError("refused")

    monkeypatch.setattr(requests, "post", _boom)
    assert extractor.extract_vision_local(_FakePage(), 1) is None


def test_page_budget_is_respected(ex, monkeypatch):
    """VISION_MAX_PAGES는 비용·시간 상한이다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_PAGES="2")
    _capture(monkeypatch)
    for _ in range(3):
        extractor.extract_vision(_FakePage(), 1)
    assert extractor._vision_call_count == 2


# ── 기본값 · off 스위치 ───────────────────────────────────────────────────────

def test_defaults_point_at_the_host_ollama(ex, monkeypatch):
    """claude CLI를 걷어내고 같은 호스트 Ollama의 qwen3-vl로 대체했다.

    기본값이 어긋나면 아무도 설정을 안 건드린 채 VLM이 또 조용히 죽는다
    (이전에 surya가 기본이라 273회 건너뛴 그대로).
    """
    for k in ("VLM_BACKEND", "VLM_URL", "VLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    extractor = ex()
    assert extractor.VLM_BACKEND == "local"
    assert extractor.VLM_URL == "http://localhost:11434"
    assert extractor.VLM_MODEL == "qwen3-vl:8b-instruct"


def test_off_backend_skips_without_calling(ex, monkeypatch):
    """VLM을 끄는 명시적 스위치. 예전엔 '설치 안 된 백엔드'가 사실상 off 역할을 했다."""
    extractor = ex(VLM_BACKEND="off")
    seen = _capture(monkeypatch)
    assert extractor.extract_vision(_FakePage(), 1) is None
    assert seen == {}


def test_off_backend_does_not_consume_the_page_budget(ex, monkeypatch):
    extractor = ex(VLM_BACKEND="off", VISION_MAX_PAGES="2")
    _capture(monkeypatch)
    for _ in range(5):
        extractor.extract_vision(_FakePage(), 1)
    assert extractor._vision_call_count == 0


def test_claude_backend_is_gone(ex, monkeypatch):
    """유료 API 경로를 제거했다. 알 수 없는 값이 와도 local로 처리한다."""
    extractor = ex(VLM_BACKEND="claude", VLM_MODEL="m")
    assert not hasattr(extractor, "CLAUDE_BIN")
    seen = _capture(monkeypatch)
    extractor.extract_vision(_FakePage(), 1)
    assert seen["url"].endswith("/v1/chat/completions")


# ── 차단기 · 타임아웃 (2층: 떨어지더라도 피해를 제한) ──────────────────────────
# 표 214페이지짜리 문서가 페이지마다 600초를 기다리며 35.6시간을 태운 적이 있다.
# 서버가 죽었으면 215번째 시도도 죽는다 — 연속 실패를 세서 문서 단위로 접는다.

def _always_fail(monkeypatch, exc=None):
    import requests
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise (exc or ConnectionError("refused"))

    monkeypatch.setattr(requests, "post", _boom)
    return calls


def test_connect_and_read_timeouts_are_separate(ex, monkeypatch):
    """접속은 즉시 돼야 하고(같은 호스트), 읽기만 오래 기다린다.

    하나로 묶어두면 Ollama가 아예 없을 때도 읽기 타임아웃만큼 기다린다.
    """
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    seen = _capture(monkeypatch)
    extractor.extract_vision_local(_FakePage(), 1)
    connect, read = seen["timeout"]
    assert connect < read
    assert connect == extractor.VLM_CONNECT_TIMEOUT
    assert read == extractor.VLM_TIMEOUT


def test_circuit_opens_after_consecutive_failures(ex, monkeypatch):
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="3")
    calls = _always_fail(monkeypatch)
    for pno in range(1, 11):
        assert extractor.extract_vision(_FakePage(), pno) is None
    assert calls["n"] == 3, "차단 후에는 호출 자체가 나가면 안 된다"
    assert extractor.vision_circuit_open()


def test_no_table_pages_do_not_open_the_circuit(ex, monkeypatch):
    """표가 없는 페이지도 None이다. 이걸 실패로 세면 멀쩡한 문서에서 VLM이 꺼진다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="3")
    _capture(monkeypatch, "이 페이지에는 표가 없습니다.")
    for pno in range(1, 11):
        extractor.extract_vision(_FakePage(), pno)
    assert not extractor.vision_circuit_open()
    assert extractor._vision_call_count == 10


def test_success_resets_the_streak(ex, monkeypatch):
    """간헐 실패로는 접지 않는다 — 성공 한 번이면 서버는 살아 있는 것이다."""
    import requests
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="3")
    script = ["fail", "fail", "ok", "fail", "fail"]
    state = {"i": 0}

    def _post(*a, **k):
        step = script[state["i"]]
        state["i"] += 1
        if step == "fail":
            raise ConnectionError("refused")
        return _Resp("| a |\n|---|\n| 1 |")

    monkeypatch.setattr(requests, "post", _post)
    for pno in range(1, 6):
        extractor.extract_vision(_FakePage(), pno)
    assert not extractor.vision_circuit_open()


def test_circuit_resets_between_documents(ex, monkeypatch):
    """서버가 잠깐 죽었다 살아난 경우 다음 문서까지 포기할 이유가 없다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="2")
    _always_fail(monkeypatch)
    for pno in range(1, 5):
        extractor.extract_vision(_FakePage(), pno)
    assert extractor.vision_circuit_open()

    extractor.reset_vision_counter()
    assert not extractor.vision_circuit_open()
    assert extractor._vision_call_count == 0


def test_circuit_open_is_logged_loudly(ex, monkeypatch, caplog):
    """조용히 접으면 '표가 원래 없는 문서'와 구분이 안 된다."""
    import logging
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="2")
    _always_fail(monkeypatch)
    with caplog.at_level(logging.ERROR, logger="insurance_chunker.extractor"):
        for pno in range(1, 6):
            extractor.extract_vision(_FakePage(), pno)
    opened = [r for r in caplog.records if getattr(r, "event", "") == "vlm_circuit_open"]
    assert len(opened) == 1, "접는 순간 한 번만 남겨야 한다"


def test_failure_is_still_swallowed_by_the_public_entry(ex, monkeypatch):
    """차단기가 예외를 쓰더라도 바깥으로 새면 안 된다 — 문서 적재는 계속돼야 한다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    _always_fail(monkeypatch, RuntimeError("HTTP 500"))
    assert extractor.extract_vision(_FakePage(), 1) is None


# ── 문서 루프까지 (차단이 실제로 호출을 멈추는가) ─────────────────────────────

@pytest.fixture
def table_pdf(tmp_path):
    """괘선 표가 있는 6페이지 PDF. PyMuPDF가 6페이지 모두 표로 잡는다."""
    import pymupdf
    doc = pymupdf.open()
    for p in range(6):
        page = doc.new_page()
        for r in range(4):
            for c in range(3):
                rect = pymupdf.Rect(60 + c * 140, 80 + r * 28,
                                    60 + (c + 1) * 140, 80 + (r + 1) * 28)
                page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                page.insert_text((rect.x0 + 5, rect.y0 + 18), f"p{p+1}r{r}c{c}", fontsize=9)
    path = tmp_path / "tbl.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def test_document_loop_stops_calling_after_the_circuit_opens(ex, monkeypatch, table_pdf):
    """차단기의 값어치는 여기서 나온다.

    없을 때 214페이지짜리 문서가 페이지마다 600초를 기다려 35.6시간을 태웠다.
    6페이지 문서에서 VLM_FAIL_STREAK=2면 호출은 2번에서 멈춰야 한다.
    """
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="2")
    calls = _always_fail(monkeypatch)
    sources = extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert len(sources["pymupdf"]) == 6, "표 6페이지가 잡혀야 의미 있는 검증이다"
    assert calls["n"] == 2, f"차단 후에도 호출이 나갔다 ({calls['n']}회)"
    assert "vlm" not in sources


def test_document_loop_runs_every_page_when_healthy(ex, monkeypatch, table_pdf):
    """차단기가 멀쩡한 문서를 일찍 끊으면 안 된다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="2")
    seen = _capture(monkeypatch, "| a | b |\n|---|---|\n| 1 | 2 |")
    sources = extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert len(sources["vlm"]) == 6
    assert seen["payload"]["model"] == "m"


# ── 문서당 호출 상한 (처리 시간 꼬리 방어) ────────────────────────────────────

def test_budget_default_is_a_real_limit(ex, monkeypatch):
    """9999는 '상한 없음'이었다. 그게 parse p90을 p50의 22배로 만든 꼬리의 원인이다."""
    monkeypatch.delenv("VISION_MAX_PAGES", raising=False)
    extractor = ex()
    assert extractor.VISION_MAX_PAGES == 50


def test_budget_stops_the_document_loop(ex, monkeypatch, table_pdf):
    """상한에 닿으면 남은 페이지는 호출하지 않는다 — 6페이지 문서에 상한 2면 2회."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_PAGES="2")
    seen = _capture(monkeypatch)
    calls = {"n": 0}
    import requests
    orig = requests.post

    def _counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(requests, "post", _counting)
    sources = extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert calls["n"] == 2
    assert len(sources["vlm"]) == 2
    assert seen["url"].endswith("/v1/chat/completions")


def test_budget_does_not_drop_the_other_sources(ex, monkeypatch, table_pdf):
    """상한은 VLM만 끊는다. 남은 페이지도 PyMuPDF 표는 그대로 있어야 한다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_PAGES="2")
    _capture(monkeypatch)
    sources = extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert len(sources["pymupdf"]) == 6, "VLM 상한이 다른 소스까지 끊으면 안 된다"


def test_budget_resets_between_documents(ex, monkeypatch, table_pdf):
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_PAGES="2")
    _capture(monkeypatch)
    extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert extractor.vision_budget_exhausted()
    extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert extractor._vision_call_count == 2, "문서마다 예산이 새로 주어져야 한다"
