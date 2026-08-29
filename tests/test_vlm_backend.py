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
