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
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_PAGES="2", VLM_TARGET="tables")
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
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="2", VLM_TARGET="tables")
    calls = _always_fail(monkeypatch)
    sources = extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert len(sources["pymupdf"]) == 6, "표 6페이지가 잡혀야 의미 있는 검증이다"
    assert calls["n"] == 2, f"차단 후에도 호출이 나갔다 ({calls['n']}회)"
    assert "vlm" not in sources


def test_document_loop_runs_every_page_when_healthy(ex, monkeypatch, table_pdf):
    """차단기가 멀쩡한 문서를 일찍 끊으면 안 된다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_FAIL_STREAK="2", VLM_TARGET="tables")
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
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_PAGES="2", VLM_TARGET="tables")
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
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_PAGES="2", VLM_TARGET="tables")
    _capture(monkeypatch)
    sources = extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert len(sources["pymupdf"]) == 6, "VLM 상한이 다른 소스까지 끊으면 안 된다"


def test_budget_resets_between_documents(ex, monkeypatch, table_pdf):
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VISION_MAX_PAGES="2", VLM_TARGET="tables")
    _capture(monkeypatch)
    extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert extractor.vision_budget_exhausted()
    extractor.extract_tables_for_doc(table_pdf, list(range(1, 7)), use_vision=True)
    assert extractor._vision_call_count == 2, "문서마다 예산이 새로 주어져야 한다"


# ── VLM 투입 대상 선정 ────────────────────────────────────────────────────────
# 예전 기본값은 "PyMuPDF가 표를 찾은 페이지"였다. 그게 거꾸로였다 — VLM은 이미 잘 되던
# 추출을 덮어쓸 기회만 가졌고(실측에서 ICD-10 코드표가 그렇게 망가졌다), 괘선 없는 표와
# 이미지로 박힌 표는 어느 소스도 보지 않았다.

@pytest.fixture
def mixed_pdf(tmp_path):
    """세 종류가 섞인 PDF.

    p1 괘선 표         → PyMuPDF가 잡는다
    p2 텍스트만         → 표 없음
    p3 이미지만(텍스트 X) → 스캔본 같은 페이지. VLM만이 닿는다
    p4 괘선 30개 + 텍스트 → 표 신호는 있는데 PyMuPDF가 표로 인식하진 않는다
    """
    import pymupdf
    doc = pymupdf.open()

    p = doc.new_page()
    for r in range(4):
        for c in range(3):
            rect = pymupdf.Rect(60 + c * 140, 80 + r * 28, 60 + (c + 1) * 140, 80 + (r + 1) * 28)
            p.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p.insert_text((rect.x0 + 5, rect.y0 + 18), f"r{r}c{c}", fontsize=9)

    p = doc.new_page()
    p.insert_text((70, 100), "본문만 있는 페이지입니다.", fontsize=11)

    img = doc.new_page()
    pix = doc[0].get_pixmap(dpi=72)
    img.insert_image(img.rect, stream=pix.tobytes("png"))

    p = doc.new_page()
    p.insert_text((70, 90), "괘선은 많지만 표로는 안 잡히는 페이지", fontsize=11)
    for i in range(30):
        p.draw_line(pymupdf.Point(60, 110 + i * 20), pymupdf.Point(520, 110 + i * 20))

    path = tmp_path / "mixed.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def _targets(extractor, pdf):
    import pymupdf
    doc = pymupdf.open(pdf)
    pages = list(range(1, len(doc) + 1))
    pm = {}
    for pno in pages:
        md = extractor.extract_pymupdf(doc[pno - 1])
        if md:
            pm[pno] = md
    gaps = extractor.gap_pages(doc, pages, pm)
    out = extractor.vlm_target_pages(sorted(pm), gaps)
    doc.close()
    return sorted(pm), gaps, out


def test_default_target_is_the_gap_not_the_tables(ex, mixed_pdf):
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    assert extractor.VLM_TARGET == "gap"
    pm, gaps, targets = _targets(extractor, mixed_pdf)
    assert 1 in pm, "괘선 표 페이지는 PyMuPDF가 잡아야 한다"
    assert 1 not in targets, "PyMuPDF가 성공한 페이지에 VLM을 넣지 않는다"
    assert set(targets) == set(gaps)


def test_pages_without_a_text_layer_come_first(ex, mixed_pdf):
    """텍스트 레이어가 없으면 텍스트 기반 소스는 못 한다 — 예산을 여기부터 쓴다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    _, gaps, targets = _targets(extractor, mixed_pdf)
    assert 3 in gaps, "이미지만 있는 페이지는 gap이어야 한다"
    assert targets[0] == 3, "텍스트 레이어 없는 페이지가 1순위다"


def test_plain_text_pages_are_not_targeted(ex, mixed_pdf):
    """표 신호가 없는 본문 페이지까지 넣으면 호출만 늘고 얻는 게 없다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    _, gaps, _ = _targets(extractor, mixed_pdf)
    assert 2 not in gaps


def test_tables_mode_restores_the_old_behaviour(ex, mixed_pdf):
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="tables")
    pm, _, targets = _targets(extractor, mixed_pdf)
    assert targets == pm


def test_both_mode_spends_the_gap_first(ex, mixed_pdf):
    """예산(VISION_MAX_PAGES)이 순서대로 깎이므로 순서가 곧 우선순위다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="both")
    pm, gaps, targets = _targets(extractor, mixed_pdf)
    assert targets[:len(gaps)] == gaps
    assert set(targets) == set(gaps) | set(pm)


def test_a_mere_logo_is_not_a_table_signal(ex, tmp_path):
    """이미지 '있음'을 신호로 쓰면 전부 통과한다.

    운영 약관 한 건(279쪽)에서 비표 페이지 181쪽이 전부 gap으로 분류됐다. 그 문서는
    모든 페이지에 페이지를 덮는 이미지가 있는 OCR 스캔본이었고, 괘선 수 중앙값은 0이었다.
    표지 로고 한 장만 있어도 같은 일이 난다.
    """
    import pymupdf
    doc = pymupdf.open()
    logo = pymupdf.open()
    lp = logo.new_page(width=40, height=40)
    lp.draw_rect(pymupdf.Rect(0, 0, 40, 40), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    stamp = logo[0].get_pixmap(dpi=72).tobytes("png")
    logo.close()

    for _ in range(3):
        p = doc.new_page()
        p.insert_image(pymupdf.Rect(20, 20, 60, 60), stream=stamp)   # 구석의 작은 로고
        p.insert_text((70, 200), "본문이 충분히 있는 평범한 페이지입니다. " * 12, fontsize=10)
    path = tmp_path / "logo.pdf"
    doc.save(path)
    doc.close()

    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    d = pymupdf.open(path)
    gaps = extractor.gap_pages(d, [1, 2, 3], {})
    d.close()
    assert gaps == [], f"로고만 있는 본문 페이지가 gap으로 들어왔다: {gaps}"


def test_page_covering_image_with_sparse_text_is_first(ex, tmp_path):
    """OCR이 비어 있는 스캔 페이지 — 텍스트 레이어가 내용을 안 담고 있다. VLM만이 닿는다."""
    import pymupdf
    src = pymupdf.open()
    sp = src.new_page()
    sp.insert_text((70, 100), "스캔된 내용", fontsize=20)
    full = src[0].get_pixmap(dpi=72).tobytes("png")
    src.close()

    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_image(p.rect, stream=full)          # 페이지 전체를 덮는 이미지
    p.insert_text((70, 60), "표지", fontsize=9)   # 텍스트는 희박
    path = tmp_path / "scanlike.pdf"
    doc.save(path)
    doc.close()

    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    d = pymupdf.open(path)
    gaps = extractor.gap_pages(d, [1], {})
    d.close()
    assert gaps == [1]


# ── PaddleOCR-VL 출력 변환 ────────────────────────────────────────────────────
# Colab 실험(2026-10-20)에서 확인한 실패 모양을 고정한다 — 좌우 두 표가 한 표로 뭉침,
# colspan을 값 반복으로 평탄화해 안내 문단이 열 수만큼 복제됨, 백지에서 같은 토큰 무한 반복.


def test_each_html_table_becomes_its_own_markdown_table(ex):
    """좌우로 나란한 두 표는 두 표여야 한다 — 모든 <tr>을 한 그리드로 합치면 행이 어긋난다."""
    extractor = ex()
    html = ("<table><tr><td>갑</td><td>1</td></tr></table>"
            "<table><tr><td>을</td><td>2</td></tr></table>")
    md = extractor._html_table_to_markdown(html)
    first, second = md.split("\n\n")
    assert "갑" in first and "을" not in first and "을" in second


def test_colspan_fills_only_the_first_cell(ex):
    """가로 병합을 값 반복으로 채우면 전폭 안내 문단이 열 수만큼 복제된다(현대해상 쪽 실측)."""
    extractor = ex()
    html = ('<table><tr><td>A</td><td>B</td><td>C</td><td>D</td></tr>'
            '<tr><td colspan="4">안내 문단</td></tr></table>')
    md = extractor._html_table_to_markdown(html)
    assert md.count("안내 문단") == 1


def test_rowspan_still_repeats_down(ex):
    """세로 병합은 지금처럼 반복한다 — 표 row 청크가 행 단위로 쪼개져도 값을 잃지 않아야 한다."""
    extractor = ex()
    html = ('<table><tr><td rowspan="2">구분</td><td>1</td></tr>'
            '<tr><td>2</td></tr></table>')
    md = extractor._html_table_to_markdown(html)
    rows = [ln for ln in md.splitlines() if "---" not in ln]
    assert rows[0].startswith("| 구분") and rows[1].startswith("| 구분")


def test_otsl_lcel_is_an_empty_cell(ex):
    """OTSL의 가로 병합(lcel)도 HTML colspan과 같은 규칙이어야 소스 간 표 모양이 같다."""
    extractor = ex()
    md = extractor._otsl_to_markdown("<fcel>안내 문단<lcel><nl><fcel>a<fcel>b<nl>")
    assert md.splitlines()[0].count("안내 문단") == 1


def test_br_inside_a_cell_is_preserved(ex):
    """PyMuPDF 표와 같은 형식(<br>) — 공백으로 뭉개면 목록 셀이 한 문장처럼 붙는다."""
    extractor = ex()
    md = extractor._html_table_to_markdown(
        "<table><tr><td>가<br/>나</td><td><b>강조</b></td></tr></table>")
    assert "가<br>나" in md
    assert "<b>" not in md, "<br> 말고 다른 태그는 지금처럼 걷어내야 한다"


def test_superscript_latex_is_normalized(ex, monkeypatch):
    """Paddle이 위첨자를 LaTeX 조각으로 낸다 — 텍스트 레이어와 대조하려면 같은 기호여야 한다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    _capture(monkeypatch, "| 보장\\(^{*}\\) | 한도$ ^{*} $ |\n|---|---|\n| 1 | 2 |")
    out = extractor.extract_vision_local(_FakePage(), 1)
    assert "보장*" in out and "한도*" in out and "^{" not in out


# ── 퇴행 출력 폐기 ────────────────────────────────────────────────────────────
# 운영 기준선에서 호출의 28%가 max_tokens에서 잘렸고(이슈 #72), 그 대부분이 백지 입력의
# 반복 루프였다. 잘린 표는 행이 끊겨 구조가 틀리다 — 반쪽을 싣는 대신 버린다.


class _FinishResp(_Resp):
    def __init__(self, content, finish="stop"):
        super().__init__(content)
        self._finish = finish

    def json(self):
        body = super().json()
        body["choices"][0]["finish_reason"] = self._finish
        return body


def test_truncated_table_is_discarded(ex, monkeypatch):
    import requests
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    resp = _FinishResp("| a |\n|---|\n| 1 |", finish="length")
    monkeypatch.setattr(requests, "post", lambda *a, **k: resp)
    assert extractor.extract_vision_local(_FakePage(), 1) is None
    assert extractor.vision_page_stats()[-1]["outcome"] == "truncated"


def test_looping_cells_are_discarded(ex, monkeypatch):
    """max_tokens 전에 멈춘 루프는 finish_reason으로 못 잡는다 — 셀 반복이 지문이다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_LOOP_CELLS="5")
    _capture(monkeypatch, "| " + " | ".join(["1"] * 6) + " |\n|" + "---|" * 6)
    assert extractor.extract_vision_local(_FakePage(), 1) is None
    assert extractor.vision_page_stats()[-1]["outcome"] == "loop"


def test_a_normal_table_with_repeats_is_kept(ex, monkeypatch):
    """세로 병합 반복·○ 표시 열은 행마다 다른 셀이 끼므로 루프로 오인하면 안 된다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_LOOP_CELLS="5")
    rows = "\n".join(f"| 담보{i} | ○ | ○ |" for i in range(10))
    _capture(monkeypatch, "| 담보 | 가입 | 갱신 |\n|---|---|---|\n" + rows)
    assert extractor.extract_vision_local(_FakePage(), 1) is not None


# ── 모델별 기본 프롬프트 ──────────────────────────────────────────────────────


def test_paddle_model_defaults_to_the_task_prompt(ex, monkeypatch):
    """PaddleOCR-VL은 태스크 토큰이 곧 프롬프트다 — 모델만 바꾸면 프롬프트도 따라와야
    배포에서 VLM_MODEL 하나만 바꿔도 동작한다."""
    monkeypatch.delenv("VLM_PROMPT", raising=False)
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B")
    assert extractor.VLM_PROMPT == "Table Recognition:"


def test_explicit_prompt_beats_the_model_default(ex, monkeypatch):
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="PaddleOCR-VL-1.6-0.9B",
                   VLM_PROMPT="커스텀")
    assert extractor.VLM_PROMPT == "커스텀"


def test_default_prompt_merge_rule_matches_the_converter(ex, monkeypatch):
    """qwen 프롬프트의 병합 규칙이 변환기와 다르면 소스마다 표 모양이 달라진다."""
    monkeypatch.delenv("VLM_PROMPT", raising=False)
    monkeypatch.delenv("VLM_MODEL", raising=False)
    extractor = ex()
    assert "세로로 병합" in extractor.VLM_PROMPT
    assert "첫 칸" in extractor.VLM_PROMPT


def test_max_tokens_comes_from_the_environment(ex, monkeypatch):
    """출력 상한이 최악 페이지의 시간이다 — 재빌드 없이 바꿔 잴 수 있어야 한다."""
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_MAX_TOKENS="2048")
    seen = _capture(monkeypatch)
    extractor.extract_vision_local(_FakePage(), 1)
    assert seen["payload"]["max_tokens"] == 2048


def test_max_tokens_default_is_unchanged(ex, monkeypatch):
    monkeypatch.delenv("VLM_MAX_TOKENS", raising=False)
    extractor = ex(VLM_BACKEND="local", VLM_MODEL="m")
    seen = _capture(monkeypatch)
    extractor.extract_vision_local(_FakePage(), 1)
    assert seen["payload"]["max_tokens"] == 4096
