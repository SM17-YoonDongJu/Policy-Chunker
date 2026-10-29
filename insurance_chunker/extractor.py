"""표 추출: PyMuPDF(폴백) + VLM(비전).

  - PyMuPDF : fitz.find_tables() (괘선 있는 표, 빠름). VLM이 못 채운 쪽의 폴백
  - VLM     : OpenAI 호환 서버(운영은 vLLM의 PaddleOCR-VL, 로컬은 Ollama).
              레이아웃(layout.py)이 잡은 표 영역을 crop해 넣는 것이 기본 경로이고,
              LAYOUT_BACKEND=off면 예전처럼 gap 휴리스틱 + 페이지 전체 입력이다

pdfplumber·camelot 경로는 제거했다 — Paddle crop이 역할을 대신하고, pdfplumber의
쪽 글자 캐시(OOM 111회 재시작의 원인, #71)가 벡터째 사라진다.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# 문서당 VLM 호출 상한 — 페이지 전체 입력(LAYOUT_BACKEND=off) 경로의 예산.
# 9999는 사실상 상한이 없다는 뜻이었고, 그게 처리 시간의 꼬리를 만들었다 — 실측에서
# 문서 20건(6.4%)이 전체 parse 시간의 53.4%를 먹었고, parse p50은 131초인데 p90이
# 2,879초로 22배다. 50 x 25초 = 약 20분이 문서당 VLM 상한이 된다.
# 상한을 넘는 페이지는 VLM을 안 거칠 뿐 표가 사라지지는 않는다 — PyMuPDF 결과가 남는다.
VISION_MAX_PAGES = int(os.environ.get("VISION_MAX_PAGES", "50"))
# 레이아웃 crop 경로의 예산 — 상한의 단위는 쪽이 아니라 박스다. 674쪽 문서에 표 208쪽이면
# 박스 300개 안팎이고, 영역당 1~3초라 300개여도 10분 안쪽이다.
VISION_MAX_CROPS = int(os.environ.get("VISION_MAX_CROPS", "300"))
# crop 렌더 해상도. 200이 기본인 이유: 모델이 쪽 전체를 약 1MP로 줄여 받아 페이지 전체
# 입력에서는 150↔200이 무의미했지만(입력 토큰 1,273으로 동일), 영역 입력은 원본 해상도가
# 거의 그대로 들어가 200 DPI의 글자가 실제로 더 선명하다(Colab 실험 C).
VLM_DPI = int(os.environ.get("VLM_DPI", "200"))
# 읽기 타임아웃. 영역 하나는 1~3초(Colab 실측)라 60초를 넘기면 느린 게 아니라 고장이다.
# LAYOUT_BACKEND=off(페이지 전체 + qwen3-vl)로 되돌릴 때는 예전 값 300을 env로 줘야 한다 —
# 그 경로는 페이지당 p50 88.6초였다.
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", "60"))
# 페이지당 출력 토큰 상한. 생성이 VLM 시간의 대부분이라(vLLM 단일 요청 18.7토큰/초) 상한이 곧
# 최악 페이지의 시간이다. smoke 벤치마크(2026-10-14)에서 50건 중 7건이 4096까지 생성했고 그 7건이
# VLM 시간의 약 53%였다 — 상한까지 가는 응답은 대개 반복 폭주라 끝까지 받아도 쓸 수 없다. 줄이면
# 결과가 바뀌는 변경이라 벤치마크 가드레일로 판정한 뒤 바꾼다. 프롬프트(비전 토큰 포함, p50 약
# 2,300)와 합쳐 vLLM max-model-len(운영 8,192)을 넘지 않게 둔다.
VLM_MAX_TOKENS = int(os.environ.get("VLM_MAX_TOKENS", "4096"))
# 접속 타임아웃은 따로 짧게 둔다. 같은 호스트의 Ollama라 붙는 건 즉시여야 하고,
# 서버가 아예 없으면 밀리초 안에 알아야지 읽기 타임아웃까지 기다릴 이유가 없다.
VLM_CONNECT_TIMEOUT = float(os.environ.get("VLM_CONNECT_TIMEOUT", "5"))
# 차단기: 연속 N회 실패하면 이 문서의 남은 페이지는 VLM을 건너뛴다.
# 없을 때 무슨 일이 생기는지 겪었다 — 표 214페이지짜리 문서가 페이지마다 600초를
# 기다리며 35.6시간을 태웠다. 서버가 죽었으면 215번째 시도도 죽는다.
VLM_FAIL_STREAK = int(os.environ.get("VLM_FAIL_STREAK", "3"))
# 문서 하나 안에서 VLM을 동시에 몇 건까지 부를지. 기본 1 = 예전 순차 동작.
# 페이지끼리 의존이 없고 서버(vLLM)는 continuous batching으로 여러 요청을 한 번에 디코딩한다.
# 운영 VLM은 생성 속도가 병목이라(18.7토큰/초, 페이지당 p50 88.6초) 동시 요청이 곧 처리량이다.
# 상한은 서버의 KV 캐시 여유다 — 운영 설정에서 6,144토큰 요청 기준 3.17배. 넘기면 서버가
# 대기시키고 그 대기도 VLM_TIMEOUT에 들어가 타임아웃만 는다. 문서 병렬(INGEST_CONCURRENCY)과
# 곱해진다는 점도 주의.
VLM_CONCURRENCY = int(os.environ.get("VLM_CONCURRENCY", "1"))
# VLM을 어느 페이지에 투입할 것인가.
#
#   gap    (기본) PyMuPDF가 닿지 못한 페이지 — 텍스트 레이어가 없거나 표를 못 찾은 곳
#   tables        PyMuPDF가 표를 찾은 페이지 (예전 동작)
#   both          둘 다. gap을 먼저 소진한다
#
# 예전 기본값은 tables였는데 그게 거꾸로였다. VLM은 PyMuPDF가 이미 성공한 페이지에만
# 들어가서, 잘 되던 추출을 덮어쓸 기회만 가졌다. 실측에서 ICD-10 코드표 한 장이
# 그렇게 망가졌다 — 150DPI에서 대문자 I와 숫자 1이 구분되지 않아 I00~I02가 100~102로
# 들어왔고, 그 페이지의 텍스트 레이어는 멀쩡해서 PyMuPDF는 100% 맞히고 있었다.
# 반대로 괘선 없는 표와 이미지로 박힌 표는 어느 소스도 보지 않고 있었다(실측 문서에서 70쪽).
VLM_TARGET = os.environ.get("VLM_TARGET", "gap")
# gap 페이지 판정 기준.
#
# 처음엔 "괘선 4개 이상 또는 이미지가 있으면 표 신호"로 뒀는데 그게 거의 전부를 통과시켰다.
# 운영 약관 한 건(279쪽)에서 비표 페이지 181쪽이 전부 gap으로 분류됐다 — 원인은 이미지
# 조건이었다. 그 문서는 모든 페이지에 페이지 전체를 덮는 이미지가 있는 OCR 스캔본이라
# get_images()가 항상 참이었고, 정작 괘선 수 중앙값은 0이었다. 표지 로고 한 장만 있어도
# 같은 일이 난다. 이미지는 '있다/없다'가 아니라 '페이지를 덮는가'로 봐야 한다.
#
# 수치는 그 문서의 분포에서 왔다(괘선 p90=25). 문서 한 건으로 정한 값이라 환경변수로
# 열어둔다 — 표본이 넓어지면 기본값을 다시 잡아야 한다.
_HINT_DRAWINGS = int(os.environ.get("VLM_HINT_DRAWINGS", "25"))
_HINT_MIN_TEXT = int(os.environ.get("VLM_HINT_MIN_TEXT", "200"))
_HINT_IMG_FRAC = float(os.environ.get("VLM_HINT_IMG_FRAC", "0.25"))
# 'local' = OpenAI 호환 /v1/chat/completions — 같은 호스트의 Ollama(기본) 또는 llama-server
# 'surya' = Surya OCR (별도 바이너리 필요 — [ocr] extra)
# 'off'   = VLM 표 추출 안 함
VLM_BACKEND = os.environ.get("VLM_BACKEND", "local")
VLM_URL = os.environ.get("VLM_URL", "http://localhost:11434")
# local 백엔드가 부르는 /v1/chat/completions는 OpenAI 호환 규격이라 Ollama도 그대로 받는다.
# 다만 Ollama는 model 필드가 필수다(llama-server는 모델 하나만 서빙해 생략 가능).
# 비워두면 페이로드에서 빼므로 llama-server 호환이 유지된다.
#
# 기본값은 운영이 서빙하는 모델이다(vLLM의 --served-model-name과 같아야 한다). 로컬에서 쓰려면
# 그 모델을 띄우고 VLM_URL을 맞춘다 — vLLM이나 llama-server(PaddleOCR-VL GGUF)로 뜬다.
# Ollama에 받아 둔 범용 VLM을 쓰려면 VLM_MODEL에 그 이름을 주면 되고, 프롬프트는 아래에서
# 모델명에 따라 자동으로 갈린다.
VLM_MODEL = os.environ.get("VLM_MODEL", "PaddleOCR-VL-1.6-0.9B")
# PaddleOCR-VL은 "Table Recognition:"이 태스크 토큰이지만, 범용 instruct VLM에는
# 명시적 지시가 필요하다. 규칙은 claude 백엔드를 쓰던 시절 쌓인 것을 옮겼다 — 약관 표는
# 숫자·한자가 많아 의역이 곧 오답이고, 읽기 어려운 셀에서 멈추면 페이지 전체를 잃는다.
# 병합 규칙은 변환기(_html_table_to_markdown)와 같아야 한다 — 세로만 반복, 가로는 첫 칸.
_DEFAULT_VLM_PROMPT = (
    "이 페이지 이미지의 표를 GitHub 마크다운 표로만 옮겨라.\n"
    "규칙(엄수):\n"
    "- 셀 텍스트는 원문 그대로(숫자·한자·줄바꿈 보존). 의역·요약 금지.\n"
    "- 세로로 병합된 셀은 값을 반복해 채워라. 가로로 병합된 셀은 첫 칸에만 쓰고 "
    "나머지는 빈 칸으로 두어라.\n"
    "- 읽기 어려운 셀은 [?]로 표기하고 계속 진행하라.\n"
    "- 표가 여러 개면 빈 줄로 구분하라.\n"
    "- 표가 전혀 없으면 아무것도 출력하지 마라.\n"
    "- 설명·머리말·코드펜스 없이 표 마크다운만 출력하라."
)
_PADDLE_TASK_PROMPT = "Table Recognition:"
# PaddleOCR-VL은 태스크 토큰이 곧 프롬프트다 — 범용 지시문을 주면 태스크 라우팅이 안 된다.
# 모델명으로 판별하므로 VLM_MODEL만 바꿔도 프롬프트를 따로 맞출 필요가 없다.
# VLM_PROMPT를 명시하면 어느 모델이든 그 값이 이긴다.
VLM_PROMPT = os.environ.get("VLM_PROMPT") or (
    _PADDLE_TASK_PROMPT if "paddleocr" in VLM_MODEL.lower() else _DEFAULT_VLM_PROMPT)
# 텍스트 레이어 없는 쪽(스캔)의 본문 복구 프롬프트. 표 프롬프트와 같은 규칙으로 모델명에
# 따라 기본값을 고른다 — Paddle은 "OCR:" 태스크 토큰이 쪽 전체 markdown을 낸다(Colab 실험
# 방법 D에서 확인).
_PADDLE_OCR_TASK_PROMPT = "OCR:"
_DEFAULT_OCR_PROMPT = (
    "이 페이지 이미지의 글을 읽는 순서대로 원문 그대로 출력하라.\n"
    "- 의역·요약·설명·머리말 금지. 본문 텍스트만.\n"
    "- 읽을 수 없는 글자는 [?]로 표기하고 계속 진행하라."
)
VLM_OCR_PROMPT = os.environ.get("VLM_OCR_PROMPT") or (
    _PADDLE_OCR_TASK_PROMPT if "paddleocr" in VLM_MODEL.lower() else _DEFAULT_OCR_PROMPT)
# 1이면 VLM 표 셀을 같은 쪽 텍스트 레이어 줄과 대조해(공백 제거, ratio>=0.8) 일치하는 줄로
# 교정한다. 입력 토큰이 영역당 1,100~1,300에서 막혀 해상도로는 못 고치는 오독(실측:
# 관혈→관협)이 대상이다. 켜고 끈 차이를 벤치의 cell_match(텍스트 레이어 일치율)로 재서
# 기본값을 정한다 — 그때까지 기본 off.
VLM_SNAP_TEXT = os.environ.get("VLM_SNAP_TEXT", "0") == "1"
# markdown 셀에서 같은 값이 이만큼 연속되면 퇴행(반복 루프) 출력으로 보고 폐기한다.
# Colab 실험(2026-10-20)에서 백지 입력이 같은 토큰을 수천 번 반복했다 — max_tokens에
# 닿으면 finish_reason으로 잡히지만, 그 전에 멈춘 루프는 이 검사만이 잡는다.
# 정상 표의 반복(○/X 열, 세로 병합 반복)은 행마다 다른 셀이 끼어 streak이 끊긴다.
VLM_LOOP_CELLS = int(os.environ.get("VLM_LOOP_CELLS", "20"))
LLAMA_CPP_BINARY = os.environ.get(
    "LLAMA_CPP_BINARY",
    os.path.expanduser("~/.local/llama.cpp/llama-b10182/llama-server"),
)

_vision_call_count = 0
_vision_fail_streak = 0
# 문서 안 VLM 호출 한 건마다 {page, seconds, outcome, image_px, prompt_tokens,
# completion_tokens}. 차단기·상한과 같은 수명(문서 단위)이다. 단건 지연이 렌더 해상도(비전
# 토큰)에서 오는지 생성 길이에서 오는지 가르려면 호출마다 이 넷이 함께 있어야 한다.
_vision_page_stats: list[dict] = []
# 이 문서에서 텍스트 레이어로 교정한 셀 수(VLM_SNAP_TEXT). 켜고 끈 효과를 지표에서 본다.
_snap_count = 0
# 이 문서에서 OCR로 복구한 쪽의 본문({page: text}). VlmStage가 만들고 _publish가 옮긴다 —
# 표(extract_vlm_tables 반환값)와 수명은 같지만 쓰임이 달라(본문 채움) 따로 든다.
_ocr_texts: dict[int, str] = {}


class _VLMCallFailed(Exception):
    """VLM 호출 자체가 실패했다 — '표가 없다'(정상 응답)와 구분하기 위한 신호."""


def reset_vision_counter() -> None:
    """문서 경계에서 호출 상한과 차단기를 함께 되돌린다.

    차단기를 문서 단위로 두는 이유: 서버가 잠깐 죽었다 살아난 경우 다음 문서까지
    포기할 이유가 없다. 반대로 한 문서 안에서는 한 번 접었으면 끝까지 접는다.
    """
    global _vision_call_count, _vision_fail_streak, _vision_page_stats, _snap_count, _ocr_texts
    _vision_call_count = 0
    _vision_fail_streak = 0
    _vision_page_stats = []
    _snap_count = 0
    _ocr_texts = {}


def vision_page_stats() -> list[dict]:
    """이 문서에서 나간 VLM 호출 기록(호출 순서). 계측용."""
    return list(_vision_page_stats)


def vision_ocr_texts() -> dict[int, str]:
    """이 문서에서 OCR로 복구한 쪽의 본문. stages.vlm이 단계 출력에 싣는다."""
    return dict(_ocr_texts)


def _vlm_outcome(exc: Exception) -> str:
    """실패를 타임아웃과 그 밖으로 가른다. 타임아웃 비율이 VLM_TIMEOUT 조정의 근거가 된다."""
    import requests
    return "timeout" if isinstance(exc, requests.Timeout) else "error"


def vision_circuit_open() -> bool:
    """차단기가 열렸는가. 문서 루프가 남은 페이지를 건너뛰는 데 쓴다."""
    return _vision_fail_streak >= VLM_FAIL_STREAK


def vision_budget_exhausted() -> bool:
    """문서당 VLM 호출 상한에 닿았는가."""
    return _vision_call_count >= VISION_MAX_PAGES


# ── PyMuPDF ───────────────────────────────────────────────────────────────────

# find_tables가 테두리 안 본문을 표로 오인한다 — 손해보험 약관은 "보상하는 사항" 류를
# 전폭 테두리(1~2열)에 수십 쪽 이어 쓰는 관습이 있어 드물지 않다(DB p236, 현대 p31,
# 메리츠 p64, 삼성 p9에서 실측). 그 쪽은 본문 줄 중심점이 전부 박스 안이라 parse_text에서
# 통째로 빠지고, 거대한 한 셀 markdown만 남는다. 가짜 표의 지문은 셀 수다 — 본문 테두리는
# 셀이 1~2개거나 쪽 대부분을 덮으면서 셀이 거의 없고, 진짜 전면 표(순환계질환 분류표
# 22행 = 44셀)는 셀이 수십 개다. 면적만 보면 진짜 전면 표가 죽는다.
TABLE_BORDER_FRAC = float(os.environ.get("TABLE_BORDER_FRAC", "0.6"))
TABLE_BORDER_MAX_CELLS = int(os.environ.get("TABLE_BORDER_MAX_CELLS", "8"))


def is_border_box(table, page_rect) -> bool:
    """find_tables 결과 하나가 표가 아니라 본문 테두리인가.

    parse_text(본문 제외)와 extract_pymupdf(표 생성)가 같은 판정을 써야 한다 —
    한쪽만 거르면 본문이 빠진 채 표도 없거나, 본문과 한 셀 표가 중복된다.
    """
    cells = [c for c in (table.cells or []) if c is not None]
    if len(cells) <= 2:
        return True
    x0, y0, x1, y1 = table.bbox
    page_area = abs(page_rect.width * page_rect.height) or 1.0
    return (abs((x1 - x0) * (y1 - y0)) >= TABLE_BORDER_FRAC * page_area
            and len(cells) <= TABLE_BORDER_MAX_CELLS)


def extract_pymupdf(fitz_page) -> Optional[str]:
    """fitz page → markdown 표 문자열. 표가 없거나 실패 시 None. 테두리 오탐은 뺀다."""
    try:
        tabs = fitz_page.find_tables()
        real = [t for t in tabs.tables if not is_border_box(t, fitz_page.rect)]
        if not real:
            return None
        parts = [tab.to_markdown() for tab in real if tab.to_markdown().strip()]
        return "\n\n".join(parts) if parts else None
    except Exception as e:
        logger.debug(f"PyMuPDF 표 추출 실패: {e}")
        return None


# ── VLM Vision (OpenAI 호환) ──────────────────────────────────────────────────

def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    return s.strip()


# PaddleOCR-VL이 위첨자를 LaTeX 조각으로 낸다: \(^{*}\)·$ ^{*} $ → *.
# 텍스트 레이어·PyMuPDF 표와 문자 단위로 대조하려면(벤치 가드레일) 같은 기호여야 한다.
_SUPERSCRIPT = re.compile(r"\\\(\s*\^\{([^{}]*)\}\s*\\\)|\$\s*\^\{([^{}]*)\}\s*\$")


def _normalize_superscripts(s: str) -> str:
    return _SUPERSCRIPT.sub(lambda m: m.group(1) if m.group(1) is not None else m.group(2), s)


def _cell_text(raw: str) -> str:
    """셀 내용 공통 정리. 줄바꿈은 <br>로 보존한다 — PyMuPDF 표와 같은 형식이라
    소스가 바뀌어도 항목 줄바꿈이 유지되고, 공백으로 뭉개면 목록 셀이 한 문장처럼 붙는다."""
    s = re.sub(r"<br\s*/?>", "<br>", raw, flags=re.I)
    s = re.sub(r"<(?!br>)[^>]+>", "", s)  # <br>만 남기고 나머지 태그 제거
    s = s.strip().replace("|", "／")
    return re.sub(r"\s*\n+\s*", "<br>", s)  # 날 줄바꿈이 남으면 markdown 행이 깨진다


def _grid_to_markdown(grid: list[list[str]]) -> Optional[str]:
    if not grid:
        return None
    width = max(len(r) for r in grid)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in grid]
    lines.insert(1, "|" + "---|" * width)
    return "\n".join(lines)


def _html_table_to_markdown(html: str) -> Optional[str]:
    """PaddleOCR-VL의 HTML 표 출력 → 파이프 markdown.

    <table> 블록마다 따로 변환해 빈 줄로 잇는다 — 모든 <tr>을 한 그리드로 합치면
    좌우로 나란한 두 표가 한 표로 뭉친다(삼성화재 쪽에서 실측).
    병합셀: 세로(rowspan)는 값 반복, 가로(colspan)는 첫 칸만 값·나머지는 빈 칸 —
    가로까지 반복하면 전폭 안내 문단이 열 수만큼 복제된다(현대해상 쪽에서 실측).
    """
    blocks = re.findall(r"<table[^>]*>(.*?)</table>", html, re.S | re.I) or [html]
    parts = [md for b in blocks if (md := _html_rows_to_markdown(b))]
    return "\n\n".join(parts) if parts else None


def _html_rows_to_markdown(html: str) -> Optional[str]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    if not rows:
        return None
    grid: list[list[str]] = []
    spans: dict[int, tuple[str, int]] = {}  # col → (rowspan 값, 남은 행 수)

    for tr in rows:
        # (?![a-zA-Z])가 없으면 <thead>가 <th>+속성 "ead"로 잡혀 attrs·cells의 짝이 어긋난다.
        cells = re.findall(r"<t[hd](?![a-zA-Z])[^>]*>(.*?)</t[hd]>", tr, re.S | re.I)
        attrs = re.findall(r"<t[hd](?![a-zA-Z])([^>]*)>", tr, re.I)
        out_row: list[str] = []
        col = 0

        def fill_pending() -> None:
            nonlocal col
            while col in spans:
                val, left = spans.pop(col)
                out_row.append(val)
                if left > 1:
                    spans[col] = (val, left - 1)
                col += 1

        for attr, cell in zip(attrs, cells):
            fill_pending()
            text = _cell_text(cell)
            rs = re.search(r'rowspan="?(\d+)"?', attr, re.I)
            cs = re.search(r'colspan="?(\d+)"?', attr, re.I)
            n_row = int(rs.group(1)) if rs else 1
            n_col = int(cs.group(1)) if cs else 1
            for k in range(n_col):
                val = text if k == 0 else ""  # 가로 병합은 첫 칸만
                out_row.append(val)
                if n_row > 1:
                    spans[col] = (val, n_row - 1)
                col += 1
        fill_pending()
        if out_row:
            grid.append(out_row)
    return _grid_to_markdown(grid)


def _otsl_to_markdown(otsl: str) -> Optional[str]:
    """PaddleOCR-VL의 OTSL 표 출력 → 파이프 markdown.

    <fcel>텍스트 = 셀, <ecel> = 빈 셀, <lcel> = 왼쪽 병합, <ucel>/<xcel> = 위 병합,
    <nl> = 행 끝. 병합 규칙은 HTML 변환과 같다 — 세로만 값 반복, 가로(lcel)는 빈 칸.
    """
    tokens = re.findall(r"<(fcel|ecel|lcel|ucel|xcel|nl)>([^<]*)", otsl)
    if not tokens:
        return None
    grid: list[list[str]] = []
    row: list[str] = []
    for tag, text in tokens:
        text = re.sub(r"\s*(?:\\n|\n)+\s*", "<br>", text.strip()).replace("|", "／")
        if tag == "nl":
            if row:
                grid.append(row)
            row = []
        elif tag == "fcel":
            row.append(text)
        elif tag == "ecel":
            row.append("")
        elif tag == "lcel":
            row.append("")  # 가로 병합은 첫 칸만
        else:  # ucel / xcel — 위 행 같은 열 값 복제
            col = len(row)
            row.append(grid[-1][col] if grid and col < len(grid[-1]) else "")
    if row:
        grid.append(row)
    return _grid_to_markdown(grid)


# ── 텍스트 레이어 스냅 (VLM_SNAP_TEXT=1) ──────────────────────────────────────

_SNAP_MAX_CHARS = 40  # 짧은 셀만 — 긴 문단 셀은 줄 단위 대조가 성립하지 않는다
VLM_SNAP_RATIO = float(os.environ.get("VLM_SNAP_RATIO", "0.8"))


def _lines_in_boxes(fitz_page, boxes: list[list[float]]) -> list[str]:
    """박스(+crop 여백) 안에 중심점이 있는 텍스트 레이어 줄 — 스냅 대조의 사전.

    parse_text가 표 영역이라 본문에서 뺀 바로 그 줄들이다. crop 렌더와 같은 여백을 써야
    경계에 걸친 줄이 화면에는 보이는데 사전에는 없는 어긋남이 안 생긴다.
    """
    pad = _CLIP_PAD_PX * 72.0 / VLM_DPI
    out: list[str] = []
    for blk in fitz_page.get_text("dict")["blocks"]:
        if blk.get("type", 1) != 0:
            continue
        for ln in blk.get("lines", []):
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            if not txt:
                continue
            cx = (ln["bbox"][0] + ln["bbox"][2]) / 2
            cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
            if any(b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad
                   for b in boxes):
                out.append(txt)
    return out


def snap_cells_to_text(md: str, lines: list[str]) -> tuple[str, int]:
    """VLM 표의 짧은 셀을 텍스트 레이어 줄로 교정한다. (교정된 md, 교정 셀 수) 반환.

    대조는 공백 제거본으로 한다. 레이어와 이미 정확히 일치하는 셀은 건드리지 않고,
    ratio >= VLM_SNAP_RATIO로 가장 가까운 줄이 있으면 그 줄의 원문으로 바꾼다.
    <br>가 든 셀(여러 항목)과 긴 셀(>_SNAP_MAX_CHARS)은 제외 — 셀 하나가 레이어 여러
    줄에 걸쳐 있어 줄 단위 대조가 안 맞는다.
    """
    import difflib

    if not lines:
        return md, 0
    dense_lines = [(re.sub(r"\s+", "", ln), ln) for ln in lines]
    exact = {d for d, _ in dense_lines}
    snapped = 0
    out_rows: list[str] = []
    matcher = difflib.SequenceMatcher()
    for row in md.splitlines():
        if not row.strip().startswith("|"):
            out_rows.append(row)
            continue
        cells = row.strip().strip("|").split("|")
        changed = False
        for i, cell in enumerate(cells):
            c = cell.strip()
            if (not c or set(c) <= {"-", ":"} or "<br>" in c
                    or len(c) > _SNAP_MAX_CHARS):
                continue
            dense = re.sub(r"\s+", "", c)
            if not dense or dense in exact:
                continue
            matcher.set_seq2(dense)  # seq1만 바꾸면 seq2 캐시가 재사용된다
            best_ratio, best_text = 0.0, None
            for d, orig in dense_lines:
                matcher.set_seq1(d)
                if (matcher.real_quick_ratio() <= best_ratio
                        or matcher.quick_ratio() <= best_ratio):
                    continue
                r = matcher.ratio()
                if r > best_ratio:
                    best_ratio, best_text = r, orig
            if best_ratio >= VLM_SNAP_RATIO and best_text is not None:
                cells[i] = f" {best_text.replace('|', '／')} "
                snapped += 1
                changed = True
        out_rows.append("|" + "|".join(cells) + "|" if changed else row)
    return "\n".join(out_rows), snapped


def _looping_lines(text: str) -> bool:
    """같은 줄이 VLM_LOOP_CELLS번 연속 — OCR 출력의 반복 루프 지문(_looping_cells의 짝)."""
    streak, prev = 0, None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s == prev:
            streak += 1
            if streak >= VLM_LOOP_CELLS:
                return True
        else:
            streak, prev = 1, s
    return False


def _looping_cells(md: str) -> bool:
    """같은 값의 셀이 VLM_LOOP_CELLS번 연속되는가 — 반복 루프에 빠진 출력의 지문.

    빈 칸(가로 병합·ecel)과 구분선(---)은 세지 않는다. 행이 바뀌어도 연속은 이어진다 —
    루프 출력은 행 구분조차 반복하기 때문이다.
    """
    streak, prev = 0, None
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            streak, prev = 0, None
            continue
        for cell in line.strip("|").split("|"):
            c = cell.strip()
            if not c or set(c) <= {"-", ":"}:
                continue
            if c == prev:
                streak += 1
                if streak >= VLM_LOOP_CELLS:
                    return True
            else:
                streak, prev = 1, c
    return False


# crop 여백(렌더 픽셀). 레이아웃 박스가 괘선에 딱 붙으면 바깥 선이 반쯤 잘려 표 테두리를
# 잃는 경우가 있다 — 박스 둘레에 이만큼을 더해 자른다.
_CLIP_PAD_PX = 8


def render_page(fitz_page, pno: int,
                clip: Optional[list[float]] = None) -> Optional[tuple[str, int]]:
    """페이지(또는 clip 영역) → (PNG base64, 픽셀 수). 렌더링 실패면 None.

    clip은 PDF 포인트 bbox(레이아웃 박스). 여백 _CLIP_PAD_PX를 두고 쪽 경계로 자른다.

    PyMuPDF 객체는 스레드 간에 안전하지 않다. VLM 호출을 병렬로 보내더라도 렌더링은 문서를 연
    스레드에서 해서 바이트로 넘긴다 — 그래서 호출(call_vlm)과 나눠 둔다.
    """
    import base64

    try:
        kwargs: dict = {"dpi": VLM_DPI}
        if clip is not None:
            import pymupdf
            pad = _CLIP_PAD_PX * 72.0 / VLM_DPI
            kwargs["clip"] = pymupdf.Rect(clip[0] - pad, clip[1] - pad,
                                          clip[2] + pad, clip[3] + pad) & fitz_page.rect
        pix = fitz_page.get_pixmap(**kwargs)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    except Exception as e:
        logger.warning(f"p{pno}: 이미지 렌더링 실패: {e}")
        return None
    # 비전 토큰 수는 입력 픽셀 수에 비례한다(패치 크기는 모델별). 가짜 페이지엔 크기가 없다.
    image_px = (getattr(pix, "width", 0) or 0) * (getattr(pix, "height", 0) or 0)
    return img_b64, image_px


def call_vlm(img_b64: str, pno: int, image_px: int = 0,
             record: Optional[Callable[[dict], None]] = None,
             task: str = "table") -> Optional[str]:
    """렌더된 이미지 → VLM → 결과. 호출 실패면 _VLMCallFailed.

    task="table"이면 markdown 표(표 없으면 None), task="ocr"이면 쪽 본문 텍스트(빈 쪽이면
    None) — 텍스트 레이어 없는 쪽의 본문 복구(#74)가 쓴다.

    record는 호출 기록(소요·결과·토큰·잘림)을 받는다. 없으면 모듈 기록(_vision_page_stats)에
    쌓는다. 병렬 호출에서는 호출부(VlmStage)가 자기 기록으로 받는다. 스레드에서 불려도 되도록
    모듈 상태는 record 밖에서는 건드리지 않는다.
    """
    import requests

    record = record or _record_vlm_call
    payload: dict = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": VLM_OCR_PROMPT if task == "ocr" else VLM_PROMPT},
            ],
        }],
        "temperature": 0,
        "max_tokens": VLM_MAX_TOKENS,
    }
    if VLM_MODEL:
        payload["model"] = VLM_MODEL

    stat: dict = {"page": pno, "image_px": image_px, "task": task}
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{VLM_URL}/v1/chat/completions", json=payload,
                             timeout=(VLM_CONNECT_TIMEOUT, VLM_TIMEOUT))
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        out = choice["message"]["content"].strip()
    except Exception as e:
        stat.update(seconds=round(time.perf_counter() - t0, 2), outcome=_vlm_outcome(e))
        record(stat)
        logger.warning(f"p{pno}: 로컬 VLM 실패: {e}",
                       extra={"event": "vlm_call_failed", "page": pno,
                              "error": f"{type(e).__name__}: {e}"})
        raise _VLMCallFailed(str(e)) from e

    usage = body.get("usage") or {}
    stat.update(seconds=round(time.perf_counter() - t0, 2),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                # "length"면 max_tokens(VLM_MAX_TOKENS)에서 잘렸다. 운영에서 VLM 지연 p90(222초)이
                # 4096토큰을 생성 속도(18.7토큰/초)로 끝까지 뽑는 시간과 같았다 —
                # 잘린 표는 시간도 품질도 잃는다.
                finish_reason=choice.get("finish_reason"))
    if task == "ocr":
        result = _strip_fences(_normalize_superscripts(out)).strip() or None
        degenerate = _looping_lines(result) if result else False
    else:
        result = _parse_vlm_output(out)
        degenerate = _looping_cells(result) if result else False
    # 퇴행 출력 폐기 — 운영 기준선에서 호출의 28%가 상한에서 잘렸고, Colab 실험에서 그 대부분이
    # 백지 입력의 반복 루프였다. 잘린 표는 행이 끊겨 구조가 틀리므로 반쪽을 싣는 대신 버리고
    # 그 쪽은 CPU 소스 폴백에 맡긴다. outcome으로 남겨 비율을 지표에서 본다.
    if result:
        reason = ("truncated" if choice.get("finish_reason") == "length"
                  else "loop" if degenerate else None)
        if reason:
            stat["outcome"] = reason
            record(stat)
            logger.warning(f"p{pno}: VLM 퇴행 출력 폐기 ({reason})",
                           extra={"event": "vlm_output_discarded", "page": pno,
                                  "reason": reason, "task": task})
            return None
    if task == "ocr":
        stat["outcome"] = "text" if result else "empty"
    else:
        stat["outcome"] = "table" if result else "empty"
    record(stat)
    return result


def extract_vision_local(fitz_page, pno: int, *,
                         raise_on_error: bool = False) -> Optional[str]:
    """fitz page → OpenAI 호환 VLM → markdown 표 문자열. 표 없으면 None.

    Ollama·llama-server·vLLM 모두 /v1/chat/completions를 같은 규격으로 받는다.
    구분은 VLM_MODEL 하나뿐이다 — Ollama·vLLM은 필수(vLLM은 served-model-name), llama-server는 생략.

    raise_on_error=True면 호출 실패를 _VLMCallFailed로 올린다. 기본값은 None 반환이라
    기존 계약(실패가 문서 전체를 죽이지 않는다)이 그대로 유지된다. 차단기만 이 구분이
    필요하다 — 표 없는 페이지가 연속된 걸 서버 장애로 오인하면 멀쩡한 표를 버린다.
    """
    rendered = render_page(fitz_page, pno)
    if rendered is None:
        return None
    img_b64, image_px = rendered
    try:
        return call_vlm(img_b64, pno, image_px)
    except _VLMCallFailed:
        if raise_on_error:
            raise
        return None


def _record_vlm_call(stat: dict) -> None:
    """호출 한 건을 문서 기록에 쌓고 구조화 로그로도 남긴다(Loki에서 분포를 보는 용도)."""
    _vision_page_stats.append(stat)
    logger.info(f"p{stat['page']}: VLM {stat['outcome']} {stat['seconds']}s",
                extra={"event": "vlm_page", **stat})


def _parse_vlm_output(out: str) -> Optional[str]:
    if not out:
        return None
    out = _normalize_superscripts(out)
    if "<fcel>" in out or "<ecel>" in out:
        return _otsl_to_markdown(out)
    if "<table" in out.lower() or "<tr" in out.lower():
        return _html_table_to_markdown(out)
    md = _strip_fences(out)
    return md if "|" in md else None


def extract_surya_tables(pdf_path: str, pages: list[int]) -> dict[int, str]:
    """Surya OCR로 표 페이지 일괄 처리 → {1-based page: markdown}.

    surya_ocr CLI를 PDF+page_range로 한 번 호출 (서버 1회 기동, 페이지 순차 처리).
    """
    import json as _json
    import pathlib
    import sys

    if not pages:
        return {}
    surya_bin = str(pathlib.Path(sys.executable).parent / "surya_ocr")
    if not os.path.exists(surya_bin):
        logger.warning("surya_ocr 미설치 — surya 백엔드 건너뜀")
        return {}
    env = {**os.environ, "LLAMA_CPP_BINARY": LLAMA_CPP_BINARY}
    page_range = ",".join(str(p - 1) for p in pages)  # surya는 0-based

    with tempfile.TemporaryDirectory() as td:
        try:
            r = subprocess.run(
                [surya_bin, pdf_path, "--page_range", page_range, "--output_dir", td],
                env=env, capture_output=True, text=True,
                timeout=max(VLM_TIMEOUT, 120 * len(pages)),
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"surya 타임아웃 ({len(pages)}페이지)")
            return {}
        if r.returncode != 0:
            logger.warning(f"surya 실패: {(r.stderr or '')[-300:]}")
            return {}

        results_files = list(pathlib.Path(td).rglob("results.json"))
        if not results_files:
            logger.warning("surya 결과 파일 없음")
            return {}
        data = _json.loads(results_files[0].read_text())
        entries = next(iter(data.values()))

    # 결과 page 값의 기준(0/1-based)을 요청 페이지와 대조해서 판별
    p_vals = [e.get("page") for e in entries]
    if set(p_vals) == set(pages):
        def page_of(e, i):
            return e["page"]
    elif set(p_vals) == {p - 1 for p in pages}:
        def page_of(e, i):
            return e["page"] + 1
    else:  # 순서 매칭 폴백
        def page_of(e, i):
            return pages[i]

    out: dict[int, str] = {}
    for i, entry in enumerate(entries):
        mds = []
        for b in entry.get("blocks", []):
            html = b.get("html", "")
            if b.get("label") == "Table" or "<table" in html.lower():
                md = _html_table_to_markdown(html)
                if md:
                    mds.append(md)
        if mds:
            out[page_of(entry, i)] = "\n\n".join(mds)
    return out


def extract_vision(fitz_page, pno: int) -> Optional[str]:
    """페이지 → VLM → markdown 표. 백엔드가 off거나 상한·차단기에 걸리면 None."""
    global _vision_call_count, _vision_fail_streak
    if VLM_BACKEND == "off":
        return None
    if vision_circuit_open():
        return None  # 이미 접었다. 로그는 접는 순간 한 번만 남긴다
    if vision_budget_exhausted():
        return None  # 상한 로그도 문서 루프가 한 번만 남긴다 (페이지마다 찍으면 수백 줄)
    _vision_call_count += 1
    try:
        md = extract_vision_local(fitz_page, pno, raise_on_error=True)
    except _VLMCallFailed:
        _vision_fail_streak += 1
        if vision_circuit_open():
            # 조용히 접으면 "표가 원래 없는 문서"와 구분이 안 된다. 반드시 소리를 낸다.
            logger.error(
                f"VLM이 연속 {VLM_FAIL_STREAK}회 실패했다 — 이 문서의 남은 페이지는 "
                f"VLM을 건너뛴다 (p{pno}에서 차단)",
                extra={"event": "vlm_circuit_open", "page": pno,
                       "streak": _vision_fail_streak})
        return None
    # 호출이 성공했으면 표가 있든 없든 서버는 살아 있다 — 연속 카운터를 푼다.
    _vision_fail_streak = 0
    return md


# ── VLM 투입 대상 선정 ────────────────────────────────────────────────────────

def _page_signals(page) -> tuple[int, int, float]:
    """(괘선 수, 텍스트 길이, 이미지가 덮는 면적비). get_text는 한 번만 호출한다."""
    area = abs(page.rect.width * page.rect.height) or 1.0
    text_len = 0
    img_frac = 0.0
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") == 1:  # 이미지 블록
            x0, y0, x1, y1 = blk["bbox"]
            img_frac += abs((x1 - x0) * (y1 - y0)) / area
        else:
            for ln in blk.get("lines", []):
                for sp in ln["spans"]:
                    text_len += len(sp["text"].strip())
    return len(page.get_drawings()), text_len, img_frac


def gap_pages(doc, page_numbers: list[int], pymupdf_tables: dict[int, str]) -> list[int]:
    """PyMuPDF가 닿지 못한 페이지를 우선순위 순으로 돌려준다.

    1순위 텍스트 레이어가 내용을 담고 있지 않은 페이지 — 아예 비었거나, 페이지를 덮는
          이미지가 있는데 텍스트가 희박하다(OCR이 못 읽은 스캔 페이지). 텍스트 기반
          소스는 손댈 게 없으므로 VLM만이 유일한 수단이다. 예산을 여기부터 쓴다.
    2순위 괘선이 많은데 표로 인식되지 않은 페이지 — 괘선이 끊겼거나 PyMuPDF가 놓친 표.

    '이미지가 있다'를 신호로 쓰지 않는 이유는 _HINT_IMG_FRAC 주석 참고.
    """
    blind: list[int] = []
    ruled: list[int] = []
    for pno in page_numbers:
        if pno in pymupdf_tables:
            continue
        try:
            n_draw, n_text, img_frac = _page_signals(doc[pno - 1])
        except Exception as e:  # 한 페이지의 판정 실패가 문서를 막으면 안 된다
            logger.debug(f"p{pno}: 페이지 판정 실패: {e}")
            continue
        if n_text == 0 or (n_text < _HINT_MIN_TEXT and img_frac >= _HINT_IMG_FRAC):
            blind.append(pno)
        elif n_draw >= _HINT_DRAWINGS:
            ruled.append(pno)
    return blind + ruled


def vlm_target_pages(pymupdf_pages: list[int], gaps: list[int]) -> list[int]:
    """VLM_TARGET에 따라 투입 대상을 정한다. 순서가 곧 예산 소진 순서다."""
    if VLM_TARGET == "tables":
        return pymupdf_pages
    if VLM_TARGET == "both":
        return gaps + pymupdf_pages
    return gaps


# ── 문서 단위 추출 ────────────────────────────────────────────────────────────
# CPU 단계(PyMuPDF + VLM 대상 선정)와 GPU 단계(VLM)를 따로 부를 수 있게 나눠 둔다.
# 한 함수에 묶여 있으면 VLM 시간이 표 추출 전체에 묻혀 따로 잴 수 없고, VLM만 설정을
# 바꿔 다시 돌리는 실험도 못 한다. extract_tables_for_doc은 둘을 이어 부르는 예전
# 진입점으로 남긴다(호출부가 eval 스크립트까지 여럿이다 — 레이아웃 없는 경로로 돈다).

def extract_cpu_tables(
    pdf_path: str,
    page_numbers: list[int],
    use_vision: bool = True,
    layout_boxes: Optional[dict[int, list[list[float]]]] = None,
    ocr_pages: Optional[list[int]] = None,
) -> tuple[dict[str, dict[int, str]], list, dict]:
    """CPU 소스(PyMuPDF)로 표를 뽑고 VLM 투입 대상을 정한다.

    layout_boxes(쪽 → 표 bbox 목록, PDF 포인트)가 있으면 VLM 대상은 쪽이 아니라 박스다 —
    {"page", "bbox"} dict 목록을 쪽 순서로 돌려주고, gap 휴리스틱은 쓰지 않는다(레이아웃이
    그 역할을 대신한다). None이면(LAYOUT_BACKEND=off) 예전 그대로 gap 휴리스틱 + 쪽 목록.
    PyMuPDF 표는 두 경로 모두 뽑는다 — 박스 쪽에서 VLM이 실패하면 폴백이다.

    ocr_pages(레이아웃이 가려낸 텍스트 레이어 없는 쪽, #74)는 {"page", "ocr": True}
    대상이 되어 쪽 전체가 본문 복구(OCR 프롬프트)로 간다. 레이아웃 경로에서만 쓴다.

    Returns:
        (table_sources, vlm_targets, info)
        table_sources: {"pymupdf": {page: md}}
        vlm_targets:   [{"page","ocr"} | {"page","bbox"}] 또는 [page]. 순서 = 예산 소진 순서
        info:          {"pymupdf_pages", "gap_pages", "crops", "ocr_pages"} — 로그·계측용
    """
    import pymupdf as fitz  # 'fitz'는 구 이름 — 그대로 쓰면 임포트마다 경고가 찍힌다

    pymupdf_tables: dict[int, str] = {}

    with fitz.open(pdf_path) as doc:
        for pno in page_numbers:
            fitz_page = doc[pno - 1]
            md = extract_pymupdf(fitz_page)
            if md:
                pymupdf_tables[pno] = md

        table_pages = sorted(pymupdf_tables.keys())
        gaps = [] if layout_boxes is not None else gap_pages(doc, page_numbers, pymupdf_tables)

    table_sources: dict[str, dict[int, str]] = {"pymupdf": pymupdf_tables}

    targets: list = []
    if use_vision and VLM_BACKEND != "off":
        if layout_boxes is not None:
            # 본문 복구(OCR)가 표 crop보다 먼저다 — 표에는 PyMuPDF 폴백이 있지만, 이 쪽의
            # 본문은 이것 말고 어디에도 없다. 예산(VISION_MAX_CROPS)이 순서대로 깎인다.
            targets = ([{"page": pno, "ocr": True} for pno in (ocr_pages or [])]
                       + [{"page": pno, "bbox": list(bbox)}
                          for pno in sorted(layout_boxes) for bbox in layout_boxes[pno]])
        else:
            targets = vlm_target_pages(table_pages, gaps)
    info = {"pymupdf_pages": len(table_pages), "gap_pages": len(gaps),
            "crops": len(targets) if layout_boxes is not None else 0,
            "ocr_pages": len(ocr_pages or []) if layout_boxes is not None else 0}
    return table_sources, targets, info


def extract_vlm_tables(
    pdf_path: str,
    targets: list,
    total_pages: int,
    info: Optional[dict] = None,
) -> dict[int, str]:
    """VLM 대상(쪽 번호 또는 {"page","bbox"} crop)의 표를 뽑는다 → {page: markdown}.

    crop 대상이면 예산이 VISION_MAX_PAGES(쪽)가 아니라 VISION_MAX_CROPS(박스)다.
    문서 경계이므로 호출 상한과 차단기를 여기서 되돌린다. total_pages·info는 로그에만 쓴다.
    """
    reset_vision_counter()
    if VLM_BACKEND == "off":
        return {}

    info = info or {}
    total = len(targets)
    crops = bool(targets) and isinstance(targets[0], dict)
    unit = "영역" if crops else "페이지"
    logger.info(
        f"VLM 대상: {total}{unit} (전체 {total_pages}페이지, "
        f"{'layout crop' if crops else f'target={VLM_TARGET}'}, "
        f"PyMuPDF표 {info.get('pymupdf_pages', 0)}쪽/미탐지 {info.get('gap_pages', 0)}쪽, "
        f"backend={VLM_BACKEND}{f', model={VLM_MODEL}' if VLM_MODEL else ''})",
        extra={"event": "vlm_targets", "target": "crop" if crops else VLM_TARGET,
               "pages": total, "pymupdf_pages": info.get("pymupdf_pages", 0),
               "gap_pages": info.get("gap_pages", 0)})
    if VLM_BACKEND == "surya":
        pages = sorted({t["page"] for t in targets}) if crops else targets
        return extract_surya_tables(pdf_path, pages)

    import pymupdf as fitz

    from .vlm_stage import VlmStage

    stage = VlmStage(budget=VISION_MAX_CROPS if crops else None)
    with fitz.open(pdf_path) as doc:
        vlm_tables = stage.run(doc, targets)
        if VLM_SNAP_TEXT and crops and vlm_tables:
            _snap_tables(doc, [t for t in targets if "bbox" in t], vlm_tables)
    _publish(stage)
    return vlm_tables


def _snap_tables(doc, targets: list[dict], vlm_tables: dict[int, str]) -> None:
    """crop 결과의 셀을 그 쪽 레이아웃 박스 안 텍스트 레이어 줄로 교정한다(제자리).

    문서 단위 교정 수는 _snap_count로 남아 단계 지표(snapped_cells)가 된다.
    """
    global _snap_count
    boxes_by_page: dict[int, list[list[float]]] = {}
    for t in targets:
        boxes_by_page.setdefault(t["page"], []).append(t["bbox"])
    snapped = 0
    for pno in list(vlm_tables):
        lines = _lines_in_boxes(doc[pno - 1], boxes_by_page.get(pno, []))
        vlm_tables[pno], n = snap_cells_to_text(vlm_tables[pno], lines)
        snapped += n
    _snap_count = snapped
    if snapped:
        logger.info(f"텍스트 레이어 스냅: 셀 {snapped}개 교정",
                    extra={"event": "vlm_snap", "snapped": snapped})


def _publish(stage) -> None:
    """VlmStage가 들고 있던 문서 단위 상태를 모듈 상태로 옮긴다.

    vision_circuit_open()·vision_page_stats()·_vision_call_count를 읽는 호출부(stages.vlm,
    extract_vision 경로, 테스트)가 예전 계약 그대로 동작하게 하려는 것이다. 단계가 끝난 뒤
    한 번 옮기므로 병렬 호출 중에는 모듈 상태를 아무도 건드리지 않는다.
    """
    global _vision_call_count, _vision_fail_streak, _vision_page_stats, _ocr_texts
    _vision_call_count = stage.calls
    _vision_fail_streak = VLM_FAIL_STREAK if stage.circuit_open else stage.streak
    _vision_page_stats = list(stage.stats)
    _ocr_texts = dict(stage.ocr_texts)


def extract_tables_for_doc(
    pdf_path: str,
    page_numbers: list[int],
    use_vision: bool = True,
) -> dict[str, dict[int, str]]:
    """전체 문서의 페이지별 표를 다중 소스로 추출.

    Args:
        page_numbers: 처리할 1-based 페이지 번호 목록.

    Returns:
        {"pymupdf": {page: md}, "pdfplumber": {page: md}, "camelot": {page: md}, "vlm": {page: md}}
        설치된 도구만 포함. combine.py가 페이지별 best-of를 선택.
    """
    table_sources, targets, info = extract_cpu_tables(pdf_path, page_numbers, use_vision)
    if use_vision and VLM_BACKEND != "off":
        vlm_tables = extract_vlm_tables(pdf_path, targets, len(page_numbers), info)
        if vlm_tables:
            table_sources["vlm"] = vlm_tables
    else:
        reset_vision_counter()

    counts = {k: len(v) for k, v in table_sources.items()}
    logger.info(f"표 추출 완료: {counts}")
    return table_sources
