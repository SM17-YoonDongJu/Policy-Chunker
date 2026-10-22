"""표 추출: PyMuPDF + pdfplumber + camelot + VLM(비전).

Policy-Chunker(main) extract.py와 동일한 전략:
  - PyMuPDF    : fitz.find_tables() (괘선 있는 표, 빠름)
  - pdfplumber : 설치돼 있으면 자동 사용
  - camelot    : 설치돼 있으면 자동 사용 (ghostscript 필요)
  - VLM        : OpenAI 호환 VLM(기본: 같은 호스트 Ollama의 qwen3-vl),
                 PyMuPDF가 표를 감지한 페이지에만 실행

combine.py가 페이지별로 더블스페이스 가장 적은 소스를 선택한다.
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

# 문서당 VLM 호출 상한. 9999는 사실상 상한이 없다는 뜻이었고, 그게 처리 시간의 꼬리를
# 만들었다 — 실측에서 문서 20건(6.4%)이 전체 parse 시간의 53.4%를 먹었고, parse p50은
# 131초인데 p90이 2,879초로 22배다. 표 403페이지 중 214페이지가 표인 문서가 실재하는데,
# VLM이 완전히 건강해도 페이지당 11~25초라 그것만 40~90분이다.
#
# 50은 시간 예산에서 나온 값이다: 50 x 25초 = 약 20분이 문서당 VLM 상한이 된다.
# 상한을 넘는 페이지는 VLM을 안 거칠 뿐 표가 사라지지는 않는다 — PyMuPDF/pdfplumber
# 결과가 남고 combine.py가 페이지별 best-of를 고른다. 전부 잃는 게 아니라 일부가 열화된다.
VISION_MAX_PAGES = int(os.environ.get("VISION_MAX_PAGES", "50"))
VLM_DPI = int(os.environ.get("VLM_DPI", "150"))
# 읽기 타임아웃. GPU 경로 실측이 페이지당 21~36초이고, max_tokens가 4096이라 최악의
# 페이지(토큰을 끝까지 쓰는 경우)도 200초 안쪽이다. 300초를 넘기면 느린 게 아니라
# 고장난 것이다 — 실제로 Ollama가 CPU로 떨어졌을 때 600초로도 한 장을 못 끝냈다.
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", "300"))
# 페이지당 출력 토큰 상한. 생성이 VLM 시간의 대부분이라(vLLM 단일 요청 18.7토큰/초) 상한이 곧
# 최악 페이지의 시간이다. smoke 벤치마크(2026-10-14)에서 50건 중 7건이 4096까지 생성했고 그 7건이
# VLM 시간의 약 53%였다 — 상한까지 가는 응답은 대개 반복 폭주라 끝까지 받아도 쓸 수 없다. 줄이면
# 결과가 바뀌는 변경이라 벤치마크 가드레일로 판정한 뒤 바꾼다. 프롬프트(비전 토큰 포함, p50 약
# 2,300)와 합쳐 vLLM max-model-len(운영 6,144)을 넘지 않게 둔다.
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
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3-vl:8b-instruct")
# PaddleOCR-VL은 "Table Recognition:"이 태스크 토큰이지만, 범용 instruct VLM(qwen3-vl 등)에는
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


class _VLMCallFailed(Exception):
    """VLM 호출 자체가 실패했다 — '표가 없다'(정상 응답)와 구분하기 위한 신호."""


def reset_vision_counter() -> None:
    """문서 경계에서 호출 상한과 차단기를 함께 되돌린다.

    차단기를 문서 단위로 두는 이유: 서버가 잠깐 죽었다 살아난 경우 다음 문서까지
    포기할 이유가 없다. 반대로 한 문서 안에서는 한 번 접었으면 끝까지 접는다.
    """
    global _vision_call_count, _vision_fail_streak, _vision_page_stats
    _vision_call_count = 0
    _vision_fail_streak = 0
    _vision_page_stats = []


def vision_page_stats() -> list[dict]:
    """이 문서에서 나간 VLM 호출 기록(호출 순서). 계측용."""
    return list(_vision_page_stats)


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


# ── pdfplumber (선택) ─────────────────────────────────────────────────────────

def _rows_to_md(rows) -> str:
    rows = [[("" if c is None else str(c)).replace("\n", " ").strip() for c in r]
            for r in rows if r]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    head = rows[0]
    out = ["| " + " | ".join(head) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def extract_pdfplumber_tables(pdf_path: str, pages: list[int]) -> dict[int, str]:
    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber 미설치 — 건너뜀")
        return {}
    out: dict[int, str] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pages:
                if p - 1 >= len(pdf.pages):
                    continue
                page = pdf.pages[p - 1]
                try:
                    tabs = page.extract_tables() or []
                finally:
                    # pdfplumber는 건드린 쪽의 글자·선 객체를 문서를 닫을 때까지 캐시에 들고 있다.
                    # 한국어 약관은 쪽당 글자 2천 개 안팎이고 글자 하나가 2KB짜리 dict라 쪽당
                    # 3~5MB다. 434쪽 문서에서 표·gap 356쪽을 한 번에 돌자 2GB를 넘겨 파드 한도
                    # 4Gi에서 죽었다(OOM 111회 재시작, 2026-10-15~20). 쪽마다 캐시를 풀면
                    # 최고점이 한 쪽 분량으로 내려간다. close()는 캐시만 비우고 문서는 안 닫는다.
                    page.close()
                mds = [_rows_to_md(t) for t in tabs]
                mds = [m for m in mds if m]
                if mds:
                    out[p] = "\n\n".join(mds)
    except Exception:
        pass
    return out


# ── camelot (선택) ────────────────────────────────────────────────────────────

def extract_camelot_tables(pdf_path: str, pages: list[int]) -> dict[int, str]:
    try:
        import camelot
    except ImportError:
        logger.debug("camelot 미설치 — 건너뜀")
        return {}
    out: dict[int, str] = {}
    for p in pages:
        for flavor in ("lattice", "stream"):
            try:
                tl = camelot.read_pdf(pdf_path, pages=str(p), flavor=flavor)
            except Exception:
                continue
            if tl and tl.n:
                mds = [_rows_to_md(t.df.values.tolist()) for t in tl]
                mds = [m for m in mds if m]
                if mds:
                    out[p] = "\n\n".join(mds)
                    break
    return out


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


def render_page(fitz_page, pno: int) -> Optional[tuple[str, int]]:
    """페이지 → (PNG base64, 픽셀 수). 렌더링 실패면 None.

    PyMuPDF 객체는 스레드 간에 안전하지 않다. VLM 호출을 병렬로 보내더라도 렌더링은 문서를 연
    스레드에서 해서 바이트로 넘긴다 — 그래서 호출(call_vlm)과 나눠 둔다.
    """
    import base64

    try:
        pix = fitz_page.get_pixmap(dpi=VLM_DPI)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    except Exception as e:
        logger.warning(f"p{pno}: 이미지 렌더링 실패: {e}")
        return None
    # 비전 토큰 수는 입력 픽셀 수에 비례한다(Qwen 계열은 28px 패치). 가짜 페이지엔 크기가 없다.
    image_px = (getattr(pix, "width", 0) or 0) * (getattr(pix, "height", 0) or 0)
    return img_b64, image_px


def call_vlm(img_b64: str, pno: int, image_px: int = 0,
             record: Optional[Callable[[dict], None]] = None) -> Optional[str]:
    """렌더된 이미지 → VLM → markdown 표. 표 없으면 None, 호출 실패면 _VLMCallFailed.

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
                {"type": "text", "text": VLM_PROMPT},
            ],
        }],
        "temperature": 0,
        "max_tokens": VLM_MAX_TOKENS,
    }
    if VLM_MODEL:
        payload["model"] = VLM_MODEL

    stat: dict = {"page": pno, "image_px": image_px}
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
    md = _parse_vlm_output(out)
    # 퇴행 출력 폐기 — 운영 기준선에서 호출의 28%가 상한에서 잘렸고, Colab 실험에서 그 대부분이
    # 백지 입력의 반복 루프였다. 잘린 표는 행이 끊겨 구조가 틀리므로 반쪽을 싣는 대신 버리고
    # 그 쪽은 CPU 소스 폴백에 맡긴다. outcome으로 남겨 비율을 지표에서 본다.
    if md:
        reason = ("truncated" if choice.get("finish_reason") == "length"
                  else "loop" if _looping_cells(md) else None)
        if reason:
            stat["outcome"] = reason
            record(stat)
            logger.warning(f"p{pno}: VLM 퇴행 출력 폐기 ({reason})",
                           extra={"event": "vlm_output_discarded", "page": pno,
                                  "reason": reason})
            return None
    stat["outcome"] = "table" if md else "empty"
    record(stat)
    return md


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
          이미지가 있는데 텍스트가 희박하다(OCR이 못 읽은 스캔 페이지). PyMuPDF도
          pdfplumber도 손댈 게 없으므로 VLM만이 유일한 수단이다. 예산을 여기부터 쓴다.
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
# CPU 단계(PyMuPDF·pdfplumber·camelot + VLM 대상 선정)와 GPU 단계(VLM)를 따로 부를 수
# 있게 나눠 둔다. 한 함수에 묶여 있으면 VLM 시간이 표 추출 전체에 묻혀 따로 잴 수 없고,
# VLM만 설정을 바꿔 다시 돌리는 실험도 못 한다. extract_tables_for_doc은 둘을 이어 부르는
# 예전 진입점으로 남긴다(호출부가 eval 스크립트까지 여럿이다).

def extract_cpu_tables(
    pdf_path: str,
    page_numbers: list[int],
    use_vision: bool = True,
) -> tuple[dict[str, dict[int, str]], list[int], dict]:
    """CPU 소스로 표를 뽑고 VLM 투입 대상을 정한다.

    Returns:
        (table_sources, vlm_targets, info)
        table_sources: {"pymupdf": {...}, "pdfplumber": {...}, "camelot": {...}} — 설치된 것만
        vlm_targets:   VLM을 부를 페이지. 순서가 곧 예산 소진 순서. VLM을 안 쓰면 빈 목록
        info:          {"pymupdf_pages": int, "gap_pages": int} — 로그·계측용
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
        gaps = gap_pages(doc, page_numbers, pymupdf_tables)

    table_sources: dict[str, dict[int, str]] = {"pymupdf": pymupdf_tables}

    # CPU 소스는 후보를 넓혀도 싸다. 괘선 없는 표는 PyMuPDF가 못 보지만
    # pdfplumber는 보기도 하므로, GPU를 쓰기 전에 여기서 먼저 건진다.
    cpu_pages = sorted(set(table_pages) | set(gaps))

    # pdfplumber (선택)
    pp = extract_pdfplumber_tables(pdf_path, cpu_pages)
    if pp:
        table_sources["pdfplumber"] = pp

    # camelot (선택)
    cm = extract_camelot_tables(pdf_path, cpu_pages)
    if cm:
        table_sources["camelot"] = cm

    # VLM — 기본은 PyMuPDF가 닿지 못한 페이지(gap)
    targets: list[int] = []
    if use_vision and VLM_BACKEND != "off":
        targets = vlm_target_pages(table_pages, gaps)
    info = {"pymupdf_pages": len(table_pages), "gap_pages": len(gaps)}
    return table_sources, targets, info


def extract_vlm_tables(
    pdf_path: str,
    targets: list[int],
    total_pages: int,
    info: Optional[dict] = None,
) -> dict[int, str]:
    """VLM 대상 페이지의 표를 뽑는다 → {page: markdown}. 백엔드가 off면 빈 dict.

    문서 경계이므로 호출 상한과 차단기를 여기서 되돌린다.
    total_pages·info는 로그에만 쓴다.
    """
    reset_vision_counter()
    if VLM_BACKEND == "off":
        return {}

    info = info or {}
    total = len(targets)
    logger.info(
        f"VLM 대상: {total}페이지 (전체 {total_pages}페이지 중 "
        f"target={VLM_TARGET}, PyMuPDF표 {info.get('pymupdf_pages', 0)}쪽/"
        f"미탐지 {info.get('gap_pages', 0)}쪽, "
        f"backend={VLM_BACKEND}{f', model={VLM_MODEL}' if VLM_MODEL else ''})",
        extra={"event": "vlm_targets", "target": VLM_TARGET, "pages": total,
               "pymupdf_pages": info.get("pymupdf_pages", 0),
               "gap_pages": info.get("gap_pages", 0)})
    if VLM_BACKEND == "surya":
        return extract_surya_tables(pdf_path, targets)

    import pymupdf as fitz

    from .vlm_stage import VlmStage

    stage = VlmStage()
    with fitz.open(pdf_path) as doc:
        vlm_tables = stage.run(doc, targets)
    _publish(stage)
    return vlm_tables


def _publish(stage) -> None:
    """VlmStage가 들고 있던 문서 단위 상태를 모듈 상태로 옮긴다.

    vision_circuit_open()·vision_page_stats()·_vision_call_count를 읽는 호출부(stages.vlm,
    extract_vision 경로, 테스트)가 예전 계약 그대로 동작하게 하려는 것이다. 단계가 끝난 뒤
    한 번 옮기므로 병렬 호출 중에는 모듈 상태를 아무도 건드리지 않는다.
    """
    global _vision_call_count, _vision_fail_streak, _vision_page_stats
    _vision_call_count = stage.calls
    _vision_fail_streak = VLM_FAIL_STREAK if stage.circuit_open else stage.streak
    _vision_page_stats = list(stage.stats)


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
