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
from typing import Optional

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
# 접속 타임아웃은 따로 짧게 둔다. 같은 호스트의 Ollama라 붙는 건 즉시여야 하고,
# 서버가 아예 없으면 밀리초 안에 알아야지 읽기 타임아웃까지 기다릴 이유가 없다.
VLM_CONNECT_TIMEOUT = float(os.environ.get("VLM_CONNECT_TIMEOUT", "5"))
# 차단기: 연속 N회 실패하면 이 문서의 남은 페이지는 VLM을 건너뛴다.
# 없을 때 무슨 일이 생기는지 겪었다 — 표 214페이지짜리 문서가 페이지마다 600초를
# 기다리며 35.6시간을 태웠다. 서버가 죽었으면 215번째 시도도 죽는다.
VLM_FAIL_STREAK = int(os.environ.get("VLM_FAIL_STREAK", "3"))
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
# 명시적 지시가 필요하다. 기본값은 후자에 맞추고, PaddleOCR-VL을 쓰면 이 값을 바꾼다.
# 규칙은 claude 백엔드를 쓰던 시절 쌓인 것을 옮겼다 — 약관 표는 숫자·한자가 많아
# 의역이 곧 오답이고, 읽기 어려운 셀에서 멈추면 페이지 전체를 잃는다.
_DEFAULT_VLM_PROMPT = (
    "이 페이지 이미지의 표를 GitHub 마크다운 표로만 옮겨라.\n"
    "규칙(엄수):\n"
    "- 셀 텍스트는 원문 그대로(숫자·한자·줄바꿈 보존). 의역·요약 금지.\n"
    "- 병합된 셀은 값을 반복해 채워라.\n"
    "- 읽기 어려운 셀은 [?]로 표기하고 계속 진행하라.\n"
    "- 표가 여러 개면 빈 줄로 구분하라.\n"
    "- 표가 전혀 없으면 아무것도 출력하지 마라.\n"
    "- 설명·머리말·코드펜스 없이 표 마크다운만 출력하라."
)
VLM_PROMPT = os.environ.get("VLM_PROMPT", _DEFAULT_VLM_PROMPT)
LLAMA_CPP_BINARY = os.environ.get(
    "LLAMA_CPP_BINARY",
    os.path.expanduser("~/.local/llama.cpp/llama-b10182/llama-server"),
)

_vision_call_count = 0
_vision_fail_streak = 0


class _VLMCallFailed(Exception):
    """VLM 호출 자체가 실패했다 — '표가 없다'(정상 응답)와 구분하기 위한 신호."""


def reset_vision_counter() -> None:
    """문서 경계에서 호출 상한과 차단기를 함께 되돌린다.

    차단기를 문서 단위로 두는 이유: 서버가 잠깐 죽었다 살아난 경우 다음 문서까지
    포기할 이유가 없다. 반대로 한 문서 안에서는 한 번 접었으면 끝까지 접는다.
    """
    global _vision_call_count, _vision_fail_streak
    _vision_call_count = 0
    _vision_fail_streak = 0


def vision_circuit_open() -> bool:
    """차단기가 열렸는가. 문서 루프가 남은 페이지를 건너뛰는 데 쓴다."""
    return _vision_fail_streak >= VLM_FAIL_STREAK


def vision_budget_exhausted() -> bool:
    """문서당 VLM 호출 상한에 닿았는가."""
    return _vision_call_count >= VISION_MAX_PAGES


# ── PyMuPDF ───────────────────────────────────────────────────────────────────

def extract_pymupdf(fitz_page) -> Optional[str]:
    """fitz page → markdown 표 문자열. 표가 없거나 실패 시 None."""
    try:
        tabs = fitz_page.find_tables()
        if not tabs.tables:
            return None
        parts = [tab.to_markdown() for tab in tabs.tables if tab.to_markdown().strip()]
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
                tabs = pdf.pages[p - 1].extract_tables() or []
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


def _html_table_to_markdown(html: str) -> Optional[str]:
    """PaddleOCR-VL의 HTML 표 출력 → 파이프 markdown. 병합셀은 값 복제로 평탄화."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    if not rows:
        return None
    grid: list[list[str]] = []
    spans: dict[int, tuple[str, int]] = {}  # col → (rowspan 값, 남은 행 수)

    for tr in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S | re.I)
        attrs = re.findall(r"<t[hd]([^>]*)>", tr, re.I)
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
            text = re.sub(r"<br\s*/?>", " ", cell)
            text = re.sub(r"<[^>]+>", "", text).strip().replace("|", "／")
            rs = re.search(r'rowspan="?(\d+)"?', attr, re.I)
            cs = re.search(r'colspan="?(\d+)"?', attr, re.I)
            n_row = int(rs.group(1)) if rs else 1
            n_col = int(cs.group(1)) if cs else 1
            for _ in range(n_col):
                out_row.append(text)
                if n_row > 1:
                    spans[col] = (text, n_row - 1)
                col += 1
        fill_pending()
        if out_row:
            grid.append(out_row)
    if not grid:
        return None
    width = max(len(r) for r in grid)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in grid]
    lines.insert(1, "|" + "---|" * width)
    return "\n".join(lines)


def _otsl_to_markdown(otsl: str) -> Optional[str]:
    """PaddleOCR-VL의 OTSL 표 출력 → 파이프 markdown.

    <fcel>텍스트 = 셀, <ecel> = 빈 셀, <lcel> = 왼쪽 병합, <ucel>/<xcel> = 위 병합,
    <nl> = 행 끝. 병합셀은 값 복제로 평탄화.
    """
    tokens = re.findall(r"<(fcel|ecel|lcel|ucel|xcel|nl)>([^<]*)", otsl)
    if not tokens:
        return None
    grid: list[list[str]] = []
    row: list[str] = []
    for tag, text in tokens:
        text = text.replace("\\n", " ").replace("\n", " ").replace("|", "／").strip()
        if tag == "nl":
            if row:
                grid.append(row)
            row = []
        elif tag == "fcel":
            row.append(text)
        elif tag == "ecel":
            row.append("")
        elif tag == "lcel":
            row.append(row[-1] if row else "")
        else:  # ucel / xcel — 위 행 같은 열 값 복제
            col = len(row)
            row.append(grid[-1][col] if grid and col < len(grid[-1]) else "")
    if row:
        grid.append(row)
    if not grid:
        return None
    width = max(len(r) for r in grid)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in grid]
    lines.insert(1, "|" + "---|" * width)
    return "\n".join(lines)


def extract_vision_local(fitz_page, pno: int, *,
                         raise_on_error: bool = False) -> Optional[str]:
    """fitz page → OpenAI 호환 VLM → markdown 표 문자열. 표 없으면 None.

    Ollama와 llama-server 둘 다 /v1/chat/completions를 같은 규격으로 받는다.
    구분은 VLM_MODEL 하나뿐이다 — Ollama는 필수, llama-server는 생략.

    raise_on_error=True면 호출 실패를 _VLMCallFailed로 올린다. 기본값은 None 반환이라
    기존 계약(실패가 문서 전체를 죽이지 않는다)이 그대로 유지된다. 차단기만 이 구분이
    필요하다 — 표 없는 페이지가 연속된 걸 서버 장애로 오인하면 멀쩡한 표를 버린다.
    """
    import base64

    import requests

    try:
        pix = fitz_page.get_pixmap(dpi=VLM_DPI)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    except Exception as e:
        logger.warning(f"p{pno}: 이미지 렌더링 실패: {e}")
        return None

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
        "max_tokens": 4096,
    }
    if VLM_MODEL:
        payload["model"] = VLM_MODEL

    try:
        resp = requests.post(f"{VLM_URL}/v1/chat/completions", json=payload,
                             timeout=(VLM_CONNECT_TIMEOUT, VLM_TIMEOUT))
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"p{pno}: 로컬 VLM 실패: {e}",
                       extra={"event": "vlm_call_failed", "page": pno,
                              "error": f"{type(e).__name__}: {e}"})
        if raise_on_error:
            raise _VLMCallFailed(str(e)) from e
        return None

    if not out:
        return None
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
    import pymupdf as fitz  # 'fitz'는 구 이름 — 그대로 쓰면 임포트마다 경고가 찍힌다

    pymupdf_tables: dict[int, str] = {}
    reset_vision_counter()

    with fitz.open(pdf_path) as doc:
        for pno in page_numbers:
            fitz_page = doc[pno - 1]
            md = extract_pymupdf(fitz_page)
            if md:
                pymupdf_tables[pno] = md

        table_sources: dict[str, dict[int, str]] = {"pymupdf": pymupdf_tables}
        table_pages = sorted(pymupdf_tables.keys())
        gaps = gap_pages(doc, page_numbers, pymupdf_tables)

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
        if use_vision and VLM_BACKEND != "off":
            targets = vlm_target_pages(table_pages, gaps)
            total = len(targets)
            logger.info(
                f"VLM 대상: {total}페이지 (전체 {len(page_numbers)}페이지 중 "
                f"target={VLM_TARGET}, PyMuPDF표 {len(table_pages)}쪽/미탐지 {len(gaps)}쪽, "
                f"backend={VLM_BACKEND}{f', model={VLM_MODEL}' if VLM_MODEL else ''})",
                extra={"event": "vlm_targets", "target": VLM_TARGET, "pages": total,
                       "pymupdf_pages": len(table_pages), "gap_pages": len(gaps)})
            if VLM_BACKEND == "surya":
                vlm_tables = extract_surya_tables(pdf_path, targets)
            else:
                vlm_tables = {}
                for i, pno in enumerate(targets, 1):
                    # 차단기가 열렸으면 남은 페이지는 호출도 로그도 하지 않는다.
                    # 안 끊으면 "처리 중" 줄만 수백 개 쌓여 로그가 사실과 어긋난다.
                    if vision_circuit_open():
                        logger.warning(
                            f"VLM 차단 — 남은 {total - i + 1}페이지 건너뜀",
                            extra={"event": "vlm_pages_skipped",
                                   "skipped": total - i + 1, "total": total})
                        break
                    if vision_budget_exhausted():
                        # 품질이 아니라 시간을 위해 접는 것이라 level이 다르다 —
                        # 남은 페이지도 PyMuPDF/pdfplumber 결과는 그대로 쓴다.
                        logger.info(
                            f"VLM 상한 {VISION_MAX_PAGES}페이지 도달 — 남은 "
                            f"{total - i + 1}페이지는 PyMuPDF/pdfplumber 결과만 쓴다",
                            extra={"event": "vlm_budget_exhausted",
                                   "limit": VISION_MAX_PAGES,
                                   "skipped": total - i + 1, "total": total})
                        break
                    logger.info(f"VLM [{i}/{total}] p{pno} 처리 중...")
                    fitz_page = doc[pno - 1]
                    md_vision = extract_vision(fitz_page, pno)
                    if md_vision:
                        vlm_tables[pno] = md_vision
                        logger.info(f"VLM [{i}/{total}] p{pno} 완료 (표 추출됨)")
            if vlm_tables:
                table_sources["vlm"] = vlm_tables

    counts = {k: len(v) for k, v in table_sources.items()}
    logger.info(f"표 추출 완료: {counts}")
    return table_sources
