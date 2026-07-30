"""표 추출: PyMuPDF + pdfplumber + camelot + Claude CLI(비전).

Policy-Chunker(main) extract.py와 동일한 전략:
  - PyMuPDF    : fitz.find_tables() (괘선 있는 표, 빠름)
  - pdfplumber : 설치돼 있으면 자동 사용
  - camelot    : 설치돼 있으면 자동 사용 (ghostscript 필요)
  - VLM        : claude CLI, PyMuPDF 표 탐지 페이지에만 실행

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

VISION_MAX_PAGES = int(os.environ.get("VISION_MAX_PAGES", "9999"))
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
VLM_DPI = int(os.environ.get("VLM_DPI", "150"))
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", "600"))
# 'surya' = Surya OCR(무료, 한국어 최고 품질) | 'local' = llama-server(PaddleOCR-VL, 무료·고속)
# | 'claude' = claude CLI(유료 API — 사용 자제)
VLM_BACKEND = os.environ.get("VLM_BACKEND", "surya")
VLM_URL = os.environ.get("VLM_URL", "http://localhost:8090")
LLAMA_CPP_BINARY = os.environ.get(
    "LLAMA_CPP_BINARY",
    os.path.expanduser("~/.local/llama.cpp/llama-b10182/llama-server"),
)

_vision_call_count = 0

_VLM_PROMPT = (
    "이미지를 Read로 열어, 페이지의 표를 GitHub 마크다운 표로만 옮겨라. 경로: {path}\n"
    "규칙(엄수):\n"
    "- 질문하지 마라. Bash·python 등 다른 도구를 쓰지 말고 Read만 사용하라.\n"
    "- 셀 텍스트는 원문 그대로(숫자·한자·줄바꿈 보존). 의역·요약 금지.\n"
    "- 읽기 어려운 셀은 [?]로 표기하고 계속 진행하라.\n"
    "- 표가 여러 개면 빈 줄로 구분.\n"
    "- 표가 전혀 없으면 첫 줄에 NO_TABLE 한 단어만 출력.\n"
    "- 설명·머리말 없이 표 마크다운만 출력."
)


def reset_vision_counter() -> None:
    global _vision_call_count
    _vision_call_count = 0


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


# ── Claude CLI Vision ─────────────────────────────────────────────────────────

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


def extract_vision_local(fitz_page, pno: int) -> Optional[str]:
    """fitz page → llama-server(PaddleOCR-VL) → markdown 표 문자열. 표 없으면 None."""
    import base64

    import requests

    try:
        pix = fitz_page.get_pixmap(dpi=VLM_DPI)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    except Exception as e:
        logger.warning(f"p{pno}: 이미지 렌더링 실패: {e}")
        return None

    try:
        resp = requests.post(
            f"{VLM_URL}/v1/chat/completions",
            json={
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": "Table Recognition:"},
                    ],
                }],
                "temperature": 0,
                "max_tokens": 4096,
            },
            timeout=VLM_TIMEOUT,
        )
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"p{pno}: 로컬 VLM 실패: {e}")
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
        page_of = lambda e, i: e["page"]
    elif set(p_vals) == {p - 1 for p in pages}:
        page_of = lambda e, i: e["page"] + 1
    else:  # 순서 매칭 폴백
        page_of = lambda e, i: pages[i]

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
    """fitz page → VLM → markdown 표 문자열. VLM_BACKEND로 로컬/claude 선택."""
    global _vision_call_count
    if _vision_call_count >= VISION_MAX_PAGES:
        logger.warning(f"Vision 상한({VISION_MAX_PAGES}회) 도달 — p{pno} 건너뜀")
        return None

    if VLM_BACKEND == "local":
        _vision_call_count += 1
        return extract_vision_local(fitz_page, pno)

    try:
        pix = fitz_page.get_pixmap(dpi=VLM_DPI)
        fd, png_path = tempfile.mkstemp(suffix=f"_p{pno}.png")
        os.close(fd)
        pix.save(png_path)
    except Exception as e:
        logger.warning(f"p{pno}: 이미지 렌더링 실패: {e}")
        return None

    try:
        prompt = _VLM_PROMPT.format(path=png_path)
        r = subprocess.run(
            [CLAUDE_BIN, "-p", prompt, "--allowedTools", "Read"],
            capture_output=True, text=True, timeout=VLM_TIMEOUT,
            encoding="utf-8",
        )
        _vision_call_count += 1
        out = (r.stdout or "").strip()
        if not out or out.split("\n", 1)[0].strip().upper().startswith("NO_TABLE"):
            return None
        md = _strip_fences(out)
        return md if "|" in md else None
    except FileNotFoundError:
        logger.error(f"claude CLI를 찾을 수 없음 (bin={CLAUDE_BIN}) — PATH 확인 필요")
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"p{pno}: claude CLI 타임아웃 ({VLM_TIMEOUT}s)")
        return None
    except Exception as e:
        logger.warning(f"p{pno}: claude CLI 실패: {e}")
        return None
    finally:
        try:
            os.remove(png_path)
        except OSError:
            pass


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
    import fitz

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

        # pdfplumber (선택)
        pp = extract_pdfplumber_tables(pdf_path, table_pages)
        if pp:
            table_sources["pdfplumber"] = pp

        # camelot (선택)
        cm = extract_camelot_tables(pdf_path, table_pages)
        if cm:
            table_sources["camelot"] = cm

        # VLM — PyMuPDF 표 탐지 페이지에만 실행
        if use_vision:
            total = len(table_pages)
            logger.info(f"VLM 대상: {total}페이지 (전체 {len(page_numbers)}페이지 중 PyMuPDF 표 탐지 페이지만, backend={VLM_BACKEND})")
            if VLM_BACKEND == "surya":
                vlm_tables = extract_surya_tables(pdf_path, table_pages)
            else:
                vlm_tables = {}
                for i, pno in enumerate(table_pages, 1):
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
