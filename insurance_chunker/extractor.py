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


def extract_vision(fitz_page, pno: int) -> Optional[str]:
    """fitz page → claude CLI(비전) → markdown 표 문자열. 표 없으면 None."""
    global _vision_call_count
    if _vision_call_count >= VISION_MAX_PAGES:
        logger.warning(f"Vision 상한({VISION_MAX_PAGES}회) 도달 — p{pno} 건너뜀")
        return None

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
            logger.info(f"VLM 대상: {total}페이지 (전체 {len(page_numbers)}페이지 중 PyMuPDF 표 탐지 페이지만)")
            vlm_tables: dict[int, str] = {}
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
