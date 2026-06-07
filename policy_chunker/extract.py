#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract.py — 약관 PDF에서 본문 텍스트 + 표를 뽑아 '결합 청크'로 만든다.

파이프라인의 첫 단계. 산출물(결합 청크)은 그대로 rechunk(run())의 입력이 된다.

  본문 : PyMuPDF로 페이지별 블록 추출(표 영역은 제외 → 본문/표 중복 방지)
  표   : 여러 도구로 뽑아 '베스트오브'(combine)로 합침
           - pymupdf    : 기본(항상)
           - pdfplumber : 설치돼 있으면 자동 사용
           - camelot    : 설치돼 있으면 자동 사용(ghostscript 필요)
           - VLM        : --vlm 시 Claude CLI로 페이지 이미지 → 마크다운 표
                          (API 키 불필요. 로컬 `claude` 실행 = 비전 추출)

설계 메모(파이프라인 QA에서 얻은 규칙):
  - 표는 페이지 단위로 한 청크(combine·rechunk이 페이지 기준으로 동작).
  - 표 영역 텍스트는 본문에서 빼서 중복을 막는다.
  - VLM은 표가 있는 페이지에만 돌린다(전 페이지에 비전 호출 = 느리고 낭비).
  - '베스트오브' 판정은 combine이 더블스페이스(셀 정렬 깨짐) 적은 쪽으로.

출력 청크 계약(= combine/rechunk이 받는 base chunk):
  chunk_id  : "{source}#p{page}#{seq:04d}"  (본문) / "{source}#p{page}#t{n}" (표)
  source, doc_type, page, text, is_table, chunk_type
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import fitz  # PyMuPDF

from .combine import combine


def _warn(msg: str) -> None:
    print(f"  ⚠️  {msg}", file=sys.stderr)

# 본문 유형 약한 분류(메타데이터용). 못 맞히면 general — rechunk이 무난히 처리.
_TYPE = [
    ("exclusion",  re.compile(r"지급하지\s*않는|면책|보상하지\s*않는|보상하지\s*아니")),
    ("payment",    re.compile(r"지급사유|보험금을\s*지급|지급할\s*보험금|보험금의\s*지급")),
    ("definition", re.compile(r"용어의\s*정의|이라\s*함은|라\s*합니다|말합니다")),
    ("coverage",   re.compile(r"보장|담보|보상하는")),
]


def classify(text: str) -> str:
    for name, rx in _TYPE:
        if rx.search(text):
            return name
    return "general"


# ───────────────────────── PyMuPDF: 본문 + 표 ─────────────────────────

def _overlap(a, b, tol: float = 1.0) -> bool:
    """두 bbox가 겹치는가(약간의 여유 tol)."""
    return not (a[2] <= b[0] + tol or b[2] <= a[0] + tol
                or a[3] <= b[1] + tol or b[3] <= a[1] + tol)


def extract_pymupdf(doc, source: str, doc_type: str):
    """본문 블록(표 영역 제외) + 페이지별 표 마크다운을 추출.

    Returns: (base_chunks, pymupdf_table_md_by_page)
    """
    base: list[dict] = []
    tbl_md: dict[int, str] = {}

    for pno in range(1, doc.page_count + 1):
        page = doc[pno - 1]

        # --- 표 탐지 ---
        try:
            found = page.find_tables()
            tables = list(found.tables) if found else []
        except Exception:
            tables = []
        tboxes, mds = [], []
        for t in tables:
            tboxes.append(t.bbox)
            try:
                md = t.to_markdown().strip()
            except Exception:
                md = ""
            if md:
                mds.append(md)
        if mds:
            tbl_md[pno] = "\n\n".join(mds)

        # --- 본문 블록(표 영역 제외) ---
        seq = 0
        for blk in page.get_text("blocks"):
            x0, y0, x1, y1, txt, _bno, btype = blk
            if btype != 0:                       # 0 = 텍스트 블록만
                continue
            txt = txt.strip()
            if not txt:
                continue
            if any(_overlap((x0, y0, x1, y1), tb) for tb in tboxes):
                continue                         # 표 영역 → 본문에서 제외
            seq += 1
            base.append({
                "chunk_id": f"{source}#p{pno}#{seq:04d}",
                "source": source, "doc_type": doc_type, "page": pno,
                "text": txt, "is_table": False, "chunk_type": classify(txt),
            })

        # --- 표 자리표시 청크(페이지당 1개). combine이 베스트오브로 교체 ---
        if pno in tbl_md:
            base.append({
                "chunk_id": f"{source}#p{pno}#t1",
                "source": source, "doc_type": doc_type, "page": pno,
                "text": tbl_md[pno], "is_table": True, "chunk_type": "table",
            })

    return base, tbl_md


# ───────────────────────── pdfplumber (선택) ─────────────────────────

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
        _warn("pdfplumber 미설치 → 이 도구는 건너뜀 (pip install pdfplumber)")
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
        return out
    return out


# ───────────────────────── camelot (선택) ─────────────────────────

def extract_camelot_tables(pdf_path: str, pages: list[int]) -> dict[int, str]:
    try:
        import camelot
    except ImportError:
        _warn("camelot 미설치 → 이 도구는 건너뜀 (pip install 'camelot-py[cv]')")
        return {}
    out: dict[int, str] = {}
    last_err = None
    for p in pages:
        for flavor in ("lattice", "stream"):
            try:
                tl = camelot.read_pdf(pdf_path, pages=str(p), flavor=flavor)
            except Exception as e:
                last_err = e
                continue
            if tl and tl.n:
                mds = [_rows_to_md(t.df.values.tolist()) for t in tl]
                mds = [m for m in mds if m]
                if mds:
                    out[p] = "\n\n".join(mds)
                    break
    if not out and last_err is not None:
        _warn(f"camelot이 표를 못 뽑음 — ghostscript 미설치일 수 있음 ({last_err})")
    return out


# ───────────────────────── VLM via Claude CLI ─────────────────────────

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


def render_page_png(doc, pno: int, dpi: int = 200) -> str:
    pix = doc[pno - 1].get_pixmap(dpi=dpi)
    fd, path = tempfile.mkstemp(suffix=f"_p{pno}.png")
    os.close(fd)
    pix.save(path)
    return path


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    return s.strip()


def vlm_table(png_path: str, claude_bin: str = "claude", timeout: int = 180):
    """페이지 PNG → Claude CLI(비전) → 마크다운 표. 표 없으면 None."""
    prompt = _VLM_PROMPT.format(path=png_path)
    try:
        r = subprocess.run(
            [claude_bin, "-p", prompt, "--allowedTools", "Read"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    out = (r.stdout or "").strip()
    if not out or out.split("\n", 1)[0].strip().upper().startswith("NO_TABLE"):
        return None
    md = _strip_fences(out)
    return md if "|" in md else None


def extract_vlm_tables(doc, pages: list[int], claude_bin: str = "claude",
                       dpi: int = 150, timeout: int = 180,
                       progress=None) -> dict[int, str]:
    out: dict[int, str] = {}
    for i, p in enumerate(pages, 1):
        if progress:
            progress(i, len(pages), p)
        png = render_page_png(doc, p, dpi)
        try:
            md = vlm_table(png, claude_bin, timeout)
        finally:
            try:
                os.remove(png)
            except OSError:
                pass
        if md:
            out[p] = md
    return out


# ───────────────────────── 진입점 ─────────────────────────

def extract(pdf_path: str, *, use_vlm: bool = False, claude_bin: str = "claude",
            dpi: int = 150, vlm_timeout: int = 180,
            prefer=("vlm", "pymupdf"), progress=None):
    """PDF → 결합 청크(본문 + 베스트오브 표). rechunk(run())의 입력.

    Returns: (combined_chunks, table_sources)
      table_sources: { 도구명 -> {page -> 마크다운} }  (진단/비교용)
    """
    doc = fitz.open(pdf_path)
    source = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_type = "약관"

    base, pymupdf_tbl = extract_pymupdf(doc, source, doc_type)
    table_pages = sorted(pymupdf_tbl.keys())   # pymupdf가 표로 판정한 페이지

    table_sources: dict[str, dict[int, str]] = {"pymupdf": pymupdf_tbl}
    pp = extract_pdfplumber_tables(pdf_path, table_pages)
    if pp:
        table_sources["pdfplumber"] = pp
    cm = extract_camelot_tables(pdf_path, table_pages)
    if cm:
        table_sources["camelot"] = cm
    if use_vlm:
        table_sources["vlm"] = extract_vlm_tables(
            doc, table_pages, claude_bin, dpi, vlm_timeout, progress)

    combined = combine(base, table_sources, prefer=prefer)
    return combined, table_sources
