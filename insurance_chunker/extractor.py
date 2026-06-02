"""표 추출: PyMuPDF + Claude Vision 두 소스.

combine.py가 페이지별로 더 깨끗한 쪽을 선택한다.
  PyMuPDF : fitz.find_tables() → markdown (괘선 있는 표에 강함, 빠름)
  Vision  : Claude가 이미지 → markdown (괘선 없는 표, 복잡 레이아웃에 강함)
"""
from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)

VISION_MODEL = os.environ.get("VISION_MODEL", "claude-sonnet-4-6")
VISION_MAX_PAGES = int(os.environ.get("VISION_MAX_PAGES", "9999"))

_vision_call_count = 0


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


# ── Claude Vision ─────────────────────────────────────────────────────────────

def extract_vision(pdfplumber_page, anthropic_client) -> Optional[str]:
    """pdfplumber page 이미지 → Claude Vision → markdown 표 문자열."""
    global _vision_call_count
    if anthropic_client is None:
        return None
    if _vision_call_count >= VISION_MAX_PAGES:
        logger.warning(f"Vision 상한({VISION_MAX_PAGES}회) 도달 — p{pdfplumber_page.page_number} 건너뜀")
        return None

    try:
        img = pdfplumber_page.to_image(resolution=150).original
    except Exception as e:
        logger.warning(f"p{pdfplumber_page.page_number}: 이미지 변환 실패: {e}")
        return None

    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        message = anthropic_client.messages.create(
            model=VISION_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "이 보험 문서 이미지에서 모든 표(table)를 마크다운으로 추출하세요.\n"
                            "표가 없으면 빈 문자열만 반환하세요. 마크다운 표 외 텍스트는 출력하지 마세요."
                        ),
                    },
                ],
            }],
        )
        _vision_call_count += 1
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1]).strip()
        return text if text else None
    except Exception as e:
        logger.warning(f"p{pdfplumber_page.page_number}: Vision 실패 (model={VISION_MODEL}): {e}")
        return None


# ── 문서 단위 추출 ────────────────────────────────────────────────────────────

def extract_tables_for_doc(
    pdf_path: str,
    pdfplumber_pages: list,
    anthropic_client=None,
) -> tuple[dict[int, str], dict[int, str]]:
    """전체 문서의 페이지별 표를 두 소스로 추출.

    Returns:
        (pymupdf_tables, vision_tables)
        각 dict: {page_1based: markdown_str}
    """
    import fitz

    pymupdf_tables: dict[int, str] = {}
    vision_tables: dict[int, str] = {}

    reset_vision_counter()

    with fitz.open(pdf_path) as doc:
        for pl_page in pdfplumber_pages:
            pno = pl_page.page_number  # 1-based
            fitz_page = doc[pno - 1]

            md_pymupdf = extract_pymupdf(fitz_page)
            if md_pymupdf:
                pymupdf_tables[pno] = md_pymupdf

            md_vision = extract_vision(pl_page, anthropic_client)
            if md_vision:
                vision_tables[pno] = md_vision

    logger.info(
        f"표 추출 완료: PyMuPDF {len(pymupdf_tables)}페이지 / Vision {len(vision_tables)}페이지"
    )
    return pymupdf_tables, vision_tables
