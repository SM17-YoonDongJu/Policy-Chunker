"""PDF 파싱: pdfplumber 텍스트 + PaddleOCR(스캔본).

표 추출은 extractor.py / combine.py 에서 별도 처리.
출처: rag/pipeline/pdf_parser.py 기반, 표 추출 분리.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pdfplumber

from .models import PageResult

logger = logging.getLogger(__name__)

_SCAN_DENSITY_THRESHOLD = 0.05
_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance
    try:
        from paddleocr import PaddleOCR  # type: ignore
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang="korean", use_gpu=True, show_log=False)
        logger.info("PaddleOCR 초기화 완료")
        return _ocr_instance
    except ImportError:
        logger.warning("PaddleOCR 미설치 — 스캔 페이지 OCR 불가")
        return None
    except Exception as e:
        logger.warning(f"PaddleOCR 초기화 실패: {e}")
        return None


def _text_density(text: str, page_area: float) -> float:
    expected_chars = page_area / (8 * 12)
    return len(text.replace(" ", "").replace("\n", "")) / max(expected_chars, 1)


def _ocr_page(page) -> str:
    ocr = _get_ocr()
    if ocr is None:
        return ""
    try:
        import numpy as np
        img = page.to_image(resolution=200).original
        result = ocr.ocr(np.array(img), cls=True)
        if result and result[0]:
            return "\n".join(line[1][0] for line in result[0])
    except Exception as e:
        logger.warning(f"OCR 실패 (p{page.page_number}): {e}")
    return ""


def parse_text(
    pdf_path: str,
    use_ocr: bool = True,
) -> tuple[list[PageResult], list]:
    """PDF 텍스트만 파싱. 표는 포함하지 않음.

    Returns:
        (pages, pdfplumber_pages_raw)
        pdfplumber_pages_raw: extractor.py가 이미지 렌더링에 재사용할 수 있도록 반환.
    """
    path = str(Path(pdf_path).resolve())
    results: list[PageResult] = []
    raw_pages = []

    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        logger.info(f"PDF 파싱 시작: {Path(pdf_path).name} ({total}페이지)")

        for page in pdf.pages:
            raw_pages.append(page)
            text = page.extract_text() or ""
            area = page.width * page.height
            is_ocr = False

            if _text_density(text, area) < _SCAN_DENSITY_THRESHOLD and use_ocr:
                logger.debug(f"  p{page.page_number}: 스캔본 → OCR")
                text = _ocr_page(page)
                is_ocr = True

            results.append(PageResult(
                page_num=page.page_number,
                text=text,
                tables=[],   # extractor.py에서 채움
                is_ocr=is_ocr,
            ))

        scanned = sum(1 for r in results if r.is_ocr)
        logger.info(f"텍스트 파싱 완료: {total}페이지 | OCR: {scanned}페이지")

    return results, raw_pages


def parse_pdf(
    pdf_path: str,
    use_ocr: bool = True,
    anthropic_client=None,
) -> list[PageResult]:
    """전체 파싱: 텍스트 + 표(PyMuPDF+Vision → best-of). 최종 PageResult 반환."""
    from .extractor import extract_tables_for_doc
    from .combine import select_best_tables

    pages, raw_pages = parse_text(pdf_path, use_ocr=use_ocr)

    pymupdf_tables, vision_tables = extract_tables_for_doc(
        pdf_path, raw_pages, anthropic_client=anthropic_client
    )
    best = select_best_tables(pymupdf_tables, vision_tables)

    for page in pages:
        entry = best.get(page.page_num)
        if entry:
            md, src = entry
            page.tables = [{"markdown": md, "source": src}]

    table_pages = sum(1 for p in pages if p.tables)
    logger.info(f"표 부착 완료: {table_pages}페이지 (best-of)")
    return pages
