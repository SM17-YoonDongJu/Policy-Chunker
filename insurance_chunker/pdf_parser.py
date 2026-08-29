"""PDF 파싱: PyMuPDF 텍스트(표 영역 제외).

Policy-Chunker(main) extract.py와 동일한 텍스트 추출 방식:
  - PyMuPDF get_text("blocks") 로 텍스트 블록 추출
  - 표 bbox와 겹치는 블록은 제외 (본문·표 중복 방지)

스캔본(이미지 PDF)은 지원하지 않는다. 폰트 신호가 없어 경계 감지가
실패하므로 OCR 전처리 후 텍스트 PDF로 변환해서 입력해야 한다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pymupdf as fitz  # 'fitz'는 구 이름 — 그대로 쓰면 임포트마다 경고가 찍힌다

from .models import PageResult

logger = logging.getLogger(__name__)


def _center_in(line_bbox, table_bbox) -> bool:
    """줄 중심점이 표 bbox 안인가 — 블록 단위 any-overlap 제외는 표에 인접한
    조 헤딩 줄까지 블록째 삼켜 텍스트/표 양쪽에서 유실시켰다(프리미엄 p332
    제9조). 줄 중심점 기준이면 경계에 걸친 헤딩은 살고 표 내용은 빠진다."""
    cx = (line_bbox[0] + line_bbox[2]) / 2
    cy = (line_bbox[1] + line_bbox[3]) / 2
    return table_bbox[0] <= cx <= table_bbox[2] and table_bbox[1] <= cy <= table_bbox[3]


def parse_text(
    pdf_path: str,
    use_ocr: bool = True,  # 하위 호환성 유지, 현재 미사용
) -> tuple[list[PageResult], list[int]]:
    """PyMuPDF get_text("blocks")로 텍스트 추출. 표 영역 제외.

    Returns:
        (pages, page_numbers)
        page_numbers: 1-based 페이지 번호 목록. extractor.py에 전달.
    """
    path = str(Path(pdf_path).resolve())
    results: list[PageResult] = []
    page_numbers: list[int] = []

    with fitz.open(path) as doc:
        total = doc.page_count
        logger.info(f"PDF 파싱 시작: {Path(pdf_path).name} ({total}페이지)")

        for pno in range(1, total + 1):
            page = doc[pno - 1]
            page_numbers.append(pno)

            # 표 bbox 탐지 (표 영역 텍스트 제외용)
            try:
                found = page.find_tables()
                tboxes = [t.bbox for t in (found.tables if found else [])]
            except Exception:
                tboxes = []

            # 텍스트 추출 — 표 영역 제외는 줄 단위로 (블록 단위는 헤딩 유실)
            blocks_text: list[str] = []
            for blk in page.get_text("dict")["blocks"]:
                if blk.get("type", 1) != 0:  # 텍스트 블록만
                    continue
                keep: list[str] = []
                for ln in blk.get("lines", []):
                    txt = "".join(s["text"] for s in ln["spans"]).strip()
                    if not txt:
                        continue
                    if any(_center_in(ln["bbox"], tb) for tb in tboxes):
                        continue
                    keep.append(txt)
                if keep:
                    blocks_text.append("\n".join(keep))

            results.append(PageResult(
                page_num=pno,
                text="\n".join(blocks_text),
                tables=[],
                is_ocr=False,
            ))

        logger.info(f"텍스트 파싱 완료: {total}페이지")

    return results, page_numbers


def parse_pdf(
    pdf_path: str,
    use_ocr: bool = True,
    use_vision: bool = True,
) -> list[PageResult]:
    """전체 파싱: 텍스트 + 표(다중 소스 best-of). 최종 PageResult 반환."""
    from .combine import select_best_tables
    from .extractor import extract_tables_for_doc

    pages, page_numbers = parse_text(pdf_path, use_ocr=use_ocr)

    table_sources = extract_tables_for_doc(
        pdf_path, page_numbers, use_vision=use_vision
    )
    best = select_best_tables(table_sources)

    for page in pages:
        entry = best.get(page.page_num)
        if entry:
            md, src = entry
            page.tables = [{"markdown": md, "source": src}]

    table_pages = sum(1 for p in pages if p.tables)
    logger.info(f"표 부착 완료: {table_pages}페이지 (best-of)")
    return pages
