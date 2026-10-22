"""combine.py — 페이지별 표 선택.

레이아웃 경로(기본): 박스가 있는 쪽은 VLM(crop) 결과, 비었으면 PyMuPDF 폴백.
레이아웃 off: 더블스페이스(연속 공백) 수가 적은 소스 — 예전 best-of 그대로.
"""
from __future__ import annotations

import re


def select_layout_tables(
    pymupdf_tables: dict[int, str],
    vlm_tables: dict[int, str],
) -> dict[int, tuple[str, str]]:
    """레이아웃 crop 경로의 표 선택 — VLM 결과 우선, 비면 PyMuPDF.

    더블스페이스 비교를 쓰지 않는 이유: crop 입력에서 두 소스는 같은 쪽의 다른 산출물이
    아니라 '레이아웃이 가리킨 영역'과 '괘선 휴리스틱'이라, 깨끗한 쪽이 아니라 레이아웃
    쪽을 믿는 게 이 전환의 전제다(Colab 실험 — Paddle crop이 PyMuPDF 표보다 구조 정확).
    VLM이 빈 쪽(퇴행 폐기·차단기·상한 초과)은 PyMuPDF가 폴백이다.
    """
    result: dict[int, tuple[str, str]] = {}
    for page in set(pymupdf_tables) | set(vlm_tables):
        md = vlm_tables.get(page)
        if md and md.strip():
            result[page] = (md, "vlm")
            continue
        md = pymupdf_tables.get(page)
        if md and md.strip():
            result[page] = (md, "pymupdf")
    return result


def _double_spaces(md: str) -> int:
    return len(re.findall(r"  +", md))


def best_table(
    page: int,
    table_sources: dict[str, dict[int, str]],
    prefer: tuple[str, ...] = ("vlm", "pymupdf"),
) -> tuple[str | None, str | None]:
    """페이지의 소스들 중 더블스페이스가 적은 쪽 반환. (markdown, source)"""
    cands = []
    order = list(prefer) + [s for s in table_sources if s not in prefer]
    for src in order:
        md = table_sources.get(src, {}).get(page)
        if md and md.strip():
            cands.append((_double_spaces(md), src, md))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])
    _, src, md = cands[0]
    return md, src


def select_best_tables(
    table_sources: dict[str, dict[int, str]],
    prefer: tuple[str, ...] = ("vlm", "pymupdf"),
) -> dict[int, tuple[str, str]]:
    """전체 문서의 페이지별 베스트 표 선택.

    Returns:
        {page: (markdown, source)}
    """
    all_pages: set[int] = set()
    for pages in table_sources.values():
        all_pages |= set(pages)
    result: dict[int, tuple[str, str]] = {}
    for page in all_pages:
        md, src = best_table(page, table_sources, prefer)
        if md:
            result[page] = (md, src)
    return result
