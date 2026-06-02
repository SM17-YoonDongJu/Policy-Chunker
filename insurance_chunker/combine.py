"""combine.py — 페이지별로 PyMuPDF vs Vision 중 더 깨끗한 표를 고른다.

품질 지표: 연속 공백(더블스페이스) 수 — 적을수록 셀 정렬이 깨끗함.
출처: Policy-Chunker (팀 채택 코드).
"""
from __future__ import annotations

import re


def _double_spaces(md: str) -> int:
    return len(re.findall(r"  +", md))


def best_table(
    page: int,
    pymupdf_tables: dict[int, str],
    vision_tables: dict[int, str],
    prefer: tuple[str, ...] = ("vision", "pymupdf"),
) -> tuple[str | None, str | None]:
    """페이지의 두 소스 중 더블스페이스가 적은 쪽 반환. (markdown, source)"""
    sources = {"pymupdf": pymupdf_tables, "vision": vision_tables}
    cands = []
    order = list(prefer) + [s for s in sources if s not in prefer]
    for src in order:
        md = sources[src].get(page)
        if md and md.strip():
            cands.append((_double_spaces(md), src, md))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])
    _, src, md = cands[0]
    return md, src


def select_best_tables(
    pymupdf_tables: dict[int, str],
    vision_tables: dict[int, str],
    prefer: tuple[str, ...] = ("vision", "pymupdf"),
) -> dict[int, tuple[str, str]]:
    """전체 문서의 페이지별 베스트 표 선택.

    Returns:
        {page: (markdown, source)}
    """
    all_pages = set(pymupdf_tables) | set(vision_tables)
    result: dict[int, tuple[str, str]] = {}
    for page in all_pages:
        md, src = best_table(page, pymupdf_tables, vision_tables, prefer)
        if md:
            result[page] = (md, src)
    return result
