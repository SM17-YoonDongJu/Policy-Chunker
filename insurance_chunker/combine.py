"""combine.py — 페이지별로 PyMuPDF vs VLM 중 더 깨끗한 표를 고른다.

품질 지표: 연속 공백(더블스페이스) 수 — 적을수록 셀 정렬이 깨끗함.
출처: Policy-Chunker (팀 채택 코드).
"""
from __future__ import annotations

import re


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
