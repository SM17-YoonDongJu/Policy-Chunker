#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine.py — 본문 청크 + '베스트오브' 표를 하나의 결합 청크로 만든다.

배경(파이프라인 QA에서 얻은 규칙):
  같은 페이지의 표를 여러 도구로 뽑으면 품질이 제각각이다.
    - camelot     : 더블스페이스 폭증(셀 정렬 깨짐)
    - pdfplumber  : 본문을 표로 오인
    - VLM / pymupdf: 대체로 양호 → 이 둘 중 더 깨끗한 쪽 채택
  '더 깨끗함'의 대용 지표로 **연속 공백(더블스페이스) 개수**를 쓴다(적을수록 좋음).

이 모듈은 표 추출 자체는 하지 않는다(도구는 외부에서 실행).
입력 계약:
  base_chunks : 본문/표가 섞인 원본 청크 리스트 (각 chunk_id에 #pN# 페이지 태그)
  table_sources : { "vlm": {page:int -> md:str, ...}, "pymupdf": {...}, ... }
출력:
  원본의 표 청크(is_table=True)를 페이지별 베스트오브 표로 교체한 결합 청크.
"""
from __future__ import annotations

import re


def _double_spaces(md: str) -> int:
    return len(re.findall(r"  +", md))


def best_table(page: int, table_sources: dict[str, dict[int, str]],
               prefer=("vlm", "pymupdf")) -> tuple[str | None, str | None]:
    """페이지의 표 중 더블스페이스가 가장 적은 것을 고른다. (md, source) 반환."""
    cands = []
    order = list(prefer) + [s for s in table_sources if s not in prefer]
    for src in order:
        md = table_sources.get(src, {}).get(page)
        if md and md.strip():
            cands.append((_double_spaces(md), src, md))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])  # 더블스페이스 적은 순
    _, src, md = cands[0]
    return md, src


def combine(base_chunks: list[dict],
            table_sources: dict[str, dict[int, str]],
            prefer=("vlm", "pymupdf")) -> list[dict]:
    """본문 청크는 유지, 표 청크는 페이지별 베스트오브로 교체."""
    out = []
    seen_table_pages = set()
    for c in base_chunks:
        if not c.get("is_table"):
            out.append(c)
            continue
        m = re.search(r"#p(\d+)#", c["chunk_id"])
        page = int(m.group(1)) if m else c.get("page", 0)
        if page in seen_table_pages:
            continue  # 같은 페이지 표는 한 번만 교체
        md, src = best_table(page, table_sources, prefer)
        if md is None:
            out.append(c)  # 대체할 표가 없으면 원본 유지
            continue
        seen_table_pages.add(page)
        tc = dict(c)
        tc["chunk_id"] = f"{c['source']}#p{page}#t1"
        tc["text"] = md.strip()
        tc["is_table"] = True
        tc["table_source"] = src
        out.append(tc)
    return out
