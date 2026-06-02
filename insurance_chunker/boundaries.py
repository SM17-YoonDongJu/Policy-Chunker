#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundaries.py — PDF의 '시각적 신호'로 약관/별표 경계를 잡는다.

핵심 교훈:
    구조 경계는 본문 텍스트에서 추정하지 말고, PDF 조판 신호(제목 폰트 크기)에서
    직접 가져온다. 텍스트 추정은 제목이 줄바꿈/병합된 구간에서 무너진다.

출처: Policy-Chunker (팀 채택 코드) — 수정 없이 통합.
"""
from __future__ import annotations

import re
import collections
from dataclasses import dataclass

import fitz  # PyMuPDF


TITLE_SUFFIX = re.compile(r"(특별약관|보통약관|특약|담보)\s*$")
DIVIDER = re.compile(r"^\d+\.\s")
BYEOLPYO = re.compile(r"^【\s*(?:별표|부표)\s*([0-9\-]+)\s*】\s*(.*)")
BASE_START = re.compile(r"^제\s*1\s*조\s*\(")
TOC = re.compile(r"[·.]{6,}|…{3,}")


@dataclass
class Boundary:
    page: int      # 1-based
    label: str
    kind: str      # "base" | "yak" | "byeolpyo"


@dataclass
class Detection:
    """문서마다 자동 측정된 값."""
    body_size: float
    title_size: float
    title_lo: float
    title_hi: float
    base_start_page: int
    front_end_page: int
    product: str | None


def _lines(doc):
    """(page_1based, max_size, text) 단위로 페이지의 모든 줄을 순회."""
    for pno in range(1, doc.page_count + 1):
        for b in doc[pno - 1].get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                sp = [s for s in ln["spans"] if s["text"].strip()]
                if not sp:
                    continue
                sz = max(s["size"] for s in sp)
                txt = "".join(s["text"] for s in sp).strip()
                yield pno, round(sz, 1), txt


def detect(doc) -> Detection:
    """제목 폰트·보통약관 시작·표지 끝·상품명을 측정한다."""
    size_chars = collections.Counter()
    suffix_sizes = collections.Counter()
    base_start_page = None
    product = None

    for pno, sz, txt in _lines(doc):
        size_chars[sz] += len(txt)
        if TITLE_SUFFIX.search(txt) and len(txt) < 45 and not DIVIDER.match(txt):
            suffix_sizes[sz] += 1
        if base_start_page is None and BASE_START.match(txt) and not TOC.search(txt):
            base_start_page = pno
        if product is None and pno <= 2 and txt.endswith("보험") and 2 <= len(txt) <= 30:
            product = txt

    body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0

    cand = {s: n for s, n in suffix_sizes.items() if s > body_size + 0.3}
    if cand:
        title_size = max(cand, key=cand.get)
    else:
        bigger = {s: n for s, n in size_chars.items() if s > body_size + 1.0}
        title_size = max(bigger, key=bigger.get) if bigger else body_size + 2.9

    title_lo = round(title_size - 0.4, 1)
    title_hi = round(title_size + 0.3, 1)

    if base_start_page is None:
        base_start_page = 16
    front_end_page = base_start_page - 1

    return Detection(
        body_size=body_size, title_size=title_size,
        title_lo=title_lo, title_hi=title_hi,
        base_start_page=base_start_page, front_end_page=front_end_page,
        product=product,
    )


def _is_divider(txt: str, product: str | None) -> bool:
    if DIVIDER.match(txt) or txt.endswith("담보"):
        return True
    if product and txt == f"{product} 특별약관":
        return True
    if re.match(r"^.{2,30}\s특별약관$", txt) and "제" not in txt and len(txt) < 25:
        return txt.endswith("보험 특별약관")
    return False


def find(doc, det: Detection | None = None) -> tuple[list[Boundary], Detection]:
    """제목 폰트·별표 헤더로 경계 목록을 만든다."""
    if det is None:
        det = detect(doc)

    bounds: list[Boundary] = []
    for pno, sz, txt in _lines(doc):
        if det.title_lo <= sz <= det.title_hi:
            if pno <= det.front_end_page or _is_divider(txt, det.product):
                continue
            bounds.append(Boundary(pno, txt, "yak"))
        else:
            m = BYEOLPYO.match(txt)
            if m and len(txt) < 45:
                lab = re.sub(r"\s+", " ", f"별표{m.group(1)} {m.group(2)}").strip()
                bounds.append(Boundary(pno, lab, "byeolpyo"))

    base_label = f"{det.product} 보통약관" if det.product else "보통약관"
    bounds.append(Boundary(det.base_start_page, base_label, "base"))
    bounds.sort(key=lambda b: (b.page, b.kind))
    return bounds, det


def label_for(bounds: list[Boundary], page: int) -> tuple[str | None, str]:
    """페이지가 속한 (label, kind). 첫 경계 이전은 (None, 'front')."""
    cur = (None, "front")
    for b in bounds:
        if b.page <= page:
            cur = (b.label, b.kind)
        else:
            break
    return cur
