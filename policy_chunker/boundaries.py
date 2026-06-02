#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundaries.py — PDF의 '시각적 신호'로 약관/별표 경계를 잡는다.

핵심 교훈(v3 실패 → v4 전환):
    구조 경계는 본문 텍스트에서 추정하지 말고, PDF 조판 신호(제목 폰트 크기)에서
    직접 가져온다. 텍스트 추정은 제목이 줄바꿈/병합된 구간에서 무너진다.

v4.2는 제목 폰트(12.9)·보통약관 시작 페이지(16)를 하드코딩했다.
여기서는 그 두 값을 문서마다 '자동 측정'한다 → 다른 약관에도 그대로 동작.
"""
from __future__ import annotations

import re
import collections
from dataclasses import dataclass

import fitz  # PyMuPDF


# 약관 제목으로 흔히 끝나는 접미사 (제목 폰트 후보 판별용)
TITLE_SUFFIX = re.compile(r"(특별약관|보통약관|특약|담보)\s*$")
# "1. 상해위험담보" 같은 담보 분류 divider / 우산 제목(상품명 + 특별약관)
DIVIDER = re.compile(r"^\d+\.\s")
# 별표/부표 헤더
BYEOLPYO = re.compile(r"^【\s*(?:별표|부표)\s*([0-9\-]+)\s*】\s*(.*)")
# 보통약관 본문 시작 신호: 목차가 아닌 '제1조(목적)' 또는 '제1조(보험계약...)'
BASE_START = re.compile(r"^제\s*1\s*조\s*\(")
TOC = re.compile(r"[·.]{6,}|…{3,}")


@dataclass
class Boundary:
    page: int      # 1-based
    label: str
    kind: str      # "base" | "yak" | "byeolpyo"


@dataclass
class Detection:
    """문서마다 자동 측정된 값(추정이 아니라 측정)."""
    body_size: float          # 본문 최빈 글자 크기
    title_size: float         # 특약 제목 글자 크기
    title_lo: float           # 제목으로 인정할 크기 하한
    title_hi: float           # 제목으로 인정할 크기 상한
    base_start_page: int      # 보통약관 본문 시작 페이지
    front_end_page: int       # 표지/목차 끝 페이지
    product: str | None       # 표지에서 추출한 상품명(있으면)


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
    size_chars = collections.Counter()         # 크기 → 글자수(본문 최빈 추정)
    suffix_sizes = collections.Counter()       # '특약/약관'으로 끝나는 줄의 크기
    base_start_page = None
    product = None

    for pno, sz, txt in _lines(doc):
        size_chars[sz] += len(txt)
        if TITLE_SUFFIX.search(txt) and len(txt) < 45 and not DIVIDER.match(txt):
            suffix_sizes[sz] += 1
        # 보통약관 본문 시작: 목차가 아닌 제1조(...)가 처음 나오는 페이지
        if base_start_page is None and BASE_START.match(txt) and not TOC.search(txt):
            base_start_page = pno
        # 상품명: 표지(1~2p)에서 '...보험'으로 끝나는 첫 큰 제목
        if product is None and pno <= 2 and txt.endswith("보험") and 2 <= len(txt) <= 30:
            product = txt

    # 본문 최빈 크기
    body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0

    # 제목 폰트 = 본문보다 크면서 '특약/약관'으로 끝나는 줄에 가장 많이 쓰인 크기
    cand = {s: n for s, n in suffix_sizes.items() if s > body_size + 0.3}
    if cand:
        title_size = max(cand, key=cand.get)
    else:
        # 폴백: 본문보다 한 단계 큰 크기 중 가장 흔한 것
        bigger = {s: n for s, n in size_chars.items() if s > body_size + 1.0}
        title_size = max(bigger, key=bigger.get) if bigger else body_size + 2.9

    # 제목 인정 범위(측정값 ± 여유). v4.2의 12.5~13.2(중심 12.85)와 동등한 폭.
    title_lo = round(title_size - 0.4, 1)
    title_hi = round(title_size + 0.3, 1)

    if base_start_page is None:
        base_start_page = 16  # 최후 폴백
    front_end_page = base_start_page - 1

    return Detection(
        body_size=body_size, title_size=title_size,
        title_lo=title_lo, title_hi=title_hi,
        base_start_page=base_start_page, front_end_page=front_end_page,
        product=product,
    )


def _is_divider(txt: str, product: str | None) -> bool:
    """담보 분류 divider / 상품명+특별약관 우산 제목 → 약관 제목이 아님."""
    if DIVIDER.match(txt) or txt.endswith("담보"):
        return True
    if product and txt == f"{product} 특별약관":
        return True
    # 상품명을 못 잡았을 때의 일반 패턴
    if re.match(r"^.{2,30}\s특별약관$", txt) and "제" not in txt and len(txt) < 25:
        # '○○보험 특별약관' 형태의 우산 제목 (개별 특약명은 보통 더 구체적/길다)
        return txt.endswith("보험 특별약관")
    return False


def find(doc, det: Detection | None = None) -> tuple[list[Boundary], Detection]:
    """제목 폰트·별표 헤더로 경계 목록을 만든다. (v4.2 로직의 적응형 버전)"""
    if det is None:
        det = detect(doc)

    bounds: list[Boundary] = []
    for pno, sz, txt in _lines(doc):
        if det.title_lo <= sz <= det.title_hi:
            if pno <= det.front_end_page or _is_divider(txt, det.product):
                continue  # 표지/목차/divider 제외
            bounds.append(Boundary(pno, txt, "yak"))
        else:
            m = BYEOLPYO.match(txt)
            if m and len(txt) < 45:
                lab = re.sub(r"\s+", " ", f"별표{m.group(1)} {m.group(2)}").strip()
                bounds.append(Boundary(pno, lab, "byeolpyo"))

    # 보통약관 경계 주입 (본문 시작 페이지)
    base_label = f"{det.product} 보통약관" if det.product else "보통약관"
    bounds.append(Boundary(det.base_start_page, base_label, "base"))
    bounds.sort(key=lambda b: (b.page, b.kind))
    return bounds, det


def label_for(bounds: list[Boundary], page: int) -> tuple[str | None, str]:
    """페이지가 속한 (label, kind). 첫 경계 이전은 ('front')."""
    cur = (None, "front")
    for b in bounds:
        if b.page <= page:
            cur = (b.label, b.kind)
        else:
            break
    return cur
