#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundaries.py — PDF의 '시각적 신호'로 약관/별표 경계를 잡는다.

핵심 교훈:
    구조 경계는 본문 텍스트에서 추정하지 말고, PDF 조판 신호(제목 폰트 크기)에서
    직접 가져온다. 텍스트 추정은 제목이 줄바꿈/병합된 구간에서 무너진다.

출처: Policy-Chunker(main) — assess() / None 반환 전략 포함.
"""
from __future__ import annotations

import re
import collections
from dataclasses import dataclass, field

import fitz  # PyMuPDF


TITLE_SUFFIX = re.compile(r"(특별약관|보통약관|특약|담보)\s*$")
# 제목 폰트 인정 임계: '특약/약관'으로 끝나는 줄이 이만큼은 있어야 한다
MIN_TITLE_EVIDENCE = 4
MIN_TITLE_GAP = 0.8
PRODUCT = re.compile(r".*보험[\sⅠⅡⅢⅣⅤIVX0-9]*$")
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
    """문서마다 자동 측정된 값.

    title_size / base_start_page가 None이면 '신호를 못 찾았다'는 뜻이다.
    예전엔 폴백값(폰트 한 단계 위, p16)을 조용히 채웠는데, 그게 틀린 라벨을
    양산했다. 이제는 못 찾으면 None으로 두고 assess()가 경고한다.
    """
    body_size: float
    title_size: float | None       # 못 찾으면 None
    title_lo: float | None
    title_hi: float | None
    base_start_page: int | None    # 못 찾으면 None
    front_end_page: int | None
    product: str | None
    title_evidence: int = 0        # 제목 폰트 판정 근거 줄 수


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
    """제목 폰트·보통약관 시작·표지 끝·상품명을 측정한다.

    찾지 못하면 폴백을 만들지 않고 None을 둔다(틀린 라벨 양산 방지).
    신뢰 여부는 assess()로 판정한다.
    """
    size_chars: collections.Counter = collections.Counter()
    suffix_sizes: collections.Counter = collections.Counter()
    base_start_page = None
    product_cands: list[tuple[float, int, str]] = []

    for pno, sz, txt in _lines(doc):
        size_chars[sz] += len(txt)
        if TITLE_SUFFIX.search(txt) and len(txt) < 45 and not DIVIDER.match(txt):
            suffix_sizes[sz] += 1
        if base_start_page is None and BASE_START.match(txt) and not TOC.search(txt):
            base_start_page = pno
        if pno <= 3 and 4 <= len(txt) <= 40 and PRODUCT.match(txt):
            product_cands.append((sz, len(txt), txt))

    body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0

    cand = {s: n for s, n in suffix_sizes.items() if s > body_size + MIN_TITLE_GAP}
    title_size = title_lo = title_hi = None
    title_evidence = 0
    if cand:
        best = max(cand, key=cand.get)
        title_evidence = cand[best]
        if title_evidence >= MIN_TITLE_EVIDENCE:
            title_size = best
            title_lo = round(best - 0.4, 1)
            title_hi = round(best + 0.3, 1)

    product = None
    if product_cands:
        product_cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
        product = product_cands[0][2].strip()

    front_end_page = (base_start_page - 1) if base_start_page else None

    return Detection(
        body_size=body_size, title_size=title_size,
        title_lo=title_lo, title_hi=title_hi,
        base_start_page=base_start_page, front_end_page=front_end_page,
        product=product, title_evidence=title_evidence,
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
    has_font = det.title_size is not None
    front_end = det.front_end_page or 0

    for pno, sz, txt in _lines(doc):
        if has_font and det.title_lo <= sz <= det.title_hi:
            if pno <= front_end or _is_divider(txt, det.product):
                continue
            bounds.append(Boundary(pno, txt, "yak"))
        else:
            m = BYEOLPYO.match(txt)
            if m and len(txt) < 45:
                lab = re.sub(r"\s+", " ", f"별표{m.group(1)} {m.group(2)}").strip()
                bounds.append(Boundary(pno, lab, "byeolpyo"))

    if det.base_start_page is not None:
        base_label = f"{det.product} 보통약관" if det.product else "보통약관"
        bounds.append(Boundary(det.base_start_page, base_label, "base"))
    bounds.sort(key=lambda b: (b.page, b.kind))
    return bounds, det


def assess(doc, det: Detection | None = None,
           bounds: list[Boundary] | None = None) -> tuple[str, list[str]]:
    """자동탐지 신뢰도 판정. ('ok' | 'weak', 사유 목록) 반환.

    'weak'이면 조문·특약 구조가 약해 자동 경계를 신뢰하기 어렵다는 뜻.
    조용히 틀린 결과를 내는 대신 호출부가 경고/거부 여부를 결정한다.
    """
    if det is None:
        bounds, det = find(doc)
    elif bounds is None:
        bounds, _ = find(doc, det)

    reasons: list[str] = []
    n_yak = sum(1 for b in bounds if b.kind == "yak")
    n_byeolpyo = sum(1 for b in bounds if b.kind == "byeolpyo")

    if det.title_size is None:
        reasons.append(
            f"제목 폰트 신호 없음 (특약/약관 제목 {det.title_evidence}줄 "
            f"< 임계 {MIN_TITLE_EVIDENCE}) → 특별약관 경계를 못 잡음")
    if det.base_start_page is None:
        reasons.append("보통약관 시작(제1조) 못 찾음 → 조문 구조가 없는 문서일 수 있음")
    if n_yak == 0:
        reasons.append("특별약관 경계 0개")
    if det.product is None:
        reasons.append("표지에서 상품명 못 잡음")

    level = "weak" if (det.title_size is None or det.base_start_page is None
                       or n_yak == 0) else "ok"
    if level == "ok" and not reasons:
        reasons.append(f"특약 {n_yak} / 별표 {n_byeolpyo} 경계, "
                       f"제목폰트 {det.title_size}(근거 {det.title_evidence}줄)")
    return level, reasons


def label_for(bounds: list[Boundary], page: int) -> tuple[str | None, str]:
    """페이지가 속한 (label, kind). 첫 경계 이전은 (None, 'front')."""
    cur = (None, "front")
    for b in bounds:
        if b.page <= page:
            cur = (b.label, b.kind)
        else:
            break
    return cur
