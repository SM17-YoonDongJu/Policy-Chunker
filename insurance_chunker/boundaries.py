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


TITLE_SUFFIX = re.compile(r"(특별약관|보통약관|특약|담보)\s*[ⅠⅡⅢⅣⅤIVX0-9()]*\s*$")
# 제목 폰트 인정 임계: '특약/약관'으로 끝나는 줄이 이만큼은 있어야 한다
MIN_TITLE_EVIDENCE = 4
MIN_TITLE_GAP = 0.8
PRODUCT = re.compile(r".*보험")
# 보험료정산 추가특약 등, 상품명이 아닌 "보험" 포함 짧은 줄이 후보로 끼어드는 걸 막는
# 최소 근거는 두지 않고 대신 크기·길이로 우선순위를 매긴다(아래 product_cands 정렬).
GWAN = re.compile(r"^제\s*\d+\s*관\b")
DIVIDER = re.compile(r"^\d+\.\s")
BYEOLPYO = re.compile(r"^【\s*(?:별표|부표)\s*([0-9\-]+)\s*】\s*(.*)")

# ── 다중 신호 경계 점수제 ────────────────────────────────────────────────────
# 폰트 크기 하나에만 기대면 (a) 노이즈(목차·마케팅 문구)를 제목으로 오인하거나
# (b) 문서에 제목 폰트 신호가 아예 없을 때 경계 감지가 통째로 죽는다.
# 여러 신호를 점수로 합산해 완충하고, 폰트는 그중 하나로 낮춘다.
_DOT_LEADER = re.compile(r"[·.]{4,}|…{2,}")
_OTHER_LAW = re.compile(r"(법률|법|규칙|시행령|고시)\s*$")
_PURE_SYMBOLIC = re.compile(r"^[\d.\-\s※▪◉]+$")
# "제N조(...)"는 조 단위 제목 — 특약(yak) 경계와는 다른 계층이라 rechunk.py의
# ART 정규식이 별도로 처리한다. 여기서 후보로 안 잡아야 짧고 굵은 조 제목이
# 전부 특약 경계로 오인되는 대량 오탐(문서당 수백 건)을 막을 수 있다.
_ART_LIKE = re.compile(r"^제\s*\d+\s*조(?:의\s*\d+)?\s*[\(（]")
# <용어풀이> [절차 안내] 같은 본문 내 콜아웃 박스 — 별표(【】)와 다른 괄호 기호라
# BYEOLPYO에 안 걸리고 짧고 굵게 조판돼 특약 제목처럼 보이지만 특약 경계는 아니다.
_CALLOUT = re.compile(r"^[<\[].{1,30}[>\]]$")
# 본문 문장이 다른 특약명을 인용하며 끝나는 경우("...다만 A특약 또는 B특별약관")
# — 접미사 매치는 통과하지만 문장 내부 어미·접속사가 섞여 있으면 제목이 아니라
# 인용 조각이다. 진짜 제목엔 이런 문장 성분이 없다.
_MID_SENTENCE = re.compile(r"(다만|습니다|합니다|또는|정하지|않은|않는|①|②|③|④|⑤)")
# 조판상 앞 특약의 마지막 문장("…따릅니다.")과 다음 특약 제목이 한 물리 줄로 붙어
# 나오는 문서(표 형태 단체약관)가 있다 — 마지막 문장 종결 뒤 꼬리만 제목 후보로 쓴다.
_SENT_TAIL = re.compile(r"^.*(?:니다|한다|된다)\.\s+(\S.*)$")
# 접미사 단독 줄("특별약관")은 표지/장 구분 페이지의 장식 — 제목이 아니다.
# "약관"은 3줄 감김의 마지막 꼬리("…보장 특별" + "약관")로도 나타난다.
_BARE_SUFFIX = re.compile(r"^(특별약관|보통약관|특약|담보|약관)$")
# 로마숫자 장 구분("Ⅰ. 상해 관련 특별약관")은 특약 묶음 헤더 — 경계로 만들면
# 뒤따르는 첫 특약과 라벨이 충돌한다.
_ROMAN_CH = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s")
# 번호 구분선("5. 기타 특별약관")과 번호 특약 제목("13. …(355간편가입)보장 특별약관")
# 구분: 구분선은 괄호 한정어가 없고 짧거나 묶음 명사(기타/관련/…)로 이뤄진다.
# 폰트로는 못 가른다(30327은 구분선과 특약 제목이 같은 12.9pt).
_GENERIC_GROUP = re.compile(r"기타|관련|담보\s*등|및\s")
# 【별표N】표기 없이 제목 폰트로 조판된 부록 표 제목(산정특례대상질병 분류표 등)
_TABLE_TITLE = re.compile(r"^.{2,40}(분류표|등급표|급표)\s*$")
# 약관 뒤에 붙는 부록 헤더 — 이 아래는 타 법령 전문이 수십 페이지 이어져서
# 법령 조번호(의료법 제36조 등)가 직전 특약의 조로 오인된다(실손류 문서).
_APPENDIX_HEAD = re.compile(
    r"^(?:약관에서\s*)?인용(?:된)?\s*법[\W\s]*규정\s*$"
    r"|^(?:가나다\s*순?\s*)?특별약관\s*색인\s*$"
    r"|^약관\s*요약서\s*$")

_W_SUFFIX, _W_FONT, _W_BOLD, _W_ISOLATED, _W_SHORT = 1.6, 1.4, 1.0, 0.8, 0.6
_W_DOT_LEADER, _W_OTHER_LAW = 2.0, 1.5
SCORE_THRESHOLD = 2.4


_SENT_END = re.compile(r"[다요]\s*\.\s*$")  # 완결 문장 — 제목은 '다.'로 끝나지 않는다


def _invalid_candidate(txt: str) -> bool:
    return (not (2 <= len(txt) <= 90) or bool(_PURE_SYMBOLIC.match(txt))
            or bool(_ART_LIKE.match(txt)) or bool(_CALLOUT.match(txt))
            or bool(_MID_SENTENCE.search(txt)) or bool(_BARE_SUFFIX.match(txt))
            or bool(_SENT_END.search(txt)))


def _signal_score(txt: str, has_suffix: bool, sz: float, is_bold: bool,
                  is_isolated: bool, det: "Detection") -> float:
    s = 0.0
    if has_suffix:
        s += _W_SUFFIX
    if det.title_size is not None:
        ratio = max(0.0, sz - det.body_size) / max(0.1, det.title_size - det.body_size)
        s += _W_FONT * min(1.0, ratio)
    if is_bold:
        s += _W_BOLD
    if is_isolated:
        s += _W_ISOLATED
    s += _W_SHORT * (1.0 if len(txt) < 25 else max(0.0, (60 - len(txt)) / 35))
    if _DOT_LEADER.search(txt):
        s -= _W_DOT_LEADER
    if _OTHER_LAW.search(txt) and not has_suffix:
        s -= _W_OTHER_LAW
    return s


def _score_candidate(txt: str, sz: float, is_bold: bool, is_isolated: bool,
                      det: "Detection") -> float:
    if _invalid_candidate(txt) or len(txt) > 60:
        return 0.0
    return _signal_score(txt, bool(TITLE_SUFFIX.search(txt)), sz, is_bold, is_isolated, det)


def _try_title(parts: list[tuple[str, float, bool, bool]],
               det: "Detection") -> str | None:
    """pending 줄 묶음을 하나의 제목으로 판정. 미완성(접미사 미도달)이면 None.

    접미사 검사는 공백 제거본으로 한다 — 줄바꿈이 '특별'/'약관' 사이를 가르면
    공백 포함 병합문에서는 접미사가 안 잡힌다.
    """
    joined = re.sub(r"\s+", " ", " ".join(p[0] for p in parts)).strip()
    dense = joined.replace(" ", "")
    if not TITLE_SUFFIX.search(dense):
        return None
    if _invalid_candidate(joined):
        return None
    sz = max(p[1] for p in parts)
    is_bold = any(p[2] for p in parts)
    is_isolated = any(p[3] for p in parts)
    s = _signal_score(joined, True, sz, is_bold, is_isolated, det)
    return joined if s >= SCORE_THRESHOLD else None
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


_BOLD_FLAG = 1 << 4  # PyMuPDF span flags 비트4 = 굵게


def _lines(doc):
    """(page_1based, max_size, text) 단위로 페이지의 모든 줄을 순회. (하위호환)"""
    for pno, sz, txt, _bold, _isolated in _lines_rich(doc):
        yield pno, sz, txt


def _lines_rich(doc):
    """(page, size, text, is_bold, is_isolated) — 다중신호 점수제 입력.

    is_isolated: 이 줄이 속한 블록이 한 줄짜리인가(제목은 보통 단독 블록으로
    조판됨, 본문 단락은 여러 줄이 한 블록에 뭉침) — 빈 줄 위아래 신호의 근사치.
    """
    for pno in range(1, doc.page_count + 1):
        for b in doc[pno - 1].get_text("dict")["blocks"]:
            block_lines = b.get("lines", [])
            is_isolated = len(block_lines) == 1
            for ln in block_lines:
                sp = [s for s in ln["spans"] if s["text"].strip()]
                if not sp:
                    continue
                sz = max(s["size"] for s in sp)
                txt = "".join(s["text"] for s in sp).strip()
                is_bold = any(s["flags"] & _BOLD_FLAG for s in sp) or any(
                    "bold" in s.get("font", "").lower() for s in sp
                )
                yield pno, round(sz, 1), txt, is_bold, is_isolated


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
        # "(무)"(무배당) 접두는 상품명 표지 줄의 강한 신호 — 동급 후보 중 우선.
        # 그다음 폰트 크기, 그다음 길이(짧은 "보험약관" 같은 잡음 줄보다 실제 상품명이 김).
        product_cands.sort(key=lambda x: (x[2].strip().startswith("(무)"), x[0], x[1]),
                           reverse=True)
        product = product_cands[0][2].strip()

    front_end_page = (base_start_page - 1) if base_start_page else None

    return Detection(
        body_size=body_size, title_size=title_size,
        title_lo=title_lo, title_hi=title_hi,
        base_start_page=base_start_page, front_end_page=front_end_page,
        product=product, title_evidence=title_evidence,
    )


def _is_divider(txt: str, product: str | None) -> bool:
    if _ROMAN_CH.match(txt):
        return True
    if DIVIDER.match(txt):
        # 번호가 붙어도 괄호 한정어가 있는 긴 이름은 특약 제목이다
        # ("13. 일반상해입원일당(1일이상)(355간편가입)보장 특별약관").
        if not TITLE_SUFFIX.search(txt.replace(" ", "")):
            return True
        if "(" in txt or "（" in txt:
            return False
        return bool(_GENERIC_GROUP.search(txt)) or len(txt) < 18
    if txt.endswith("담보"):
        return True
    if product and txt == f"{product} 특별약관":
        return True
    if re.match(r"^.{2,30}\s특별약관$", txt) and "제" not in txt and len(txt) < 25:
        return txt.endswith("보험 특별약관")
    # "제N관 OOO"(예: 제1관 목적 및 용어의 정의)는 보통약관·각 특약 내부에 반복되는
    # 하위 목차 제목이다. 제목 폰트로 잡히면 그때마다 새 "yak" 경계가 되어 현재
    # 특약/보통약관 라벨을 덮어쓰므로, 여러 특약이 같은 관 이름을 재사용하는 문서에서
    # 서로 다른 특약의 조항이 같은 section으로 뒤섞이는 원인이 된다. 새 경계로 만들지
    # 않고 현재 라벨(부모 특약/보통약관)이 계속 이어지게 한다.
    if GWAN.match(txt):
        return True
    return False


def find(doc, det: Detection | None = None) -> tuple[list[Boundary], Detection]:
    """다중 신호 점수제(폰트+굵기+고립도+접미사+길이)로 경계 목록을 만든다.

    폰트 크기 단독 판정(예전 방식)은 (a) 목차·마케팅 문구를 제목으로 오인하거나
    (b) 문서에 제목 폰트 신호가 없으면 경계 감지가 통째로 죽는 문제가 있었다.
    점수제는 폰트를 여러 신호 중 하나로 낮춰서 신호 없는 문서에서도 접미사·굵기·
    고립도만으로 버틸 여지를 주고, 노이즈는 음의 신호(점선·타법령 인용)로 누른다.
    """
    if det is None:
        det = detect(doc)

    bounds: list[Boundary] = []
    front_end = det.front_end_page or 0
    # 제목이 여러 줄로 감길 수 있어("2. 일반상해사망(체증형,10년후2배지급)" +
    # "(355간편가입)보장 특별약관") 경계 확정은 항상 pending 묶음 단위로 한다:
    # 신호가 있는 줄(score>0)을 같은 (페이지,폰트크기) 키로 모으고, 묶음의
    # 공백 제거본이 특약 접미사에 도달한 순간 병합 라벨로 판정한다. 접미사에
    # 도달하지 못한 묶음(법령명·콜아웃·상품명 조각)은 경계가 되지 않는다.
    pending: list[tuple[str, float, bool, bool]] = []
    pending_key: tuple[int, float] | None = None

    def _flush():
        nonlocal pending, pending_key
        pending, pending_key = [], None

    for pno, sz, txt, is_bold, is_isolated in _lines_rich(doc):
        m = BYEOLPYO.match(txt)
        if m and len(txt) < 45:
            lab = re.sub(r"\s+", " ", f"별표{m.group(1)} {m.group(2)}").strip()
            bounds.append(Boundary(pno, lab, "byeolpyo"))
            _flush()
            continue
        if pno <= front_end:
            _flush()
            continue
        if (det.title_lo is not None and sz >= det.title_lo
                and _TABLE_TITLE.match(txt) and not _MID_SENTENCE.search(txt)):
            bounds.append(Boundary(pno, txt, "byeolpyo"))
            _flush()
            continue
        if _APPENDIX_HEAD.match(txt) and (is_isolated or
                (det.title_lo is not None and sz >= det.title_lo)):
            bounds.append(Boundary(pno, txt, "byeolpyo"))
            _flush()
            continue

        mt = _SENT_TAIL.match(txt)
        if mt:
            txt = mt.group(1).strip()

        key = (pno, sz)
        if _score_candidate(txt, sz, is_bold, is_isolated, det) <= 0:
            # 접미사 단독 줄("특별약관")은 표지 장식이라 후보가 아니지만,
            # 같은 키의 미완성 제목 조각이 대기 중이면 감긴 제목의 꼬리다
            # ("75. 암진단비(유사암제외)(355간편가입)보장" + "특별약관").
            if not (pending and pending_key == key and _BARE_SUFFIX.match(txt)):
                _flush()
                continue
        if pending_key != key:
            pending = []
        pending_key = key
        pending = (pending + [(txt, sz, is_bold, is_isolated)])[-3:]

        label = _try_title(pending, det)
        if label is None:
            continue                      # 접미사 미도달 — 다음 줄과 병합 대기
        if len(pending) > 1:
            # 마지막 줄이 그 자체로 완결된 제목이고 이어붙일 단서(여는 괄호로
            # 시작, 접미사 단독, 앞부분 괄호 미닫힘, 번호 시작 앞줄)가 없으면
            # 앞의 대기분은 우연히 낀 본문 조각이다 — 버리고 단독 제목만 쓴다.
            last_txt = pending[-1][0]
            head = " ".join(p[0] for p in pending[:-1])
            continuation = (last_txt.startswith(("(", "（"))
                            or _BARE_SUFFIX.match(last_txt)
                            or DIVIDER.match(pending[0][0])
                            or (head.count("(") + head.count("（")
                                > head.count(")") + head.count("）")))
            if not continuation:
                alone = _try_title([pending[-1]], det)
                if alone is not None:
                    label = alone
        if _is_divider(label, det.product):
            _flush()
            continue
        bounds.append(Boundary(pno, label, "yak"))
        _flush()

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
