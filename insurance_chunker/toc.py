"""toc.py — 문서 자체 목차에서 특약 제목을 수확한다.

왜 필요한가: 특약 경계 검출은 폰트·접미사 같은 '조판 관습'에 의존해서 보험사가
바뀌면 무너진다(KB 실측: 섹션 7개, art_nonmono 473). 반면 목차는 조판이 아니라
**문서가 스스로 선언한 데이터**다. 목차를 읽으면 두 가지를 보험사 무관하게 얻는다:

  (a) 자가 검증 점수 — "목차가 선언한 특약 N개 중 우리가 몇 개를 잡았나".
      사람이 라벨링하지 않아도 문서별 품질을 판정할 수 있다.
  (b) 검출 신호 — 알려진 제목 문자열을 본문에서 매칭(분류 문제 → 매칭 문제).

목차 조판은 보험사마다 다르다(6개 문서 실측):
  A. 점선 리더 + 페이지번호   "…특별약관·········· 123"   (메리츠, DB프로미)
  B. 점선 리더, 페이지번호 없음 "약관 이용 가이드북··········"   (KB)
  C. 제목과 페이지번호가 다른 줄 "보통약관" / "62"            (삼성)
  D. 번호 없이 들여쓴 제목만     "  외상성절단진단비 특별약관"   (DB빅히트)
셋 다 지원한다.
"""
from __future__ import annotations

import re
from typing import Optional

from .boundaries import TITLE_SUFFIX, is_title_like

# A/B: 점선 리더(2자 이상) + 선택적 페이지번호
_LEADER = re.compile(r"^(.*?)\s*[·․.‧⋅…]{2,}\s*(\d{1,4})?\s*$")
# C: 페이지번호만 있는 줄
_ONLY_NUM = re.compile(r"^\s*\d{1,4}\s*$")
# D: 번호·기호 접두("13.", "1-2.", "- ", "∙ ")를 떼기
_NUM_PREFIX = re.compile(r"^\s*(?:[-–—∙·▪]\s*)?(?:\d{1,3}(?:-\d{1,3})?\s*[.)]?\s*)?")
# 목차 자체의 장 구분·안내 항목은 특약이 아니다(앞부분 가이드/유의사항 블록).
_TOC_NOISE = re.compile(
    r"^(목\s*차|약관\s*이용|보험용어|주요내용|가입자\s*유의|개인신용정보|자주\s*발생|"
    r"쉽게\s*이해|보험약관\s*이해|색인|요약서|QR|보험계약의?\s*개요|반드시\s*알아|"
    r"보험계약\s*이해|상품\s*및|분쟁|민원|고객\s*권리|이\s*상품은|주요\s*민원)")
# 전화번호·URL·순수 기호 등 제목이 아닌 항목
_NOT_TITLE = re.compile(r"^[\W\d]*$|^\(?0\d{1,2}\)?[-\s]?\d{3,4}[-\s]?\d{4}|https?://|www\.")
# 한글이 최소 2자는 있어야 특약 제목으로 본다
_HAS_KO = re.compile(r"[가-힣]{2,}")


def _norm(s: str) -> str:
    """번호 접두어와 공백 차이를 무시한 비교용 정규화.

    조판 줄바꿈이 단어 중간에 공백을 만들고("입 원일당"), 목차와 본문의
    번호 표기가 달라서 문자열 완전일치로는 같은 제목을 다르게 센다.
    """
    s = _NUM_PREFIX.sub("", s.strip())
    return re.sub(r"\s+", "", s)


def _toc_page_range(doc, front_end: Optional[int]) -> range:
    """목차가 있을 만한 앞부분. front_end(제1조 시작) 이전 + 여유.

    front_end가 비정상적으로 작게 잡히는 문서가 있어(삼성 1232p인데 2로 검출)
    그 경우엔 최소 60p를 훑는다 — 목차가 그 뒤에 있어도 놓치지 않게.
    """
    end = front_end if front_end else min(doc.page_count, 45)
    if end < 10:
        end = min(doc.page_count, 60)
    return range(0, min(end + 3, doc.page_count))


def extract_toc_titles(doc, front_end: Optional[int] = None) -> set[str]:
    """목차에서 특약 제목 후보를 수확 (정규화된 문자열 집합)."""
    return set(extract_toc_titles_ordered(doc, front_end))


def extract_toc_titles_ordered(doc, front_end: Optional[int] = None) -> list[str]:
    """목차 등장 순서를 보존한 특약 제목 목록 (정규화·중복 제거).

    순서가 중요한 이유: 목차는 특약을 문서 순서대로 나열하고, 본문의 조번호
    리셋도 같은 순서로 일어난다. 둘을 정렬하면 "경계는 리셋으로, 이름은 목차로"
    조합할 수 있다(rechunk.clean).

    Args:
        doc: fitz.Document
        front_end: 본문(제1조) 시작 페이지. None이면 앞 45p를 훑는다.
    """
    titles: _OrderedSet = _OrderedSet()
    for pno in _toc_page_range(doc, front_end):
        lines = [l for l in doc[pno].get_text().split("\n") if l.strip()]
        if len(lines) < 5:
            continue
        # 컬럼 폭에 밀려 두 줄로 감긴 제목("…(감액없" / "음)")을 잇기 위한 버퍼.
        # 종결자(점선/페이지번호/접미사)를 만날 때까지 모았다가 한 번에 flush한다.
        buf: list[str] = []

        def take(head: str) -> None:
            _add(titles, " ".join(buf + [head]) if buf else head)
            buf.clear()

        for i, raw in enumerate(lines):
            txt = raw.strip()
            if not txt or _TOC_NOISE.match(txt):
                buf.clear()
                continue

            # A/B — 점선 리더(페이지번호는 있어도 없어도 됨)
            m = _LEADER.match(txt)
            if m and m.group(1).strip():
                take(m.group(1).strip())
                continue

            # C — 다음 줄이 페이지번호만 있는 줄이면 이 줄이 제목
            if i + 1 < len(lines) and _ONLY_NUM.match(lines[i + 1]):
                take(txt)
                continue

            # D — 번호·페이지 없이 들여쓴 제목. 오탐을 막기 위해
            #     특약 접미사로 끝나는 줄만 인정한다.
            if raw != raw.lstrip() and TITLE_SUFFIX.search(txt):
                take(txt)
                continue

            # 종결자가 없는 줄 — 다음 줄로 이어지는 제목 조각일 수 있다.
            # 무한 누적을 막기 위해 최근 2줄만 유지한다.
            buf.append(txt)
            del buf[:-2]
    return list(titles)


class _OrderedSet:
    """등장 순서를 보존하는 집합 (dict 키 순서 이용)."""

    def __init__(self) -> None:
        self._d: dict[str, None] = {}

    def add(self, v: str) -> None:
        self._d.setdefault(v, None)

    def clear(self) -> None:
        self._d.clear()

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)


def _add(acc, raw: str) -> None:
    raw = raw.strip()
    t = _norm(raw)
    # 너무 짧거나(장 구분 조각), 전화번호·URL, 한글이 없는 항목, 본문 문장은 버린다.
    if not (4 <= len(t) <= 80):
        return
    if _NOT_TITLE.match(raw) or not _HAS_KO.search(t) or _TOC_NOISE.match(raw):
        return
    if is_title_like(raw):
        acc.add(t)


def match_rate(detected_sections: set[str], toc_titles: set[str]) -> tuple[float, list[str]]:
    """목차가 선언한 제목 중 검출된 섹션으로 커버된 비율과, 누락 목록.

    조판 줄바꿈·번호 차이 때문에 완전일치가 아니라 **양방향 부분일치**로 본다
    (목차 제목이 섹션 라벨에 포함되거나 그 반대).
    """
    if not toc_titles:
        return 0.0, []
    det = {_norm(s) for s in detected_sections if s}
    missing = []
    for t in toc_titles:
        if any(t in d or d in t for d in det):
            continue
        missing.append(t)
    return round(1 - len(missing) / len(toc_titles), 3), sorted(missing)
