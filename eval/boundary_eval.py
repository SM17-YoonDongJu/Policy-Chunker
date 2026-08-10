"""경계 감지 정밀도/재현율 채점 — boundaries.py 재설계 실험용.

30327: 자체 검증된(0갭) 결과를 정답으로 신뢰.
프리미엄간편보험2604: 클로드 참조본에서 "깨끗한 제목"만 필터링한 정답(완벽하지 않음,
    클로드 자신도 이 문서에서 제목 파편화 버그가 있어 silver 기준으로만 사용).
DB빅히트: 참조본 없음 — 특약 경계 개수만 정성 비교.

실행: .venv/bin/python eval/boundary_eval.py
"""
from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, ".")
import fitz
from insurance_chunker.boundaries import detect, find

CLEAN_TITLE = re.compile(r"^.{2,60}(특별약관|보통약관|특약|담보)[ⅠⅡⅢⅣⅤ0-9()]*$")
_TOC_ENTRY = re.compile(r"^(.*?)\s*[·.…]{3,}\s*\d+\s*$")


def norm(s: str) -> str:
    """번호 접두어("13. ")와 공백 차이를 무시하고 비교 — 조판 줄바꿈이 만든
    단어 내 공백("입 원일당")과 클로드 참조본의 무번호 표기 때문에 문자열
    완전일치는 같은 제목을 다른 제목으로 센다."""
    s = re.sub(r"^\d+\.\s*", "", s.strip())
    return re.sub(r"\s+", "", s)


def clean_reference(claude_json_path: str) -> set[str]:
    claude = json.load(open(claude_json_path))
    secs = {c.get("section") or "" for c in claude}
    return {norm(s) for s in secs if CLEAN_TITLE.match(norm(s))}


def reference_from_toc(pdf_path: str) -> set[str]:
    """문서 자체 목차(점선 리더 + 페이지 번호)에서 특약 제목을 추출 — 클로드
    참조본과 달리 문서가 스스로 선언한 정답이다. 목차 제목이 두 줄로 감기면
    리더가 붙은 줄에서 앞 줄 조각을 이어붙인다."""
    doc = fitz.open(pdf_path)
    front_end = detect(doc).front_end_page or min(doc.page_count, 45)
    ref: set[str] = set()
    buf: list[str] = []
    for pno in range(1, front_end + 1):
        for line in doc[pno - 1].get_text().splitlines():
            s = line.strip()
            if not s:
                buf = []
                continue
            m = _TOC_ENTRY.match(s)
            if m:
                n = norm(" ".join(buf + [m.group(1)]))
                buf = []
                # 로마숫자 장 헤더("Ⅱ. 질병 관련 특별약관")는 특약 묶음 제목이라
                # 경계 감지가 의도적으로 구분선 처리한다 — 정답에서 제외.
                if CLEAN_TITLE.match(n) and not re.match(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.", n):
                    ref.add(n)
            else:
                buf = (buf + [s])[-2:]
    return ref


def score(detected: set[str], reference: set[str]) -> dict:
    tp = len(detected & reference)
    precision = tp / len(detected) if detected else 0.0
    recall = tp / len(reference) if reference else 0.0
    return {"precision": precision, "recall": recall, "tp": tp,
            "detected": len(detected), "reference": len(reference)}


def run(pdf_path: str, label: str, reference: set[str] | None) -> None:
    doc = fitz.open(pdf_path)
    det = detect(doc)
    bounds, _ = find(doc, det)
    detected = {norm(b.label) for b in bounds if b.kind == "yak"}

    print(f"\n=== {label} ===")
    print(f"감지된 특약 경계: {len(detected)}개")
    if reference is not None:
        m = score(detected, reference)
        print(f"정밀도={m['precision']:.0%} 재현율={m['recall']:.0%} "
              f"(일치 {m['tp']} / 감지 {m['detected']} / 정답 {m['reference']})")
        missed = reference - detected
        extra = detected - reference
        if missed:
            print(f"  놓친 것 예시: {list(missed)[:5]}")
        if extra:
            print(f"  오탐 예시: {list(extra)[:5]}")
    else:
        print("(참조본 없음 — 개수만 기록)")


if __name__ == "__main__":
    for pdf, label in (
        ("in/상해보험_단체안심생활보험_30327.pdf", "30327"),
        ("in/프리미엄간편보험2604.pdf", "프리미엄간편보험2604"),
        ("in/단체상해_빅히트_동부.pdf", "DB빅히트"),
    ):
        ref = reference_from_toc(pdf)
        # 목차가 없거나 빈약한 문서는 참조 없이 개수만 기록
        run(pdf, f"{label} (목차 정답 {len(ref)}개)", ref if len(ref) >= 10 else None)
