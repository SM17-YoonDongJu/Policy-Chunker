"""청크 무결성 검사: 원문 대비 텍스트 커버리지 + 조 번호 연속성.

1. 커버리지 — PDF 원문(서식 페이지 제외)의 문자들이 청크에 얼마나 담겼는지.
   누락(원문에 있는데 청크에 없음)이 크면 flush/clean 단계에서 버려진 것.
2. 조 연속성 — 같은 section 안에서 제N조가 건너뛰면 경계 감지 실패 의심.
3. 중복 — 같은 본문이 여러 청크에 들어갔는지.

실행: .venv/bin/python eval/integrity_check.py eval/chunks_30327.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, ".")


def norm(s: str) -> str:
    """공백·기호 차이를 무시한 문자 스트림."""
    return re.sub(r"[\s|․·⋅‧,.\-()\[\]【】〔〕:;'\"]+", "", s)


def main() -> None:
    chunk_file = sys.argv[1] if len(sys.argv) > 1 else "eval/chunks_30327.jsonl"
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else "in/상해보험_단체안심생활보험_30327.pdf"
    chunks = [json.loads(l) for l in open(chunk_file)]

    # ── 1. 커버리지 ──────────────────────────────────────────────────────────
    import fitz
    doc = fitz.open(pdf_path)
    # 청킹이 다룬 페이지 범위만 (서식 페이지 스킵 반영: 청크 최소 페이지부터)
    first_page = min(c["page"] for c in chunks)
    page_text = {p + 1: doc[p].get_text() for p in range(first_page - 1, len(doc))}

    chunk_by_page = defaultdict(str)
    for c in chunks:
        chunk_by_page[c["page"]] += c["content"]

    total_src = total_missing = 0
    worst: list[tuple[float, int, int]] = []
    for pno, text in page_text.items():
        src = norm(text)
        if not src:
            continue
        # 페이지 텍스트의 각 12자 슬라이스가 해당 페이지 ±1 청크에 존재하는지 샘플링
        joined = norm(chunk_by_page.get(pno - 1, "") + chunk_by_page.get(pno, "")
                      + chunk_by_page.get(pno + 1, ""))
        miss = 0
        n_probe = 0
        for i in range(0, max(1, len(src) - 12), 40):
            probe = src[i:i + 12]
            if len(probe) < 12:
                break
            n_probe += 1
            if probe not in joined:
                miss += 1
        if n_probe == 0:
            continue
        rate = miss / n_probe
        total_src += n_probe
        total_missing += miss
        if rate > 0.3:
            worst.append((rate, pno, n_probe))

    cov = 1 - (total_missing / total_src if total_src else 0)
    print(f"1) 원문 커버리지: {cov:.1%} (프로브 {total_src}개 중 누락 {total_missing})")
    worst.sort(reverse=True)
    for rate, pno, n in worst[:8]:
        print(f"   ⚠ p{pno}: 누락률 {rate:.0%} ({n}프로브)")
    if len(worst) > 8:
        print(f"   … 외 {len(worst)-8}페이지")

    # ── 2. 조 번호 연속성 ────────────────────────────────────────────────────
    print("\n2) 조 번호 연속성 (섹션별)")
    by_sec: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        if c.get("article"):
            m = re.match(r"제(\d+)조", c["article"])
            if m:
                by_sec[c["section"]].append(int(m.group(1)))
    gaps = 0
    for sec, arts in by_sec.items():
        seen = sorted(set(arts))
        missing = [n for n in range(seen[0], seen[-1] + 1) if n not in seen]
        if missing:
            gaps += 1
            print(f"   ⚠ {sec[:40]}: 제{missing}조 누락 (보유: {seen[0]}~{seen[-1]})")
    if not gaps:
        print("   이상 없음 — 모든 섹션에서 조 번호 연속")

    # ── 3. 본문 중복 ─────────────────────────────────────────────────────────
    print("\n3) 본문 중복")
    seen_body: dict[str, int] = {}
    dups = 0
    for c in chunks:
        key = norm(c["content"])[:200]
        if key in seen_body:
            dups += 1
        else:
            seen_body[key] = c["chunk_index"]
    print(f"   중복 의심: {dups}건 / {len(chunks)}청크")


if __name__ == "__main__":
    main()
