"""3문서(30327/프리미엄간편보험2604/DB빅히트) 청킹 회귀 요약 — 조번호갭이 핵심 게이트.

실행: .venv/bin/python eval/regress3.py
"""
from __future__ import annotations

import re
import sys
import warnings
import logging
from collections import defaultdict

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf

DOCS = [
    ("30327", "in/상해보험_단체안심생활보험_30327.pdf", "메리츠화재", "단체안심생활보험"),
    ("프리미엄간편보험2604", "in/프리미엄간편보험2604.pdf", "메리츠화재", "프리미엄간편보험2604"),
    ("DB빅히트", "in/단체상해_빅히트_동부.pdf", "동부화재", "빅히트"),
]


def run(label: str, pdf: str, insurer: str, product: str) -> None:
    meta = DocMeta(source_pdf=pdf, doc_hash=compute_doc_hash(pdf), doc_type="policy_terms",
                    insurer=insurer, product_name=product)
    pages = parse_pdf(pdf, use_ocr=False, use_vision=False)
    chunks, _ = chunk_document(pages, meta, pdf_path=pdf)

    by_sec = defaultdict(list)
    n_art_kind = 0
    n_has_art = 0
    for c in chunks:
        if c.section is not None and not c.table_id:
            n_art_kind += 1
            if c.article_number:
                n_has_art += 1
        if c.article_number:
            m = re.match(r"제(\d+)조", c.article_number)
            if m:
                by_sec[c.section].append(int(m.group(1)))

    gap_rows = []
    for sec, arts in by_sec.items():
        s = sorted(set(arts))
        missing = [n for n in range(s[0], s[-1] + 1) if n not in s]
        if missing:
            gap_rows.append((len(missing), sec, s[0], s[-1]))
    gap_rows.sort(reverse=True)
    total_gap = sum(r[0] for r in gap_rows)
    restore_rate = n_has_art / n_art_kind if n_art_kind else 0.0

    print(f"\n=== {label} ===")
    print(f"청크 {len(chunks)}, 섹션 {len(by_sec)}개, 조복원율 {restore_rate:.0%}, "
          f"조번호갭 {total_gap}개 (갭섹션 {len(gap_rows)}개)")
    for r in gap_rows[:6]:
        print(f"   ⚠ 갭{r[0]} {r[1][:35]} ({r[2]}~{r[3]})")


if __name__ == "__main__":
    for args in DOCS:
        run(*args)
