"""DB빅히트 조복원율 분석: 조 라벨 None 청크의 정당/비정당 분류.

실행: .venv/bin/python eval/db_restore_analysis.py
"""
from __future__ import annotations

import logging
import re
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf

PDF = "in/단체상해_빅히트_동부.pdf"
_APPENDIX = re.compile(r"^(별표|붙임|부표)|분류표|장해분류|등급표")


def main() -> None:
    meta = DocMeta(source_pdf=PDF, doc_hash=compute_doc_hash(PDF),
                   doc_type="policy_terms", insurer="동부화재", product_name="빅히트")
    pages = parse_pdf(PDF, use_ocr=False, use_vision=False)
    chunks, _ = chunk_document(pages, meta, pdf_path=PDF)

    firsts = set()
    prev = object()
    for c in chunks:
        if c.section != prev:
            firsts.add(id(c))
            prev = c.section

    total = has = 0
    cat = {"별표/붙임": 0, "특약 첫청크(도입부)": 0, "그 외(진짜 미복원)": 0}
    samples = []
    for c in chunks:
        if c.section is None or c.table_id:
            continue
        total += 1
        if c.article_number:
            has += 1
            continue
        if _APPENDIX.search(c.section):
            cat["별표/붙임"] += 1
        elif id(c) in firsts:
            cat["특약 첫청크(도입부)"] += 1
        else:
            cat["그 외(진짜 미복원)"] += 1
            if len(samples) < 10:
                samples.append(f"  p{c.page_number} [{c.section[:30]}] {c.content[:60]!r}")

    print(f"비표 청크 {total}, 조 있음 {has} ({has/total:.0%})")
    for k, v in cat.items():
        print(f"  None – {k}: {v} ({v/total:.1%})")
    adj = has / (total - cat["별표/붙임"] - cat["특약 첫청크(도입부)"])
    print(f"보정 복원율(정당한 None 제외): {adj:.0%}")
    print("진짜 미복원 샘플:")
    print("\n".join(samples))


if __name__ == "__main__":
    main()
