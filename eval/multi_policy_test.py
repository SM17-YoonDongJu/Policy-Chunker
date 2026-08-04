"""여러 약관 PDF에 파이프라인 풀 매트릭스 실행.

각 PDF마다: 파싱 → 청킹 → chunk_type 분포(키워드) → LLM 재분류(이어받기) → 분포 비교.
무결성 검사는 별도로: .venv/bin/python eval/integrity_check.py <chunks> <pdf>

실행: .venv/bin/python eval/multi_policy_test.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter

sys.path.insert(0, ".")

from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf
from insurance_chunker.llm_classifier import classify_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TARGETS = [
    ("in/실손의료비_삼성화재다이렉트.pdf", "삼성화재", "다이렉트 실손의료비보험", "samsung_silson"),
    ("in/단체상해_빅히트_동부.pdf", "DB손해보험", "빅히트단체상해보험", "db_bighit"),
]


def run_one(pdf: str, insurer: str, product: str, slug: str) -> None:
    print(f"\n{'='*70}\n### {insurer} {product} ({pdf})\n{'='*70}", flush=True)
    chunk_file = f"eval/chunks_{slug}.jsonl"

    if not os.path.exists(chunk_file):
        meta = DocMeta(
            source_pdf=pdf.split("/")[-1], doc_hash=compute_doc_hash(pdf),
            doc_type="policy_terms", insurer=insurer, product_name=product,
        )
        t0 = time.time()
        pages = parse_pdf(pdf, use_ocr=False, use_vision=False)
        chunks, tables = chunk_document(pages, meta, pdf_path=pdf)
        print(f"청킹: {len(chunks)}청크, 표 {len(tables)}건 ({time.time()-t0:.0f}s)", flush=True)
        with open(chunk_file, "w") as f:
            for c in chunks:
                f.write(json.dumps({
                    "chunk_index": c.chunk_index, "page": c.page_number,
                    "section": c.section, "article": c.article_number,
                    "article_title": c.article_title, "chunk_type": c.chunk_type,
                    "token_count": c.token_count, "table_id": c.table_id,
                    "content": c.content,
                }, ensure_ascii=False) + "\n")

    chunks = [json.loads(l) for l in open(chunk_file)]
    print("키워드 분포:", dict(Counter(c["chunk_type"] for c in chunks).most_common()), flush=True)

    # LLM 재분류 (이어받기)
    llm_file = f"eval/chunks_{slug}_llm_types.jsonl"
    done = set()
    if os.path.exists(llm_file):
        done = {json.loads(l)["chunk_index"] for l in open(llm_file)}
    todo = [c for c in chunks if c["chunk_index"] not in done]
    print(f"LLM 재분류: {len(done)}건 완료, {len(todo)}건 남음", flush=True)
    out = open(llm_file, "a")
    t0 = time.time()
    for i, c in enumerate(todo):
        llm_ct = classify_llm(c["content"], title=c.get("article_title")) or c["chunk_type"]
        out.write(json.dumps({"chunk_index": c["chunk_index"], "keyword": c["chunk_type"],
                              "llm": llm_ct}, ensure_ascii=False) + "\n")
        out.flush()
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    out.close()

    rows = [json.loads(l) for l in open(llm_file)]
    diff = sum(1 for r in rows if r["keyword"] != r["llm"])
    print(f"LLM 분포: {dict(Counter(r['llm'] for r in rows).most_common())}", flush=True)
    print(f"키워드 대비 불일치: {diff}/{len(rows)} ({diff/len(rows):.0%})", flush=True)


def main() -> None:
    for pdf, insurer, product, slug in TARGETS:
        run_one(pdf, insurer, product, slug)


if __name__ == "__main__":
    main()
