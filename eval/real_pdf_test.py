"""실제 약관 PDF로 하이브리드 분류 테스트.

1. PDF 청킹 (키워드 분류 기준)
2. 면책/지급 복합문(애매 케이스) 추출
3. 애매 케이스를 LLM으로 재분류 → 키워드 판정과 비교 출력

실행: .venv/bin/python eval/real_pdf_test.py in/상해보험_단체안심생활보험_30327.pdf [샘플수]
"""
from __future__ import annotations

import logging
import sys
import time

sys.path.insert(0, ".")

from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf
from insurance_chunker.rechunk import _is_ambiguous
from insurance_chunker.llm_classifier import classify_llm, llm_available

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    pdf_path = sys.argv[1]
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    meta = DocMeta(
        source_pdf=pdf_path.split("/")[-1],
        doc_hash=compute_doc_hash(pdf_path),
        doc_type="policy_terms",
        insurer="메리츠화재",
        product_name="단체안심생활보험",
        effective_date="2026-05-29",
    )

    t0 = time.time()
    pages = parse_pdf(pdf_path, use_ocr=False, use_vision=False)
    chunks, table_metas = chunk_document(pages, meta, pdf_path=pdf_path)
    print(f"\n청킹 완료: {len(chunks)}청크, 표 {len(table_metas)}건 ({time.time()-t0:.0f}s)")

    from collections import Counter
    dist = Counter(c.chunk_type for c in chunks)
    print("chunk_type 분포:", dict(dist.most_common()))

    # 면책/지급 복합문 — 키워드 분류가 오판할 수 있는 케이스
    ambiguous = [
        c for c in chunks
        if not c.table_id and _is_ambiguous(
            (c.article_title or "") + " " + c.content, c.chunk_type)
    ]
    print(f"\n애매 케이스(면책·지급 키워드 동시 등장): {len(ambiguous)}건 / {len(chunks)}청크")

    if not llm_available():
        print("LLM 미준비 — 키워드 결과만 출력하고 종료")
        return

    print(f"\n=== LLM 재분류 (앞 {n_sample}건) ===")
    diff = 0
    times = []
    for c in ambiguous[:n_sample]:
        t1 = time.time()
        llm_ct = classify_llm(c.content, title=c.article_title)
        dt = time.time() - t1
        times.append(dt)
        mark = " " if llm_ct == c.chunk_type else "≠"
        if llm_ct != c.chunk_type:
            diff += 1
        art = f"{c.article_number or ''}({c.article_title or ''})"
        print(f"  [{mark}] {art[:34]:<36} 키워드={c.chunk_type:<12} LLM={llm_ct} ({dt:.1f}s)")

    n = len(ambiguous[:n_sample])
    if n:
        print(f"\n판정 불일치: {diff}/{n}건 | LLM 평균 {sum(times)/len(times):.1f}s/건")
        est = len(ambiguous) * (sum(times) / len(times))
        print(f"전체 애매 케이스({len(ambiguous)}건) LLM 처리 예상: {est/60:.1f}분")


if __name__ == "__main__":
    main()
