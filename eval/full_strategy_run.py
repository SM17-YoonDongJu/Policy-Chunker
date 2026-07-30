"""전체 표 추출 전략(PyMuPDF + pdfplumber + camelot + VLM) 풀가동 청킹.

실행: .venv/bin/python eval/full_strategy_run.py in/상해보험_단체안심생활보험_30327.pdf
결과: eval/chunks_30327_full.jsonl + 소스별 채택 통계
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter

sys.path.insert(0, ".")

from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    pdf_path = sys.argv[1]
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
    t_parse = time.time() - t0

    src_dist = Counter(
        p.tables[0]["source"] for p in pages if p.tables
    )
    print(f"\n표 소스 채택 분포 (best-of): {dict(src_dist.most_common())}")

    chunks, table_metas = chunk_document(pages, meta, pdf_path=pdf_path)
    print(f"청킹 완료: {len(chunks)}청크, 표 {len(table_metas)}건 "
          f"(파싱 {t_parse:.0f}s + 청킹 {time.time()-t0-t_parse:.0f}s)")

    dist = Counter(c.chunk_type for c in chunks)
    print("chunk_type 분포:", dict(dist.most_common()))

    out = "eval/chunks_30327_full.jsonl"
    with open(out, "w") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_index": c.chunk_index, "page": c.page_number,
                "section": c.section, "article": c.article_number,
                "article_title": c.article_title, "chunk_type": c.chunk_type,
                "token_count": c.token_count, "table_id": c.table_id,
                "content": c.content,
            }, ensure_ascii=False) + "\n")
    print(f"저장: {out}")

    tm_out = "eval/tables_30327_full.jsonl"
    with open(tm_out, "w") as f:
        for t in table_metas:
            f.write(json.dumps({
                "page": t.page_number, "extractor": t.extractor,
                "caption": t.caption, "rows": t.row_count, "cols": t.col_count,
                "markdown": t.markdown,
            }, ensure_ascii=False) + "\n")
    print(f"저장: {tm_out}")


if __name__ == "__main__":
    main()
