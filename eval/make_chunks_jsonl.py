"""현재 청킹 코드로 PDF를 재청킹해 평가용 jsonl 생성.

실행: .venv/bin/python eval/make_chunks_jsonl.py <pdf> <out.jsonl> <보험사> <상품명>
출력 필드는 eval/chunks_30327_final.jsonl(v2)과 동일 스키마.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf


def main(pdf: str, out: str, insurer: str, product: str) -> None:
    meta = DocMeta(source_pdf=pdf, doc_hash=compute_doc_hash(pdf),
                   doc_type="policy_terms", insurer=insurer, product_name=product)
    pages = parse_pdf(pdf, use_ocr=False, use_vision=False)
    chunks, _ = chunk_document(pages, meta, pdf_path=pdf)
    with open(out, "w") as f:
        for i, c in enumerate(chunks, 1):
            f.write(json.dumps({
                "chunk_index": i,
                "page": c.page_number,
                "section": c.section,
                "article": c.article_number,
                "article_title": c.article_title,
                "chunk_type": c.chunk_type,
                "token_count": c.token_count,
                "table_id": c.table_id,
                "is_boilerplate": getattr(c, "is_boilerplate", False),
                "content": c.content,
            }, ensure_ascii=False) + "\n")
    print(f"{out}: {len(chunks)} chunks")


if __name__ == "__main__":
    main(*sys.argv[1:5])
