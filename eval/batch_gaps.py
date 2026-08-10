"""in/ 전체 약관 PDF 조번호갭 일괄 측정 (LLM 분류 없이 빠르게, 이어받기 가능).

실행: .venv/bin/python eval/batch_gaps.py
결과: eval/batch_gaps.jsonl (문서당 한 줄 append — 완료분은 스킵)
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf

OUT = Path("eval/batch_gaps.jsonl")


def measure(pdf: Path) -> dict:
    t0 = time.time()
    meta = DocMeta(source_pdf=str(pdf), doc_hash=compute_doc_hash(str(pdf)),
                   doc_type="policy_terms", insurer="-", product_name=pdf.stem)
    pages = parse_pdf(str(pdf), use_ocr=False, use_vision=False)
    chunks, _ = chunk_document(pages, meta, pdf_path=str(pdf))

    by_sec = defaultdict(list)
    n_art_kind = n_has = 0
    for c in chunks:
        if c.section is not None and not c.table_id:
            n_art_kind += 1
            if c.article_number:
                n_has += 1
        if c.article_number and c.section:
            m = re.match(r"제(\d+)조", c.article_number)
            if m:
                by_sec[c.section].append(int(m.group(1)))
    gaps = 0
    worst = []
    for sec, arts in by_sec.items():
        s = sorted(set(arts))
        g = sum(1 for n in range(s[0], s[-1] + 1) if n not in s)
        if g:
            gaps += g
            worst.append((g, sec[:40]))
    worst.sort(reverse=True)
    return {"pdf": pdf.stem, "chunks": len(chunks), "sections": len(by_sec),
            "gaps": gaps, "restore": round(n_has / n_art_kind, 3) if n_art_kind else 0,
            "worst": worst[:3], "sec": round(time.time() - t0)}


def main() -> None:
    done = set()
    if OUT.exists():
        done = {json.loads(l)["pdf"] for l in OUT.open() if l.strip()}
    for pdf in sorted(Path("in").glob("*.pdf")):
        if pdf.stem in done:
            print(f"skip {pdf.stem}", flush=True)
            continue
        try:
            row = measure(pdf)
        except Exception as e:
            row = {"pdf": pdf.stem, "error": str(e)[:200]}
        with OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
