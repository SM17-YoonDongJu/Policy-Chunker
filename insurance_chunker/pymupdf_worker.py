"""PyMuPDF 표 작업 프로세스 — 쪽 묶음의 표를 뽑아 JSON으로 낸다(extractor.PymupdfTables가 부른다).

  python -m insurance_chunker.pymupdf_worker <pdf> <쪽,쪽,...>
  → {"tables": {"쪽": markdown}, "seconds": 초, "pid": pid}

multiprocessing(spawn)을 쓰지 않는 이유: spawn은 작업자마다 부모의 __main__ 스크립트를 다시
import한다. 운영에서 부모는 ingest_catalog.py 같은 CLI라, 작업자마다 그 스크립트의 최상위 코드가
다시 돈다. 이 모듈만 실행하면 작업자가 하는 일이 정확히 이것뿐이다.
"""
from __future__ import annotations

import json
import os
import sys
import time


def main() -> None:
    import pymupdf as fitz

    from .extractor import extract_pymupdf

    pdf_path, pages_arg = sys.argv[1], sys.argv[2]
    pages = [int(p) for p in pages_arg.split(",") if p]
    out: dict[str, str] = {}
    t0 = time.perf_counter()
    with fitz.open(pdf_path) as doc:
        for pno in pages:
            md = extract_pymupdf(doc[pno - 1])
            if md:
                out[str(pno)] = md
    json.dump({"tables": out, "seconds": time.perf_counter() - t0, "pid": os.getpid()},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
