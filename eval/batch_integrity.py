"""여러 문서에 원문 커버리지 + 조 연속성 + 중복 검사를 일괄 실행 (이어받기 가능).

실행: .venv/bin/python eval/batch_integrity.py
결과: eval/batch_integrity.jsonl (문서당 한 줄 append)
"""
from __future__ import annotations

import json
import logging
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import fitz
from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf

OUT = Path("eval/batch_integrity.jsonl")

DOCS = [
    "간편31건강보험2604", "간편한정기보험2601", "다이렉트실손_계약전환2605",
    "다이렉트암보험_재가입2601", "단체안심상해보험_30324", "상해보험_30273",
    "상해안심보험2601", "실손의료비보험2605", "올바른정기보험2601",
    "올바른치아보험2601", "임원단체안심상해보험_30167", "전국민생활체육단체보험_29926",
    "치아보험_이목구비2601", "통합간편건강보험2604", "프리미엄간편보험2604",
    "프리미엄건강보험2604", "함께하는단체보험2601",
]


def norm(s: str) -> str:
    return re.sub(r"[\s|․·⋅‧,.\-()\[\]【】〔〕:;'\"]+", "", s)


def check(pdf_path: str) -> dict:
    meta = DocMeta(source_pdf=pdf_path, doc_hash=compute_doc_hash(pdf_path),
                   doc_type="policy_terms", insurer="-", product_name=Path(pdf_path).stem)
    pages = parse_pdf(pdf_path, use_ocr=False, use_vision=False)
    chunks, _ = chunk_document(pages, meta, pdf_path=pdf_path)

    doc = fitz.open(pdf_path)
    chunk_dicts = [{"page": c.page_number, "content": c.content, "article": c.article_number,
                    "section": c.section} for c in chunks]
    first_page = min(c["page"] for c in chunk_dicts)
    page_text = {p + 1: doc[p].get_text() for p in range(first_page - 1, doc.page_count)}

    chunk_by_page = defaultdict(str)
    for c in chunk_dicts:
        chunk_by_page[c["page"]] += c["content"]

    total_probes = missing_probes = 0
    worst_pages = []
    for pno, text in page_text.items():
        raw_n = norm(text)
        if len(raw_n) < 20:
            continue
        # 조 단위 청크는 page_number가 조 시작 페이지로 고정돼, 조가 페이지
        # 경계를 넘으면 내용이 인접 페이지 번호로 잡힐 수 있다 — ±1 합쳐서 대조
        # (integrity_check.py와 동일 방식).
        chunk_n = norm(chunk_by_page.get(pno - 1, "") + chunk_by_page.get(pno, "")
                      + chunk_by_page.get(pno + 1, ""))
        probes = [raw_n[i:i + 12] for i in range(0, max(1, len(raw_n) - 12), 40)]
        if not probes:
            continue
        miss = sum(1 for p in probes if len(p) == 12 and p not in chunk_n)
        total_probes += len(probes)
        missing_probes += miss
        if len(probes) and miss / len(probes) > 0.5:
            worst_pages.append((pno, round(miss / len(probes), 2)))
    worst_pages.sort(key=lambda x: -x[1])

    by_sec = defaultdict(list)
    for c in chunk_dicts:
        if c["article"] and c["section"]:
            m = re.match(r"제(\d+)조", c["article"])
            if m:
                by_sec[c["section"]].append(int(m.group(1)))
    gaps = 0
    for arts in by_sec.values():
        s = sorted(set(arts))
        gaps += sum(1 for n in range(s[0], s[-1] + 1) if n not in s)

    contents = [c["content"] for c in chunk_dicts if not c.get("article") or True]
    dup = sum(1 for i, a in enumerate(contents) for b in contents[i + 1:i + 3]
              if a == b and len(a) > 30)

    return {
        "coverage": round(1 - missing_probes / total_probes, 4) if total_probes else None,
        "gaps": gaps, "chunks": len(chunk_dicts), "dup": dup,
        "worst_pages": worst_pages[:5],
    }


def main() -> None:
    done = set()
    if OUT.exists():
        done = {json.loads(l)["pdf"] for l in OUT.open() if l.strip()}
    for stem in DOCS:
        if stem in done:
            print(f"skip {stem}", flush=True)
            continue
        pdf_path = f"in/{stem}.pdf"
        try:
            row = {"pdf": stem, **check(pdf_path)}
        except Exception as e:
            row = {"pdf": stem, "error": str(e)[:300]}
        with OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
