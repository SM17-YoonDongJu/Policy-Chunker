"""in/ 의 모든 약관 PDF 일괄 처리: 청킹 → 무결성 요약 → Gemma4 분류(이어받기).

실행: nohup .venv/bin/python eval/batch_all_policies.py > eval/batch_all.log 2>&1 &
1단계(청킹+무결성)는 빠르게 전체 완료 후, 2단계(LLM 분류)를 순차 진행.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

from insurance_chunker.chunker import chunk_document
from insurance_chunker.models import DocMeta, compute_doc_hash
from insurance_chunker.pdf_parser import parse_pdf
from insurance_chunker.llm_classifier import classify_llm

DONE_SLUGS = {"상해보험_단체안심생활보험_30327", "실손의료비_삼성화재다이렉트", "단체상해_빅히트_동부"}


def log(m):
    print(m, flush=True)


def integrity(chunks: list[dict], pdf: str) -> str:
    """조 연속성 갭 요약 한 줄."""
    by_sec = defaultdict(list)
    for c in chunks:
        if c.get("article"):
            m = re.match(r"제(\d+)조", c["article"])
            if m:
                by_sec[c["section"]].append(int(m.group(1)))
    gaps = 0
    for arts in by_sec.values():
        s = sorted(set(arts))
        gaps += sum(1 for n in range(s[0], s[-1] + 1) if n not in s)
    return f"섹션 {len(by_sec)}개, 조번호 갭 {gaps}개"


def main() -> None:
    pdfs = sorted(p for p in Path("in").glob("*.pdf") if p.stem not in DONE_SLUGS)
    log(f"대상 {len(pdfs)}개 약관\n===== 1단계: 청킹 + 무결성 =====")

    summary = []
    for pdf in pdfs:
        slug = pdf.stem
        cf = Path(f"eval/chunks_{slug}.jsonl")
        if not cf.exists():
            t0 = time.time()
            try:
                meta = DocMeta(source_pdf=pdf.name, doc_hash=compute_doc_hash(str(pdf)),
                               doc_type="policy_terms", insurer="메리츠화재", product_name=slug)
                pages = parse_pdf(str(pdf), use_ocr=False, use_vision=False)
                chunks, _ = chunk_document(pages, meta, pdf_path=str(pdf))
            except Exception as e:
                log(f"[ERR] {slug}: {type(e).__name__} {e}")
                continue
            with cf.open("w") as f:
                for c in chunks:
                    f.write(json.dumps({
                        "chunk_index": c.chunk_index, "page": c.page_number,
                        "section": c.section, "article": c.article_number,
                        "article_title": c.article_title, "chunk_type": c.chunk_type,
                        "token_count": c.token_count, "table_id": c.table_id,
                        "is_boilerplate": c.is_boilerplate, "content": c.content,
                    }, ensure_ascii=False) + "\n")
            dt = time.time() - t0
        else:
            dt = 0
        rows = [json.loads(l) for l in cf.open()]
        bp = sum(1 for c in rows if c.get("is_boilerplate"))
        info = f"{slug}: {len(rows)}청크 (boilerplate {bp}), {integrity(rows, str(pdf))} ({dt:.0f}s)"
        summary.append(info)
        log("  " + info)

    log("\n===== 2단계: Gemma4 분류 (이어받기) =====")
    for pdf in pdfs:
        slug = pdf.stem
        cf = Path(f"eval/chunks_{slug}.jsonl")
        if not cf.exists():
            continue
        chunks = [json.loads(l) for l in cf.open()]
        targets = [c for c in chunks if not c.get("is_boilerplate")]
        out_path = Path(f"eval/chunks_{slug}_llm_types.jsonl")
        done = set()
        if out_path.exists():
            done = {json.loads(l)["chunk_index"] for l in out_path.open()}
        todo = [c for c in targets if c["chunk_index"] not in done]
        log(f"[{slug}] 분류 대상 {len(targets)} (완료 {len(done)}, 남음 {len(todo)})")
        t0 = time.time()
        with out_path.open("a") as out:
            for i, c in enumerate(todo):
                ct = classify_llm(c["content"], title=c.get("article_title")) or c["chunk_type"]
                out.write(json.dumps({"chunk_index": c["chunk_index"],
                                      "keyword": c["chunk_type"], "llm": ct},
                                     ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 200 == 0:
                    log(f"  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)")
        rows = [json.loads(l) for l in out_path.open()]
        diff = sum(1 for r in rows if r["keyword"] != r["llm"])
        log(f"  완료 — LLM 분포 {dict(Counter(r['llm'] for r in rows).most_common(4))} | "
            f"키워드 불일치 {diff}/{len(rows)}")

    log("\n===== 전체 완료 =====")
    for s in summary:
        log("  " + s)


if __name__ == "__main__":
    main()
