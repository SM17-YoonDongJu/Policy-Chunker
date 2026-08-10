"""목차 기반 자가검증 점수 — 골든셋 없이 문서별 청킹 품질을 판정한다.

왜 필요한가: gaps·art_nonmono는 대리지표라 "놓친 특약"을 못 본다.
  - gaps=0      : 여러 특약이 한 섹션으로 뭉치면 결번이 사라져 오히려 0이 된다(KB).
  - nonmono=0   : 특약을 아예 못 잡으면 리셋이 관측되지 않아 0이 된다(DB프로미).
반면 목차는 문서가 스스로 선언한 특약 목록이라, 검출률이 곧 품질이다.
보험사가 몇 곳이든 사람 라벨링 없이 문서가 스스로 채점한다.

실행: .venv/bin/python eval/toc_score.py            # 전체(이어받기)
      .venv/bin/python eval/toc_score.py --summary  # 저장된 결과만 출력
결과: eval/toc_score.jsonl (문서당 한 줄 append)
"""
from __future__ import annotations

import json
import sys
import warnings
import logging
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import fitz  # noqa: E402
from insurance_chunker.boundaries import detect  # noqa: E402
from insurance_chunker.chunker import chunk_document  # noqa: E402
from insurance_chunker.models import DocMeta, compute_doc_hash  # noqa: E402
from insurance_chunker.pdf_parser import parse_pdf  # noqa: E402
from insurance_chunker.toc import extract_toc_titles_ordered, match_rate  # noqa: E402

OUT = Path("eval/toc_score.jsonl")

DOCS = [
    ("메리츠화재", "in/상해보험_단체안심생활보험_30327.pdf"),
    ("KB손해보험", "in/KB_금쪽같은자녀보험Plus_26.04.pdf"),
    ("현대해상", "in/현대_굿앤굿2040종합보험Hi2604.pdf"),
    ("삼성화재", "in/삼성_건강보험천만안심2601_2종.pdf"),
    ("DB프로미라이프", "in/DB_프로미라이프참좋은더보장간병2601.pdf"),
    ("DB빅히트", "in/단체상해_빅히트_동부.pdf"),
]

# 2026-08-11 베이스라인(리셋 분할·목차 명명 도입 전). 개선폭 비교용.
BASELINE = {"메리츠화재": 0.832, "KB손해보험": 0.002, "현대해상": 0.607,
            "삼성화재": 0.459, "DB프로미라이프": 0.280, "DB빅히트": 0.987}


def score_one(insurer: str, pdf: str) -> dict:
    doc = fitz.open(pdf)
    toc = extract_toc_titles_ordered(doc, detect(doc).front_end_page)
    meta = DocMeta(source_pdf=pdf, doc_hash=compute_doc_hash(pdf),
                   doc_type="policy_terms", insurer=insurer, product_name=Path(pdf).stem)
    chunks, _ = chunk_document(parse_pdf(pdf, use_ocr=False, use_vision=False),
                              meta, pdf_path=pdf)
    secs = {c.section for c in chunks if c.section}
    rate, missing = match_rate(secs, set(toc))
    return {"insurer": insurer, "pdf": Path(pdf).stem, "toc": len(toc),
            "sections": len(secs), "chunks": len(chunks), "score": rate,
            "missing_n": len(missing), "missing_sample": missing[:20]}


def main() -> None:
    done = set()
    if OUT.exists():
        done = {json.loads(l)["pdf"] for l in OUT.open() if l.strip()}
    for insurer, pdf in DOCS:
        stem = Path(pdf).stem
        if stem in done:
            print(f"skip {insurer}", flush=True)
            continue
        if not Path(pdf).exists():
            print(f"[없음] {pdf}", flush=True)
            continue
        try:
            row = score_one(insurer, pdf)
        except Exception as e:
            row = {"insurer": insurer, "pdf": stem, "error": str(e)[:300]}
        with OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        base = BASELINE.get(insurer)
        delta = f" (기준 {base} → {row.get('score')})" if base is not None else ""
        print(f"[완료] {insurer}: TOC {row.get('toc')} / 검출 {row.get('sections')} "
              f"→ {row.get('score')}{delta}", flush=True)
    summarize()


def summarize() -> None:
    if not OUT.exists():
        print("결과 없음")
        return
    rows = [json.loads(l) for l in OUT.open() if l.strip()]
    print(f"\n{'보험사':18}{'TOC':>6}{'검출':>6}{'점수':>8}{'기준':>8}{'변화':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['insurer'][:16]:18} ERROR {r['error'][:50]}")
            continue
        base = BASELINE.get(r["insurer"])
        d = f"{r['score'] - base:+.3f}" if base is not None else "-"
        print(f"{r['insurer'][:16]:18}{r['toc']:>6}{r['sections']:>6}"
              f"{r['score']:>8.3f}{(base if base is not None else 0):>8.3f}{d:>9}")
    ok = [r for r in rows if "error" not in r]
    if ok:
        avg = sum(r["score"] for r in ok) / len(ok)
        bavg = sum(BASELINE.get(r["insurer"], 0) for r in ok) / len(ok)
        print(f"{'평균':18}{'':>6}{'':>6}{avg:>8.3f}{bavg:>8.3f}{avg - bavg:>+9.3f}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarize()
    else:
        main()
