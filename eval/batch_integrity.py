"""여러 문서에 원문 커버리지 + 조 연속성 + 중복 + 표 구조/조 헤딩 검사를 일괄 실행.

실행: .venv/bin/python eval/batch_integrity.py            # 17문서 배치 (이어받기 가능)
      .venv/bin/python eval/batch_integrity.py --vlm <pdf>  # VLM 표 소스 승률 단건 측정
결과: eval/batch_integrity.jsonl (문서당 한 줄 append)

지표 (OHRBench 대조 — eval/GAP_ANALYSIS.md §3):
  coverage      원문 유실률. norm()이 파이프·공백을 지우므로 표 구조 오류는 못 잡음.
                주의: 호 분할(RECHUNK_ENUM_SPLIT) 후에는 분할 경계를 가로지르는
                프로브가 false-missing이 되어 5~8%p 낮게 나온다(실제 유실 아님) —
                코드 버전이 같은 실행끼리만 비교할 것
  gaps          section별 조번호 결번 수
  dup           인접 청크 동일 본문 수
  table_ragged  표 청크 중 행별 열 개수가 어긋난(구조 붕괴 의심) 비율  ← 표 오류 사각 보완
  table_page_recall  원문 표 페이지 중 표 청크가 존재하는 페이지 비율
  art_nonmono   section 내 조번호 역행/비정상 점프(>10) 수 — 타법령 인용 오인 감시
  cite_runs     "인용 조문"으로 병합된 청크 수 (참고용)
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
                    "section": c.section, "article_title": c.article_title} for c in chunks]
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

    # 표 구조 충실도: 파이프 행의 열 개수가 행마다 어긋나면 정렬 붕괴 의심.
    # coverage의 norm()이 파이프를 지워 표 붕괴에 무감각한 사각을 보완한다.
    table_chunks = ragged = 0
    table_chunk_pages: set[int] = set()
    for c in chunk_dicts:
        rows = [r for r in c["content"].splitlines() if r.lstrip().startswith("|")]
        if len(rows) < 2:
            continue
        table_chunks += 1
        table_chunk_pages.add(c["page"])
        widths = [r.count("|") for r in rows]
        if max(widths) - min(widths) > 1:
            ragged += 1
    src_table_pages = {p.number + 1 for p in doc
                       if p.number + 1 >= first_page and p.find_tables().tables}
    page_recall = (round(len(src_table_pages & table_chunk_pages) / len(src_table_pages), 3)
                   if src_table_pages else None)

    # 조 헤딩 오인 감시: 등장순 조번호의 역행/큰 점프는 타법령 인용을 조로
    # 오인했거나 경계가 샌 신호다 (결번(gaps)과 달리 "그럴듯한 오번호"를 잡는다).
    seq_by_sec = defaultdict(list)
    for c in chunk_dicts:
        if c["article"] and c["section"]:
            m = re.match(r"제(\d+)조", c["article"])
            if m:
                n = int(m.group(1))
                if not seq_by_sec[c["section"]] or seq_by_sec[c["section"]][-1] != n:
                    seq_by_sec[c["section"]].append(n)
    nonmono = sum(1 for seq in seq_by_sec.values()
                  for prev, cur in zip(seq, seq[1:]) if cur < prev or cur - prev > 10)
    cite_runs = sum(1 for c in chunk_dicts if c.get("article_title") == "인용 조문")

    # 경계 붕괴 시그니처: 여러 특약이 한 섹션으로 뭉치면 1~N조가 모두 들어차
    # 결번(gaps)이 사라지는 대신 조번호 역행(art_nonmono)이 폭증한다. 즉 gaps=0은
    # 양호 신호가 아닐 수 있다 — 두 지표를 함께 읽어야 오판하지 않는다(KB 실측).
    n_sec = len({c["section"] for c in chunk_dicts if c["section"]})
    collapse = gaps == 0 and nonmono > max(30, n_sec)

    return {
        "coverage": round(1 - missing_probes / total_probes, 4) if total_probes else None,
        "gaps": gaps, "chunks": len(chunk_dicts), "dup": dup,
        "sections": n_sec, "boundary_collapse": collapse,
        "table_chunks": table_chunks,
        "table_ragged": round(ragged / table_chunks, 3) if table_chunks else None,
        "table_page_recall": page_recall,
        "art_nonmono": nonmono, "cite_runs": cite_runs,
        "worst_pages": worst_pages[:5],
    }


def vlm_winrate(pdf_path: str) -> dict:
    """VLM(Surya) vs pymupdf 표 소스 승률 단건 측정. 배치와 분리 — VLM 비용이 크다.

    OHRBench의 "파서마다 표 품질 편차" 경고에 대응해, best_table 선택에서
    VLM이 실제로 이기는 비율과 정렬 개선폭(더블스페이스 감소)을 수치화한다.
    """
    from insurance_chunker.combine import _double_spaces, select_best_tables
    from insurance_chunker.extractor import extract_tables_for_doc

    with fitz.open(pdf_path) as doc:
        pages = list(range(1, doc.page_count + 1))
    srcs = extract_tables_for_doc(pdf_path, pages, use_vision=True)
    best = select_best_tables(srcs)
    if not best:
        return {"pdf": Path(pdf_path).stem, "table_pages": 0}
    vlm_wins = sum(1 for _, s in best.values() if s == "vlm")
    deltas = []
    for pg, (md, _) in best.items():
        pm = srcs.get("pymupdf", {}).get(pg)
        if pm:
            deltas.append(_double_spaces(pm) - _double_spaces(md))
    return {
        "pdf": Path(pdf_path).stem, "table_pages": len(best),
        "vlm_win_rate": round(vlm_wins / len(best), 3),
        "dspace_gain_avg": round(sum(deltas) / len(deltas), 1) if deltas else None,
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
    if len(sys.argv) > 2 and sys.argv[1] == "--vlm":
        print(json.dumps(vlm_winrate(sys.argv[2]), ensure_ascii=False))
    else:
        main()
