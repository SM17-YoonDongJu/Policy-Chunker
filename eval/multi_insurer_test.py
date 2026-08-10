"""보험사별 대표 약관 1건씩 청킹해 일반화 수준을 측정 (이어받기 가능).

지금까지 검증은 사실상 메리츠화재 단일 보험사였다(batch_integrity 17문서 전부).
보험사가 바뀌면 특약 제목 조판 관습이 달라져 경계 검출이 무너질 수 있으므로
(KB 실측: 섹션 37개, art_nonmono 263), 보험사당 1건을 상시 관측한다.

실행: .venv/bin/python eval/multi_insurer_test.py
결과: eval/multi_insurer.jsonl (문서당 한 줄 append) + 요약표 출력

판정 핵심:
  boundary_collapse = gaps==0 && art_nonmono > max(30, 섹션수)
    → 여러 특약이 한 섹션으로 뭉치면 결번이 사라지는 대신 조번호 역행이 폭증한다.
      즉 gaps=0만 보면 "완벽"으로 오판한다(KB에서 실제로 그랬다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from eval.batch_integrity import check  # noqa: E402

OUT = Path("eval/multi_insurer.jsonl")

# (보험사, PDF 경로, 비고) — 보험사당 1건. 메리츠는 기존 게이트 문서를 그대로 쓴다.
DOCS = [
    ("메리츠화재", "in/상해보험_단체안심생활보험_30327.pdf", "기존 기준 문서(30327)"),
    ("KB손해보험", "in/KB_금쪽같은자녀보험Plus_26.04.pdf", "1250p, 접미사 없는 특약명"),
    ("현대해상", "in/현대_굿앤굿2040종합보험Hi2604.pdf", "종합보험"),
    ("삼성화재", "in/삼성_건강보험천만안심2601_2종.pdf", "건강보험 자동갱신형"),
    ("DB손해보험", "in/DB_프로미라이프참좋은더보장간병2601.pdf", "프로미라이프 간병"),
    ("DB손해보험(구동부)", "in/단체상해_빅히트_동부.pdf", "기존 게이트 문서"),
]


def main() -> None:
    done = set()
    if OUT.exists():
        done = {json.loads(l)["pdf"] for l in OUT.open() if l.strip()}

    for insurer, pdf, note in DOCS:
        stem = Path(pdf).stem
        if stem in done:
            print(f"skip {insurer}/{stem}", flush=True)
            continue
        if not Path(pdf).exists():
            print(f"[없음] {pdf}", flush=True)
            continue
        try:
            row = {"pdf": stem, "insurer": insurer, "note": note, **check(pdf)}
        except Exception as e:
            row = {"pdf": stem, "insurer": insurer, "note": note, "error": str(e)[:300]}
        with OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[완료] {insurer}: {json.dumps({k: v for k, v in row.items() if k not in ('worst_pages','note')}, ensure_ascii=False)}", flush=True)

    summarize()


def summarize() -> None:
    if not OUT.exists():
        return
    rows = [json.loads(l) for l in OUT.open() if l.strip()]
    print(f"\n{'보험사':22} {'섹션':>5} {'조갭':>5} {'nonmono':>8} {'복원율':>7} "
          f"{'cover':>7} {'ragged':>7}  붕괴")
    for r in rows:
        if "error" in r:
            print(f"{r['insurer'][:20]:22} ERROR {r['error'][:50]}")
            continue
        restore = (r["chunks"] and r.get("sections") is not None)
        flag = "⚠️ 붕괴" if r.get("boundary_collapse") else ""
        print(f"{r['insurer'][:20]:22} {r.get('sections','-'):>5} {r['gaps']:>5} "
              f"{r.get('art_nonmono','-'):>8} {'-':>7} "
              f"{r['coverage']:>7} {str(r.get('table_ragged','-')):>7}  {flag}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarize()
    else:
        main()
