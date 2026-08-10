"""보험사별 약관 1건씩 교차 검증 — 파이프라인이 메리츠 전용인지 일반화되는지.

기존 검증 범위는 사실상 메리츠화재 한 곳이었다(batch_integrity 17문서 전부 메리츠,
검색 평가 47문항도 메리츠 30327 단일). KB 1건 테스트(`KB_GENERALIZATION_TEST.md`)에서
경계 검출 붕괴가 확인됐으므로, 나머지 보험사로 범위를 넓혀 **붕괴가 KB 특유인지
타사 전반인지**를 가른다.

문서 선정: 기준 문서(메리츠 단체안심생활보험 337p)와 비교 가능하도록 보험사별
**단체상해 계열**을 우선 선택. 노션 '보험 약관 파일' DB에서 `eval/notion_download.py`로 수집.

    .venv/bin/python eval/cross_insurer_test.py          # 이어받기 가능
    .venv/bin/python eval/cross_insurer_test.py --report # 결과만 재출력

지표 해석은 batch_integrity.py 참조. 특히 `boundary_collapse`는
"gaps=0 + art_nonmono 폭증" 시그니처 — 여러 특약이 한 섹션으로 뭉쳐 결번이 사라진
상태라, gaps=0을 양호 신호로 읽으면 오판한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.batch_integrity import check  # noqa: E402

OUT = Path(__file__).resolve().parent / "cross_insurer.jsonl"

# (보험사, 파일stem, 비고)
DOCS = [
    ("메리츠화재", "상해보험_단체안심생활보험_30327", "기준 문서 337p"),
    ("삼성화재", "삼성_신종단체상해보험II_2607", "단체상해 1251p"),
    ("현대해상", "현대_마음플러스상해종합_2601", "상해종합 583p"),
    ("DB손해보험", "DB_브라보단체보험_2604", "단체보험 338p"),
    ("KB손해보험", "KB_안심경영단체상해_2601", "단체상해 306p"),
    ("KB손해보험", "KB_금쪽같은자녀보험Plus_26.04", "자녀보험 1250p (기존 테스트분)"),
]


def load() -> dict[str, dict]:
    return {json.loads(l)["pdf"]: json.loads(l) for l in OUT.open()} if OUT.exists() else {}


def run() -> None:
    done = load()
    for insurer, stem, note in DOCS:
        if stem in done:
            print(f"skip {stem}", flush=True)
            continue
        pdf = f"in/{stem}.pdf"
        if not Path(pdf).exists():
            print(f"[없음] {pdf}", flush=True)
            continue
        print(f"→ {insurer} / {stem} ({note})", flush=True)
        try:
            row = {"pdf": stem, "insurer": insurer, "note": note, **check(pdf)}
        except Exception as e:  # noqa: BLE001
            row = {"pdf": stem, "insurer": insurer, "note": note, "error": str(e)[:300]}
        with OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "worst_pages"},
                         ensure_ascii=False), flush=True)


def report() -> None:
    rows = load()
    L = ["# 보험사별 약관 교차 검증", "",
         "기존 검증은 사실상 메리츠화재 단일 보험사였다. 보험사별 1건씩으로 범위를 넓혀",
         "청킹 파이프라인이 특정 보험사 조판에 맞춰진 것인지 확인한다.", "",
         "| 보험사 | 문서 | 청크 | 섹션 | coverage | gaps | art_nonmono | 경계붕괴 | table_ragged | table_recall |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for insurer, stem, _ in DOCS:
        r = rows.get(stem)
        if not r:
            L.append(f"| {insurer} | {stem} | — | — | (미실행) | | | | | |")
            continue
        if r.get("error"):
            L.append(f"| {insurer} | {stem} | — | — | 오류: {r['error'][:60]} | | | | | |")
            continue
        cov = f"{r['coverage']:.3f}" if r.get("coverage") is not None else "—"
        tr = f"{r['table_ragged']:.3f}" if r.get("table_ragged") is not None else "—"
        pr = f"{r['table_page_recall']:.3f}" if r.get("table_page_recall") is not None else "—"
        L.append(f"| {insurer} | {stem} | {r['chunks']} | {r['sections']} | {cov} | "
                 f"{r['gaps']} | {r['art_nonmono']} | "
                 f"{'**붕괴**' if r.get('boundary_collapse') else 'ok'} | {tr} | {pr} |")
    out = OUT.with_suffix(".md")
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        run()
    report()
