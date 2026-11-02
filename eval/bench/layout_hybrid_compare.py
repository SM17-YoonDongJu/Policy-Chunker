"""레이아웃 모델 크기 비교(문서 여러 건) — V3(운영) 대 S, 그리고 'S + 조건부 V3 재실행' 절충.

smoke 1건에서 S는 V3보다 22배 빨랐지만 표 박스 재현율 86%였고, CPU 소스(PyMuPDF)도 없는 쪽의 표
3개를 잃었다. 절충(S가 표를 못 찾았고 괘선이 많고 PyMuPDF도 표가 없는 쪽만 V3로 다시)은 그 3개를
모두 되찾았다. 문서 한 건 결과라 core 문서셋으로 넓혀 본다. VLM은 부르지 않는다(CPU만).

  python eval/bench/layout_hybrid_compare.py --pdfs a.pdf b.pdf ... --out /tmp/hybrid.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_probe import boxes_of, iou, render  # noqa: E402

DRAWINGS_MIN = 25


def pymupdf_table_pages(doc) -> set[int]:
    out = set()
    for p in doc:
        try:
            if p.find_tables().tables:
                out.add(p.number + 1)
        except Exception:  # noqa: BLE001 - 한 쪽의 실패가 비교를 막지 않게
            pass
    return out


def detect(model, doc, pages) -> tuple[dict[int, list], float]:
    t0 = time.perf_counter()
    out = {p: boxes_of(next(iter(model.predict(render(doc[p - 1], 150), batch_size=1))), 150)
           for p in pages}
    return out, time.perf_counter() - t0


def recall(base: dict, pred: dict) -> tuple[int, int, set[int]]:
    hit = tot = 0
    lost: set[int] = set()
    for p, bb in base.items():
        for b in bb:
            tot += 1
            if any(iou(b, o) >= 0.8 for o in pred.get(p, [])):
                hit += 1
            else:
                lost.add(p)
    return hit, tot, lost


def main() -> None:
    import pymupdf
    from paddleocr import LayoutDetection

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/hybrid.json")
    args = ap.parse_args()

    v3 = LayoutDetection(model_name="PP-DocLayoutV3", device="cpu")
    s = LayoutDetection(model_name="PP-DocLayout-S", device="cpu")
    rows = []
    for path in args.pdfs:
        doc = pymupdf.open(path)
        # 백지 제외(운영과 같게)
        pages = [p.number + 1 for p in doc
                 if p.get_text("words") or p.get_images() or p.get_drawings()]
        detect(v3, doc, pages[:1])
        detect(s, doc, pages[:1])                                           # 워밍업
        base, v3_s = detect(v3, doc, pages)
        base = {p: v for p, v in base.items() if v}
        sb, s_s = detect(s, doc, pages)
        pym = pymupdf_table_pages(doc)
        nd = {p: len(doc[p - 1].get_drawings()) for p in pages}
        redo = [p for p in pages if not sb[p] and nd[p] >= DRAWINGS_MIN and p not in pym]
        hyb = dict(sb)
        for p in redo:
            hyb[p] = base.get(p, [])  # V3 재실행 = 기준과 같은 모델·설정
        h_s, tot, lost_s = recall(base, sb)
        h_h, _, lost_h = recall(base, hyb)
        row = {
            "pdf": Path(path).name[:16], "pages": len(pages),
            "table_pages": len(base), "boxes": tot,
            "v3_sec": round(v3_s, 1), "s_sec": round(s_s, 1),
            "hybrid_sec": round(s_s + v3_s / len(pages) * len(redo), 1), "redo_pages": len(redo),
            "s_recall": round(h_s / tot, 3) if tot else 1.0,
            "s_uncovered": sorted(lost_s - pym),
            "hybrid_recall": round(h_h / tot, 3) if tot else 1.0,
            "hybrid_uncovered": sorted(lost_h - pym),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    tot_v3 = sum(r["v3_sec"] for r in rows)
    redo = sum(r["redo_pages"] for r in rows)
    print(f"\n합계: V3 {tot_v3:.0f}s, S {sum(r['s_sec'] for r in rows):.0f}s, "
          f"절충 {sum(r['hybrid_sec'] for r in rows):.0f}s (V3 재실행 {redo}쪽)")
    print(f"CPU 소스 없이 잃는 표 쪽 — S {sum(len(r['s_uncovered']) for r in rows)}, "
          f"절충 {sum(len(r['hybrid_uncovered']) for r in rows)}")
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
