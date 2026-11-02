"""레이아웃 단계(PP-DocLayout, CPU) 가속 실험 — 설정별 쪽당 시간과 기준 대비 표 박스 일치.

Paddle 전환 뒤 VLM(0.9B, 표 영역 단위)이 빨라져 병목이 레이아웃으로 넘어갔다. smoke 문서(296쪽)에서
레이아웃만 613초였고 쪽당 p50 1.9초, CPU를 사실상 한 코어만 썼다(2026-11-02, M5 Pro). 이 스크립트는
레이아웃의 손잡이(스레드·배치·해상도·모델 크기)를 바꿔 가며 시간과 결과를 같은 쪽 표본에서 잰다.

기준은 운영 설정(PP-DocLayoutV3, 배치 1, 150DPI, 스레드 기본)으로 같은 표본을 돌린 결과다. 일치는
쪽 단위로 본다 — 기준의 표 박스 하나하나에 IoU ≥ 0.8인 박스가 있으면 일치, 없으면 놓침. 기준에
없는 박스는 오탐으로 센다.

  python eval/bench/layout_probe.py --pdf X.pdf --pages 60 --configs base,t4,t4b4,dpi120,S
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

CONFIGS = {
    # 이름: (모델, 스레드, 배치, DPI)
    "base": ("PP-DocLayoutV3", None, 1, 150),
    "t4": ("PP-DocLayoutV3", 4, 1, 150),
    "t8": ("PP-DocLayoutV3", 8, 1, 150),
    "t4b4": ("PP-DocLayoutV3", 4, 4, 150),
    "t4b8": ("PP-DocLayoutV3", 4, 8, 150),
    "dpi120": ("PP-DocLayoutV3", 4, 1, 120),
    "dpi100": ("PP-DocLayoutV3", 4, 1, 100),
    "M": ("PP-DocLayout-M", 4, 1, 150),
    "S": ("PP-DocLayout-S", 4, 1, 150),
}


def render(page, dpi: int):
    import numpy as np
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n > 3:
        img = img[:, :, :3]
    return img[:, :, ::-1].copy()


def boxes_of(res, dpi: int, min_score: float = 0.5) -> list[list[float]]:
    data = res.json["res"] if hasattr(res, "json") else res
    scale = 72.0 / dpi
    return [[round(float(v) * scale, 1) for v in b["coordinate"]]
            for b in data.get("boxes", [])
            if b.get("label") == "table" and float(b.get("score", 0)) >= min_score]


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def run(pdf: str, pages: list[int], name: str) -> dict:
    import pymupdf
    from paddleocr import LayoutDetection

    model_name, threads, batch, dpi = CONFIGS[name]
    kw = {"model_name": model_name, "device": "cpu"}
    if threads:
        kw["cpu_threads"] = threads
    t_load = time.perf_counter()
    model = LayoutDetection(**kw)
    load_s = time.perf_counter() - t_load
    doc = pymupdf.open(pdf)
    imgs = [render(doc[p - 1], dpi) for p in pages]   # 렌더는 시간에서 뺀다(설정 간 공통)
    model.predict(imgs[0], batch_size=1)               # 첫 호출 워밍업(그래프 준비)
    t0 = time.perf_counter()
    out = {}
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        res = list(model.predict(chunk if batch > 1 else chunk[0], batch_size=batch))
        for p, r in zip(pages[i:i + batch], res):
            out[p] = boxes_of(r, dpi)
    sec = time.perf_counter() - t0
    return {"config": name, "model": model_name, "threads": threads, "batch": batch, "dpi": dpi,
            "load_s": round(load_s, 1), "sec": round(sec, 1),
            "sec_per_page": round(sec / len(pages), 3), "boxes": out}


def agree(base: dict, other: dict) -> dict:
    hit = miss = extra = 0
    miss_pages = []
    for p, bb in base.items():
        ob = other.get(p, [])
        used = set()
        for b in bb:
            j = next((k for k, o in enumerate(ob) if k not in used and iou(b, o) >= 0.8), None)
            if j is None:
                miss += 1
                miss_pages.append(p)
            else:
                hit += 1
                used.add(j)
        extra += len(ob) - len(used)
    for p, ob in other.items():
        if p not in base:
            extra += len(ob)
    total = hit + miss
    return {"box_recall": round(hit / total, 3) if total else 1.0, "missed": miss,
            "extra": extra, "missed_pages": sorted(set(miss_pages))[:10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--layout-json", required=True,
                    help="stage_runner layout 산출물 — 표 쪽/비표 쪽 표본을 고르는 데만 쓴다")
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--configs", default="base,t4")
    ap.add_argument("--out", default="/tmp/layout_probe.json")
    args = ap.parse_args()

    import pymupdf
    n = pymupdf.open(args.pdf).page_count
    table_pages = sorted(int(p) for p in
                         json.loads(Path(args.layout_json).read_text())["output"]["boxes"])
    other = [p for p in range(1, n + 1) if p not in set(table_pages)]
    rnd = random.Random(7)
    half = args.pages // 2
    pages = sorted(rnd.sample(table_pages, min(half, len(table_pages)))
                   + rnd.sample(other, min(half, len(other))))

    results = []
    for name in args.configs.split(","):
        r = run(args.pdf, pages, name)
        results.append(r)
        print(f"{name:<7} {r['model']:<16} threads={r['threads']} batch={r['batch']} "
              f"dpi={r['dpi']}  {r['sec_per_page']}s/쪽 (load {r['load_s']}s)", flush=True)
    base = next((r for r in results if r["config"] == "base"), results[0])
    print(f"\n표본 {len(pages)}쪽(표 쪽 {sum(1 for p in pages if p in table_pages)}) — "
          f"기준 {base['config']}: 표 박스 {sum(len(v) for v in base['boxes'].values())}개")
    for r in results:
        a = agree({p: v for p, v in base["boxes"].items() if v}, r["boxes"])
        speed = base["sec_per_page"] / r["sec_per_page"] if r["sec_per_page"] else 0
        r["agree"] = a
        print(f"  {r['config']:<7} ×{speed:4.1f}  recall {a['box_recall']:.3f}  "
              f"놓침 {a['missed']} 오탐 {a['extra']}  {a['missed_pages']}")
    Path(args.out).write_text(json.dumps({"pages": pages, "results": results},
                                         ensure_ascii=False))


if __name__ == "__main__":
    main()
