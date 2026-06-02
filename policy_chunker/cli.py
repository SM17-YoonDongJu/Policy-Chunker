#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy-chunker CLI

  # 1) PDF에서 바로 추출까지 (권장)
  python -m policy_chunker <약관.pdf> --extract -o out.json
  python -m policy_chunker <약관.pdf> --extract --vlm -o out.json   # 표를 Claude CLI 비전으로

  # 2) 이미 추출된 결합 청크(JSON)를 입력으로
  python -m policy_chunker <약관.pdf> <chunks.json> -o out.json

  # 경계 자동 탐지값만 보기
  python -m policy_chunker <약관.pdf> --detect-only

<약관.pdf>    : 경계(제목 폰트) 자동 탐지 + (--extract 시) 본문/표 추출에 사용
<chunks.json> : 본문+표가 들어있는 결합 청크 (페이지 태그 #pN#NNNN 필요)

제목 폰트·보통약관 시작 페이지·상품명은 문서마다 자동 측정된다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import fitz

from . import run, detect, find, assess, extract


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="policy-chunker",
                                 description="한국 보험약관 PDF → RAG 청크")
    ap.add_argument("pdf", help="약관 PDF (경계 탐지 + 추출용)")
    ap.add_argument("chunks", nargs="?", help="결합 청크 JSON (생략 시 --extract 필요)")
    ap.add_argument("-o", "--out", default=None,
                    help="출력 JSON 경로 (기본: outputs/<PDF이름>_chunks.json)")
    ap.add_argument("--target", type=int, default=700, help="목표 청크 크기(자)")
    ap.add_argument("--hard-max", type=int, default=1400, help="분할 한계(자)")
    ap.add_argument("--detect-only", action="store_true",
                    help="경계 자동 탐지값만 출력하고 종료")
    # --- 추출 단계 ---
    ap.add_argument("--extract", action="store_true",
                    help="PDF에서 본문+표를 직접 추출(chunks 인자 불필요)")
    ap.add_argument("--vlm", action="store_true",
                    help="표를 Claude CLI(비전)로 추출 (API 키 불필요)")
    ap.add_argument("--claude-bin", default="claude", help="claude 실행 파일")
    ap.add_argument("--dpi", type=int, default=200,
                    help="VLM용 페이지 렌더 DPI (작은 글씨 표는 300 권장)")
    ap.add_argument("--vlm-timeout", type=int, default=180, help="VLM 페이지당 타임아웃(초)")
    ap.add_argument("--save-combined", help="추출한 결합 청크를 따로 저장할 경로")
    ap.add_argument("--force", action="store_true",
                    help="자동탐지 신뢰도가 낮아도(🔴) 강행")
    args = ap.parse_args(argv)

    # 결과물은 기본적으로 outputs/ 폴더에 둔다.
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.pdf))[0]
        args.out = os.path.join("outputs", f"{stem}_chunks.json")

    doc = fitz.open(args.pdf)
    bounds, det = find(doc)
    level, reasons = assess(doc, det, bounds)
    flag = "🟢 ok" if level == "ok" else "🔴 weak"

    if args.detect_only:
        print("[자동 탐지]")
        print(f"  본문 폰트      : {det.body_size}")
        print(f"  제목 폰트      : {det.title_size}  (근거 {det.title_evidence}줄, "
              f"인정범위 {det.title_lo}~{det.title_hi})")
        print(f"  보통약관 시작  : {'p%d' % det.base_start_page if det.base_start_page else '못 찾음'}")
        print(f"  표지/목차 끝   : {'p%d' % det.front_end_page if det.front_end_page else '-'}")
        print(f"  상품명         : {det.product}")
        print(f"[신뢰도] {flag}")
        for r in reasons:
            print(f"  - {r}")
        return 0

    # 신뢰도 게이트: 약하면 조용히 틀린 결과를 내지 않고 멈춘다(--force로 강행).
    print(f"[신뢰도] {flag}")
    for r in reasons:
        print(f"  - {r}")
    if level == "weak" and not args.force:
        print("\n⚠️  이 문서는 조문·특약 구조가 약해 자동 경계를 신뢰하기 어렵습니다.")
        print("    (스캔본이거나, 조문 대신 표 중심으로 구성된 약관일 수 있음)")
        print("    그래도 돌리려면 --force, 또는 경계만 보려면 --detect-only.")
        return 3

    # 입력 청크 확보: --extract 면 PDF에서 추출, 아니면 chunks.json 로드
    if args.extract:
        def prog(i, n, p):
            print(f"  [VLM] {i}/{n}  p{p}", file=sys.stderr)
        chunks, tsrc = extract(
            args.pdf, use_vlm=args.vlm, claude_bin=args.claude_bin,
            dpi=args.dpi, vlm_timeout=args.vlm_timeout,
            progress=prog if args.vlm else None)
        print(f"[추출] 결합청크 {len(chunks)}개  "
              f"표도구={','.join(f'{k}:{len(v)}p' for k, v in tsrc.items())}")
        if args.save_combined:
            _ensure_dir(args.save_combined)
            json.dump(chunks, open(args.save_combined, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
    elif args.chunks:
        chunks = json.load(open(args.chunks, encoding="utf-8"))
    else:
        ap.error("chunks 인자가 없으면 --extract 를 줘야 합니다")
        return 2

    final, rep, det = run(args.pdf, chunks, target=args.target, hard_max=args.hard_max)
    _ensure_dir(args.out)
    json.dump(final, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    base_start = f"p{det.base_start_page}" if det.base_start_page else "없음"
    print(f"\n[탐지] 제목폰트={det.title_size} 보통약관시작={base_start} "
          f"상품명={det.product}")
    print(f"[경계] 약관 {rep['n_yakwan_bounds']} / 별표 {rep['n_byeolpyo_bounds']}")
    print(f"[청크] {rep['n_chunks']}개  "
          f"med/mean/max={rep['char_median']}/{rep['char_mean']}/{rep['char_max']}자  "
          f"<50자:{rep['under_50']}")
    print(f"[품질] 고유약관 {rep['n_unique_yakwan']}  별표청크 {rep['n_byeolpyo_chunks']}  "
          f"푸터잔존 {rep['footer_left']}  제47조누수 {rep['art47_leak']}")
    print(f"[출력] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
