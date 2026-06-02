#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy-chunker CLI

  python -m policy_chunker <약관.pdf> <chunks.json> -o out.json
      [--target 700] [--hard-max 1400]

  <약관.pdf>    : 경계(제목 폰트) 자동 탐지에 사용
  <chunks.json> : 본문+표가 들어있는 결합 청크 (페이지 태그 #pN#NNNN 필요)

제목 폰트·보통약관 시작 페이지·상품명은 문서마다 자동 측정된다.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import run, detect
import fitz


def main(argv=None):
    ap = argparse.ArgumentParser(prog="policy-chunker",
                                 description="한국 보험약관 PDF → RAG 청크")
    ap.add_argument("pdf", help="약관 PDF (경계 탐지용)")
    ap.add_argument("chunks", nargs="?", help="결합 청크 JSON (본문+표)")
    ap.add_argument("-o", "--out", default="chunks_out.json", help="출력 JSON 경로")
    ap.add_argument("--target", type=int, default=700, help="목표 청크 크기(자)")
    ap.add_argument("--hard-max", type=int, default=1400, help="분할 한계(자)")
    ap.add_argument("--detect-only", action="store_true",
                    help="경계 자동 탐지값만 출력하고 종료")
    args = ap.parse_args(argv)

    if args.detect_only or not args.chunks:
        det = detect(fitz.open(args.pdf))
        print("[자동 탐지]")
        print(f"  본문 폰트      : {det.body_size}")
        print(f"  제목 폰트      : {det.title_size}  (인정범위 {det.title_lo}~{det.title_hi})")
        print(f"  보통약관 시작  : p{det.base_start_page}")
        print(f"  표지/목차 끝   : p{det.front_end_page}")
        print(f"  상품명         : {det.product}")
        if not args.chunks:
            return 0

    chunks = json.load(open(args.chunks, encoding="utf-8"))
    final, rep, det = run(args.pdf, chunks, target=args.target, hard_max=args.hard_max)
    json.dump(final, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n[탐지] 제목폰트={det.title_size} 보통약관시작=p{det.base_start_page} "
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
