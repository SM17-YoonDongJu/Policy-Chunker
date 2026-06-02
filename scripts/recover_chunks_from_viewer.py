#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_chunks_from_viewer.py — 뷰어 HTML에 내장된 청크(const DATA=[...])를
표준 청크 JSON으로 복원한다. (스크립트는 잃어버렸지만 결과물만 HTML로 남았을 때)

축약 필드 매핑:
  h→header b→body ps→page_start pe→page_end sec→section sk→section_kind
  yak→yakwan art→article_no ct→chunk_type tbl→is_table cl→char_len ts→table_source

사용:
  python scripts/recover_chunks_from_viewer.py 약관청크_뷰어.html -o data/recovered.json
"""
import argparse
import json
import re

FIELD = {
    "h": "header", "b": "body", "ps": "page_start", "pe": "page_end",
    "sec": "section", "sk": "section_kind", "yak": "yakwan", "art": "article_no",
    "ct": "chunk_type", "tbl": "is_table", "cl": "char_len", "ts": "table_source",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("-o", "--out", default="recovered_chunks.json")
    args = ap.parse_args()

    html = open(args.html, encoding="utf-8").read()
    m = re.search(r"const DATA\s*=\s*(\[.*?\])\s*[;\n]", html, re.S)
    if not m:
        raise SystemExit("뷰어에서 const DATA 배열을 찾지 못했습니다.")
    data = json.loads(m.group(1))

    out = []
    for i, d in enumerate(data, 1):
        c = {FIELD.get(k, k): v for k, v in d.items()}
        c.setdefault("header", "")
        c["text"] = (c.get("header", "") + "\n" + c.get("body", "")).strip()
        c["chunk_id"] = f"recovered#rc{i:04d}"
        out.append(c)

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"복원: {len(out)}청크 → {args.out}")


if __name__ == "__main__":
    main()
