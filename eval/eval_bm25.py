#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BM25 키워드 검색 베이스라인 평가.
bm25와 hybrid search 비교 위한 베이스라인 코드
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re


def tokenize(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"[가-힣]+|[a-z]+|\d+", text)
    toks = []
    for w in words:
        toks.append(w)
        if len(w) >= 3:
            toks += [w[i:i + 2] for i in range(len(w) - 1)]
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = sum(self.dl) / max(self.N, 1)
        df = collections.Counter()
        for d in docs:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [collections.Counter(d) for d in docs]

    def search(self, query: str, topn=20) -> list[int]:
        q = tokenize(query)
        scores = [0.0] * self.N
        for t in q:
            idf = self.idf.get(t)
            if not idf:
                continue
            for i in range(self.N):
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:topn]


def load_eval(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def is_hit(chunk: dict, row: dict) -> bool:
    if "gold_ids" in row:
        return chunk.get("chunk_id") in set(row["gold_ids"])
    if "gold_contains" in row:
        return any(g in chunk.get("content", "") for g in row["gold_contains"])
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    chunks = json.load(open(args.chunks, encoding="utf-8"))
    docs = [tokenize(c.get("content", "")) for c in chunks]
    bm = BM25(docs)
    rows = load_eval(args.eval)
    maxk = max(args.k)

    recall = {k: 0 for k in args.k}
    rr_sum = 0.0
    by_type = collections.defaultdict(lambda: [0, 0])
    misses = []

    for row in rows:
        ranked = bm.search(row["q"], topn=maxk)
        hit_rank = next((r + 1 for r, idx in enumerate(ranked) if is_hit(chunks[idx], row)), None)
        for k in args.k:
            if hit_rank and hit_rank <= k:
                recall[k] += 1
        rr_sum += (1.0 / hit_rank) if hit_rank else 0.0
        t = row.get("type", "기타")
        by_type[t][1] += 1
        if hit_rank and hit_rank <= maxk:
            by_type[t][0] += 1
        if not hit_rank or hit_rank > 5:
            misses.append((row, [chunks[i].get("header", chunks[i].get("chunk_id", "")) for i in ranked[:3]]))

    n = len(rows)
    print(f"== BM25 baseline ==  ({n} questions)")
    for k in args.k:
        print(f"  Recall@{k}: {recall[k] / n:.2f}  ({recall[k]}/{n})")
    print(f"  MRR@{maxk}: {rr_sum / n:.2f}")
    print(f"  유형별 Recall@{maxk}:")
    for t, (h, tot) in sorted(by_type.items()):
        print(f"    {t:<8} {h / tot:.2f}  ({h}/{tot})")
    if args.show_misses and misses:
        print("\n  놓친 질문(top-3):")
        for row, top3 in misses:
            print(f"    Q: {row['q']}")
            for h in top3:
                print(f"       - {h}")


if __name__ == "__main__":
    main()
