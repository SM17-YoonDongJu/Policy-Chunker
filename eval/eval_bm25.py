#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_bm25.py — 키워드(BM25) 검색 베이스라인.

의존성 없이 동작(순수 파이썬 BM25). 덴스 임베딩과 비교할 기준선이다.
키워드 검색이 약한 지점(동의어·구어체·표 내용)이 곧 덴스/하이브리드가 필요한 지점.

사용:
  python eval/eval_bm25.py --chunks out.json --eval eval/eval_set.example.jsonl --k 1 3 5 10

평가셋(jsonl) 한 줄 형식:
  {"q": "질문", "type": "표|면책|특약|절차|사실조회", "gold_ids": ["...#rc0123"]}
  또는 {"q": "...", "gold_contains": ["정답에 꼭 들어갈 단어", ...]}
"""
from __future__ import annotations

import argparse
import json
import math
import re
import collections


def tokenize(text: str) -> list[str]:
    """한국어/숫자/영문 토큰 + 2-gram 보강(어미 변화에 약간 강건)."""
    text = text.lower()
    words = re.findall(r"[가-힣]+|[a-z]+|\d+", text)
    toks = []
    for w in words:
        toks.append(w)
        if len(w) >= 3:  # 한글 단어는 2-gram도 추가
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
        ranked = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        return ranked[:topn]


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
        text = chunk.get("text", "")
        return any(g in text for g in row["gold_contains"])
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    chunks = json.load(open(args.chunks, encoding="utf-8"))
    docs = [tokenize(c.get("text", "")) for c in chunks]
    bm = BM25(docs)
    rows = load_eval(args.eval)
    maxk = max(args.k)

    recall = {k: 0 for k in args.k}
    rr_sum = 0.0
    by_type = collections.defaultdict(lambda: [0, 0])  # type -> [hit@max, total]
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
        if hit_rank and hit_rank <= max(args.k):
            by_type[t][0] += 1
        if not hit_rank or hit_rank > 5:
            misses.append((row, [chunks[i].get("header", "") for i in ranked[:3]]))

    n = len(rows)
    print(f"== BM25 baseline ==  ({n} questions)")
    for k in args.k:
        print(f"  Recall@{k}: {recall[k] / n:.2f}  ({recall[k]}/{n})")
    print(f"  MRR@{maxk}: {rr_sum / n:.2f}")
    print("  유형별 Recall@%d:" % max(args.k))
    for t, (h, tot) in sorted(by_type.items()):
        print(f"    {t:<8} {h / tot:.2f}  ({h}/{tot})")
    if args.show_misses and misses:
        print("\n  놓친 질문(top-3 헤더):")
        for row, top3 in misses:
            print(f"    Q: {row['q']}")
            for h in top3:
                print(f"       - {h}")


if __name__ == "__main__":
    main()
