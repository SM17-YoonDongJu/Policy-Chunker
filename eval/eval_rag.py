#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_rag.py — 덴스 임베딩 검색 평가 (Recall@k, MRR).

백엔드:
  --backend st     sentence-transformers (예: BAAI/bge-m3, intfloat/multilingual-e5-large)
  --backend openai OpenAI 임베딩 (text-embedding-3-small 등, OPENAI_API_KEY 필요)

사용:
  pip install numpy sentence-transformers
  python eval/eval_rag.py --chunks out.json --eval eval/eval_set.example.jsonl \
      --backend st --model BAAI/bge-m3 --k 1 3 5 10

평가셋 형식·정답 판정은 eval_bm25.py와 동일(gold_ids / gold_contains).
하이브리드 효과를 보려면 BM25 결과와 나란히 비교하라.
"""
from __future__ import annotations

import argparse
import json
import collections

import numpy as np

from eval_bm25 import load_eval, is_hit  # noqa: E402  (같은 폴더에서 실행)


def embed_st(texts, model_name, is_query=False):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    # e5 계열은 query:/passage: 접두사를 권장
    if "e5" in model_name.lower():
        texts = [("query: " if is_query else "passage: ") + t for t in texts]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                       batch_size=64)
    return np.asarray(emb, dtype=np.float32)


def embed_openai(texts, model_name):
    from openai import OpenAI
    client = OpenAI()
    out = []
    for i in range(0, len(texts), 256):
        batch = texts[i:i + 256]
        resp = client.embeddings.create(model=model_name, input=batch)
        out.extend(d.embedding for d in resp.data)
    arr = np.asarray(out, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--backend", choices=["st", "openai"], default="st")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    chunks = json.load(open(args.chunks, encoding="utf-8"))
    texts = [c.get("text", "") for c in chunks]
    rows = load_eval(args.eval)
    queries = [r["q"] for r in rows]

    if args.backend == "st":
        doc_emb = embed_st(texts, args.model, is_query=False)
        q_emb = embed_st(queries, args.model, is_query=True)
    else:
        doc_emb = embed_openai(texts, args.model)
        q_emb = embed_openai(queries, args.model)

    sims = q_emb @ doc_emb.T  # 정규화돼 있으므로 코사인
    maxk = max(args.k)

    recall = {k: 0 for k in args.k}
    rr_sum = 0.0
    by_type = collections.defaultdict(lambda: [0, 0])
    misses = []

    for qi, row in enumerate(rows):
        ranked = np.argsort(-sims[qi])[:maxk]
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
            misses.append((row, [chunks[i].get("header", "") for i in ranked[:3]]))

    n = len(rows)
    print(f"== Dense [{args.backend}:{args.model}] ==  ({n} questions)")
    for k in args.k:
        print(f"  Recall@{k}: {recall[k] / n:.2f}  ({recall[k]}/{n})")
    print(f"  MRR@{maxk}: {rr_sum / n:.2f}")
    print("  유형별 Recall@%d:" % maxk)
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
