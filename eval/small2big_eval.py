"""Small-to-Big 검색 평가 — 자식(항/호 단위)으로 찾고 부모(조 단위)로 반환.

자식 인덱스: v3 부모를 항(①②…)/호(1. 2.) 마커로 분할, parent_index 연결.
검색: 자식 BM25+임베딩 RRF → 부모 승격(dedup) → 부모 top-20 리랭크.
비교: v3 rerank(기존 최고) vs small2big.

실행: RERANK=1 .venv/bin/python eval/small2big_eval.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RERANK", "1")

from insurance_chunker.tokenizer import tokenize_korean  # noqa: E402
from eval.retrieval_eval import (  # noqa: E402
    BM25, EVAL_DIR, QUERY_INSTRUCT, QUESTIONS, TOP_K, VERSIONS,
    _reranker, build_embed_cache, is_hit, load_jsonl, metrics, rrf,
)

# 항 ①-⑳ / 줄 시작 호 "1." — 자식 분할 지점
_CLAUSE = re.compile(r"(?=①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|⑯|⑰|⑱|⑲|⑳)|(?=^\d{1,2}\.\s)", re.M)
_MIN_CHILD = 40


def build_children(parents: list[dict]) -> list[dict]:
    children = []
    for p in parents:
        lines = p["content"].split("\n")
        prefix = lines[0]  # "메리츠화재 | 상품 | 특약 | 조"
        body = "\n".join(lines[2:]) if len(lines) > 2 else p["content"]
        raw = [s for s in _CLAUSE.split(body) if s and s.strip()]
        # 너무 짧은 조각은 앞 조각에 붙임
        parts: list[str] = []
        for s in raw:
            if parts and len(s.strip()) < _MIN_CHILD:
                parts[-1] += s
            else:
                parts.append(s)
        if not parts:
            parts = [body]
        for j, s in enumerate(parts):
            children.append({
                "key": f"{p['chunk_index']}_{j}",
                "parent_index": p["chunk_index"],
                "content": prefix + "\n" + s.strip(),
            })
    return children


def main() -> None:
    questions = load_jsonl(QUESTIONS)
    parents = load_jsonl(VERSIONS["v3"])
    parent_by_idx = {p["chunk_index"]: p for p in parents}

    children = build_children(parents)
    print(f"부모 {len(parents)} → 자식 {len(children)} (평균 {len(children)/len(parents):.1f}개/부모)")

    # 임베딩 (자식 + 질문은 기존 캐시)
    qvecs = {r["key"]: r["vec"] for r in load_jsonl(EVAL_DIR / "emb_cache_queries.jsonl")}
    cvecs = build_embed_cache(EVAL_DIR / "emb_cache_children_v3.jsonl",
                              [(c["key"], c["content"]) for c in children])

    doc_tokens = [tokenize_korean(c["content"]).split() for c in children]
    bm25 = BM25(doc_tokens)
    emb = np.array([cvecs[c["key"]] for c in children])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    n = len(children)

    agg: dict[str, dict[str, list[float]]] = {}
    for q in questions:
        bm_rank = list(np.argsort(-bm25.scores(tokenize_korean(q["question"]).split())))
        qv = np.array(qvecs[q["qid"]]); qv = qv / np.linalg.norm(qv)
        em_rank = list(np.argsort(-(emb @ qv)))
        fused = rrf(bm_rank[:80], em_rank[:80], n)

        # 자식 → 부모 승격 (등장 순서 유지, dedup)
        seen, parent_rank = set(), []
        for ci in fused:
            pi = children[ci]["parent_index"]
            if pi not in seen:
                seen.add(pi)
                parent_rank.append(pi)
            if len(parent_rank) >= 40:
                break

        # 부모 top-20 리랭크
        cand = parent_rank[:20]
        pairs = [(q["question"], parent_by_idx[pi]["content"][:1500]) for pi in cand]
        sc = _reranker().predict(pairs, batch_size=16, show_progress_bar=False)
        reranked = [cand[j] for j in np.argsort(-np.asarray(sc))] + parent_rank[20:]

        systems = {"small2big(자식검색만)": parent_rank, "small2big+rerank": reranked}
        for name, rank in systems.items():
            top = [parent_by_idx[pi] for pi in rank[:TOP_K]]
            hits = [is_hit(c, q, "lenient") for c in top]
            m = metrics(hits)
            agg.setdefault(name, {})
            for k, v in m.items():
                agg[name].setdefault(k, []).append(v)
            agg[name].setdefault("_t_" + q["qtype"], []).append(m["R@5"])

    print(f"\n=== small2big lenient ({len(questions)}문항) — 기준: v3+rerank R@1 0.707 / R@5 0.829, "
          f"claude+rerank R@1 0.659 / R@5 0.902 ===")
    print("| 시스템 | R@1 | R@5 | MRR |")
    print("|---|---|---|---|")
    for name, ms in agg.items():
        print(f"| {name} | {np.mean(ms['R@1']):.3f} | {np.mean(ms['R@5']):.3f} | {np.mean(ms['MRR']):.3f} |")
    qtypes = sorted({k for ms in agg.values() for k in ms if k.startswith("_t_")})
    for name, ms in agg.items():
        print(f"  {name}: " + " ".join(f"{t[3:]}={np.mean(ms.get(t,[0])):.2f}" for t in qtypes))


if __name__ == "__main__":
    main()
