"""쿼리 리라이팅 평가 — Gemma4로 구어 질문을 약관 용어로 변환 후 검색 성능 비교.

비교 (v3 인덱스, lenient):
  base      : 원본 질문 (rrf + rerank)  — 기존 최고 구성
  rewrite   : 리라이팅 질문으로 검색, rerank는 원본 질문으로
  both      : 원본·리라이팅 4개 랭킹 RRF 융합 + 원본으로 rerank

실행: RERANK=1 .venv/bin/python eval/query_rewrite_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RERANK", "1")

from insurance_chunker.tokenizer import tokenize_korean  # noqa: E402
from eval.retrieval_eval import (  # noqa: E402
    BM25, EVAL_DIR, QUERY_INSTRUCT, QUESTIONS, TOP_K, VERSIONS,
    _reranker, build_embed_cache, is_hit, load_jsonl, metrics, rrf,
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
REWRITE_MODEL = "gemma4:26b-a4b-it-qat"
VER = "v3"

_RW_PROMPT = """당신은 보험 약관 검색 질의 변환기입니다. 사용자의 일상 질문을 약관 조문에 쓰이는 공식 용어로 바꿔 검색 질의 한 문장을 만드세요.

규칙:
- 구어를 약관 용어로: "넘어져 다쳤다"→"상해", "돈 나와?"→"보험금 지급사유", "안 나와?"→"보험금을 지급하지 않는 사유"
- 핵심 명사를 유지하고, 예상되는 조문 제목 표현을 포함
- 설명 없이 변환된 질의 한 줄만 출력"""


def rewrite(q: str) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": REWRITE_MODEL,
        "messages": [{"role": "system", "content": _RW_PROMPT},
                     {"role": "user", "content": q}],
        "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 80}}, timeout=120)
    r.raise_for_status()
    out = r.json()["message"]["content"].strip().split("\n")[0].strip()
    return out or q


def main() -> None:
    questions = load_jsonl(QUESTIONS)

    # 1) 리라이팅 (캐시)
    rw_path = EVAL_DIR / "queries_rewritten.jsonl"
    rw: dict[str, str] = {}
    if rw_path.exists():
        rw = {r["qid"]: r["rewritten"] for r in load_jsonl(rw_path)}
    with rw_path.open("a") as f:
        for q in questions:
            if q["qid"] in rw:
                continue
            rw[q["qid"]] = rewrite(q["question"])
            f.write(json.dumps({"qid": q["qid"], "question": q["question"],
                                "rewritten": rw[q["qid"]]}, ensure_ascii=False) + "\n")
            f.flush()
    print("리라이팅 샘플:")
    for q in questions[:3]:
        print(f"  {q['question']} → {rw[q['qid']]}")

    # 2) 임베딩 캐시 (원본은 기존 캐시 재사용)
    qvecs = {r["key"]: r["vec"] for r in load_jsonl(EVAL_DIR / "emb_cache_queries.jsonl")}
    rw_vecs = build_embed_cache(EVAL_DIR / "emb_cache_queries_rw.jsonl",
                                [(q["qid"], QUERY_INSTRUCT + rw[q["qid"]]) for q in questions])

    # 3) 인덱스
    chunks = load_jsonl(VERSIONS[VER])
    n = len(chunks)
    doc_tokens = [tokenize_korean(c["content"]).split() for c in chunks]
    bm25 = BM25(doc_tokens)
    cvecs = {r["key"]: r["vec"] for r in load_jsonl(EVAL_DIR / f"emb_cache_{VER}.jsonl")}
    emb = np.array([cvecs[str(c["chunk_index"])] for c in chunks])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def ranks(question_text: str, qvec) -> tuple[list[int], list[int]]:
        bm = list(np.argsort(-bm25.scores(tokenize_korean(question_text).split())))
        qv = np.array(qvec); qv = qv / np.linalg.norm(qv)
        em = list(np.argsort(-(emb @ qv)))
        return bm, em

    def rerank(cand: list[int], original_q: str) -> list[int]:
        pairs = [(original_q, chunks[i]["content"][:1500]) for i in cand]
        sc = _reranker().predict(pairs, batch_size=16, show_progress_bar=False)
        return [cand[j] for j in np.argsort(-np.asarray(sc))]

    agg: dict[str, dict[str, list[float]]] = {}
    for q in questions:
        bm_o, em_o = ranks(q["question"], qvecs[q["qid"]])
        bm_r, em_r = ranks(rw[q["qid"]], rw_vecs[q["qid"]])

        fuse_o = rrf(bm_o[:50], em_o[:50], n)
        fuse_r = rrf(bm_r[:50], em_r[:50], n)
        # both: 4개 랭킹 융합
        score = np.zeros(n)
        for rank in (bm_o[:50], em_o[:50], bm_r[:50], em_r[:50]):
            for pos, idx in enumerate(rank):
                score[idx] += 1.0 / (60 + pos + 1)
        fuse_b = list(np.argsort(-score))

        systems = {
            "base(rrf+rerank)": rerank(fuse_o[:20], q["question"]),
            "rewrite(+rerank)": rerank(fuse_r[:20], q["question"]),
            "both(+rerank)": rerank(fuse_b[:20], q["question"]),
        }
        for name, rank in systems.items():
            top = [chunks[i] for i in rank[:TOP_K]]
            hits = [is_hit(c, q, "lenient") for c in top]
            m = metrics(hits)
            agg.setdefault(name, {})
            for k, v in m.items():
                agg[name].setdefault(k, []).append(v)
            agg[name].setdefault("_qtype_" + q["qtype"], []).append(m["R@5"])

    print(f"\n=== {VER} lenient (41문항) ===")
    print("| 시스템 | R@1 | R@5 | MRR |")
    print("|---|---|---|---|")
    for name, ms in agg.items():
        print(f"| {name} | {np.mean(ms['R@1']):.3f} | {np.mean(ms['R@5']):.3f} | {np.mean(ms['MRR']):.3f} |")
    print("\n유형별 R@5:")
    qtypes = sorted({k for ms in agg.values() for k in ms if k.startswith("_qtype_")})
    for name, ms in agg.items():
        row = " ".join(f"{t[7:]}={np.mean(ms.get(t, [0])):.2f}" for t in qtypes)
        print(f"  {name}: {row}")


if __name__ == "__main__":
    main()
