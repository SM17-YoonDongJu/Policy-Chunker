"""v1/v2 청킹 검색 품질 평가: Recall@1/@5, MRR@10 (BM25 / 임베딩 / RRF).

사용법:
  .venv/bin/python eval/retrieval_eval.py --check-gold   # 골드 라벨 도달 가능성 검증
  .venv/bin/python eval/retrieval_eval.py --embed        # 임베딩 캐시 생성(이어받기 가능)
  .venv/bin/python eval/retrieval_eval.py --run          # 평가 실행 + 결과 저장

판정 기준:
  strict  : 검색된 청크의 (section, article)이 골드 쌍과 일치.
            표 질문(article=null)은 section 일치 + 정답 문자열 포함.
  lenient : section 일치 + (article 일치 or 정답 키워드가 content에 포함).
            v1의 조 병합 버그로 라벨이 어긋나도 내용이 맞으면 hit → 라벨 품질과
            검색 품질을 분리해서 보기 위한 보조 지표.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from insurance_chunker.tokenizer import tokenize_korean  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
VERSIONS = {
    "v1": EVAL_DIR / "chunks_30327_full.jsonl",
    "v2": EVAL_DIR / "chunks_30327_final.jsonl",
}
QUESTIONS = EVAL_DIR / "questions_30327.jsonl"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
QUERY_INSTRUCT = ("Instruct: Given a Korean insurance policy question, "
                  "retrieve relevant policy clauses that answer the question.\nQuery: ")
TOP_K = 10
BATCH = 32


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


# ---------- 판정 ----------

def is_hit(chunk: dict, q: dict, mode: str) -> bool:
    sec, art, content = chunk["section"], chunk.get("article"), chunk["content"]
    kws = q.get("keywords", [])
    for g in q["gold"]:
        if sec != g["section"]:
            continue
        if g["article"] is None:  # 표 질문: section + 정답 문자열
            if any(k in content for k in kws):
                return True
            continue
        if art == g["article"]:
            return True
        if mode == "lenient" and any(k in content for k in kws):
            return True
    return False


def metrics(ranked_hits: list[bool]) -> dict:
    r1 = 1.0 if any(ranked_hits[:1]) else 0.0
    r5 = 1.0 if any(ranked_hits[:5]) else 0.0
    mrr = 0.0
    for i, h in enumerate(ranked_hits[:TOP_K]):
        if h:
            mrr = 1.0 / (i + 1)
            break
    return {"R@1": r1, "R@5": r5, "MRR": mrr}


# ---------- BM25 ----------

class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.tf: list[dict] = []
        df: dict = defaultdict(int)
        for d in docs_tokens:
            counts: dict = defaultdict(int)
            for t in d:
                counts[t] += 1
            self.tf.append(counts)
            for t in counts:
                df[t] += 1
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for t in query_tokens:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, counts in enumerate(self.tf):
                f = counts.get(t)
                if f:
                    dl = self.doc_len[i]
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


# ---------- 임베딩 (이어받기 캐시) ----------

def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = requests.post(f"{OLLAMA_URL}/api/embed",
                         json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    resp.raise_for_status()
    embs = resp.json()["embeddings"]
    assert len(embs) == len(texts)
    return embs


def build_embed_cache(cache_path: Path, items: list[tuple[str, str]]) -> dict[str, list[float]]:
    """items: [(key, text)]. 완료분은 스킵하고 append."""
    done: dict[str, list[float]] = {}
    if cache_path.exists():
        for row in load_jsonl(cache_path):
            done[row["key"]] = row["vec"]
    todo = [(k, t) for k, t in items if k not in done]
    if todo:
        print(f"  {cache_path.name}: {len(done)} cached, {len(todo)} to embed")
        with cache_path.open("a") as f:
            for i in range(0, len(todo), BATCH):
                batch = todo[i:i + BATCH]
                vecs = embed_batch([t for _, t in batch])
                for (k, _), v in zip(batch, vecs):
                    f.write(json.dumps({"key": k, "vec": v}) + "\n")
                    done[k] = v
                f.flush()
                print(f"  {cache_path.name}: {min(i + BATCH, len(todo))}/{len(todo)}")
    return done


# ---------- 실행 ----------

def check_gold():
    questions = load_jsonl(QUESTIONS)
    for ver, path in VERSIONS.items():
        chunks = load_jsonl(path)
        print(f"=== {ver} ({path.name}) ===")
        for q in questions:
            n_strict = sum(is_hit(c, q, "strict") for c in chunks)
            n_len = sum(is_hit(c, q, "lenient") for c in chunks)
            flag = "  <-- UNREACHABLE(lenient)" if n_len == 0 else (
                   "  <-- no strict match" if n_strict == 0 else "")
            print(f"  {q['qid']} strict={n_strict:3d} lenient={n_len:3d}{flag}")


def do_embed():
    questions = load_jsonl(QUESTIONS)
    build_embed_cache(EVAL_DIR / "emb_cache_queries.jsonl",
                      [(q["qid"], QUERY_INSTRUCT + q["question"]) for q in questions])
    for ver, path in VERSIONS.items():
        chunks = load_jsonl(path)
        build_embed_cache(EVAL_DIR / f"emb_cache_{ver}.jsonl",
                          [(str(c["chunk_index"]), c["content"]) for c in chunks])


def rrf(rank_a: list[int], rank_b: list[int], n: int, k: int = 60) -> list[int]:
    """rank_a/rank_b: 문서 인덱스의 정렬 리스트(상위부터). RRF 융합 정렬 반환."""
    score = np.zeros(n)
    for r, idx in enumerate(rank_a):
        score[idx] += 1.0 / (k + r + 1)
    for r, idx in enumerate(rank_b):
        score[idx] += 1.0 / (k + r + 1)
    return list(np.argsort(-score))


def run():
    questions = load_jsonl(QUESTIONS)
    qvecs = {r["key"]: r["vec"] for r in load_jsonl(EVAL_DIR / "emb_cache_queries.jsonl")}
    results = {}          # (ver, method, mode) -> {metric: mean}
    per_q_rows = []

    for ver, path in VERSIONS.items():
        chunks = load_jsonl(path)
        n = len(chunks)
        print(f"=== {ver}: {n} chunks ===")

        print("  BM25 인덱스 구축...")
        doc_tokens = [tokenize_korean(c["content"]).split() for c in chunks]
        bm25 = BM25(doc_tokens)

        cvecs = {r["key"]: r["vec"] for r in load_jsonl(EVAL_DIR / f"emb_cache_{ver}.jsonl")}
        emb_matrix = np.array([cvecs[str(c["chunk_index"])] for c in chunks])
        emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

        agg = defaultdict(lambda: defaultdict(list))
        for q in questions:
            bm_scores = bm25.scores(tokenize_korean(q["question"]).split())
            bm_rank = list(np.argsort(-bm_scores))

            qv = np.array(qvecs[q["qid"]])
            qv = qv / np.linalg.norm(qv)
            emb_rank = list(np.argsort(-(emb_matrix @ qv)))

            rankings = {
                "bm25": bm_rank,
                "embed": emb_rank,
                "rrf": rrf(bm_rank[:50], emb_rank[:50], n),
            }
            for method, rank in rankings.items():
                top = [chunks[i] for i in rank[:TOP_K]]
                for mode in ("strict", "lenient"):
                    hits = [is_hit(c, q, mode) for c in top]
                    m = metrics(hits)
                    for k, v in m.items():
                        agg[(method, mode)][k].append(v)
                    per_q_rows.append({
                        "ver": ver, "method": method, "mode": mode, "qid": q["qid"],
                        "qtype": q["qtype"], **m,
                        "top5": [{"idx": c["chunk_index"], "section": c["section"],
                                  "article": c.get("article")} for c in top[:5]],
                    })
        for (method, mode), ms in agg.items():
            results[(ver, method, mode)] = {k: float(np.mean(v)) for k, v in ms.items()}

    # 저장
    with (EVAL_DIR / "retrieval_eval_perq.jsonl").open("w") as f:
        for row in per_q_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report(results, per_q_rows, len(questions))


def report(results: dict, per_q_rows: list[dict], n_q: int):
    lines = ["# 검색 품질 평가: v1 vs v2 청킹",
             "",
             f"- 문서: 메리츠 단체안심생활보험(30327) / 질문 {n_q}개",
             f"- 임베딩: {EMBED_MODEL} (코사인), BM25: Kiwi 형태소 + Okapi, RRF k=60",
             "- strict: (section, article) 라벨 일치 / lenient: section 일치 + 내용(키워드) 일치",
             ""]
    for mode in ("strict", "lenient"):
        lines.append(f"## {mode}")
        lines.append("")
        lines.append("| 방법 | 버전 | R@1 | R@5 | MRR@10 |")
        lines.append("|---|---|---|---|---|")
        for method in ("bm25", "embed", "rrf"):
            for ver in VERSIONS:
                m = results[(ver, method, mode)]
                lines.append(f"| {method} | {ver} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['MRR']:.3f} |")
        lines.append("")

    # 유형별 (lenient, R@5)
    lines.append("## 질문 유형별 R@5 (lenient)")
    lines.append("")
    qtypes = sorted({r["qtype"] for r in per_q_rows})
    lines.append("| 방법 | 버전 | " + " | ".join(qtypes) + " |")
    lines.append("|---|---|" + "---|" * len(qtypes))
    for method in ("bm25", "embed", "rrf"):
        for ver in VERSIONS:
            cells = []
            for qt in qtypes:
                vals = [r["R@5"] for r in per_q_rows
                        if r["ver"] == ver and r["method"] == method
                        and r["mode"] == "lenient" and r["qtype"] == qt]
                cells.append(f"{np.mean(vals):.2f}")
            lines.append(f"| {method} | {ver} | " + " | ".join(cells) + " |")
    lines.append("")

    out = EVAL_DIR / "retrieval_eval_results.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-gold", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.check_gold:
        check_gold()
    if a.embed:
        do_embed()
    if a.run:
        run()
    if not (a.check_gold or a.embed or a.run):
        ap.print_help()
