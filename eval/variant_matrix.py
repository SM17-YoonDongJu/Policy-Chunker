"""랭그래프 질의 4형태 × 검색 방법 매트릭스.

`eval/qgen.py`가 만든 질문셋의 `variants`(utterance / rewritten / followup / report)를
같은 청크셋·같은 검색기로 돌려 **질의 모양이 검색 품질을 얼마나 좌우하는지**를 잰다.
retrieval_eval.py를 변형마다 프로세스로 재실행하면 BM25 인덱스를 4번 짓게 되므로,
인덱스·청크 임베딩은 1회만 만들고 질의만 갈아끼운다.

    .venv/bin/python eval/variant_matrix.py --questions eval/questions_gen_v6.jsonl
    RERANK=1 .venv/bin/python eval/variant_matrix.py --ver v6 --ver v3

읽는 법: utterance(챗봇 원문 발화) 대비 각 형태의 증감이 곧 그래프 레이어의 선택
(리라이팅을 켤지, 멀티턴 히스토리를 어떻게 넘길지, report_worker 질의를 어떻게 만들지)이
검색에 주는 효과다.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval.retrieval_eval as R  # noqa: E402
from insurance_chunker.tokenizer import tokenize_korean  # noqa: E402
from insurance_chunker.embedder import QUERY_INSTRUCT  # noqa: E402

VARIANTS = ("utterance", "rewritten", "followup", "report")

# rerank에 넣을 질의는 "사용자가 실제로 한 말"이어야 한다. rewritten은 검색용 기계 질의라
# cross-encoder에 그대로 넣으면 안 된다 — query_rewrite_eval.py에서 검증된 구성도
# "리라이팅으로 검색, rerank는 원본으로"다. followup/report는 그 자체가 입력이므로 그대로.
RERANK_WITH = {"rewritten": "utterance"}


def rerank_query(q: dict, variant: str) -> str:
    src = RERANK_WITH.get(variant)
    return q["variants"][src] if src and src in q.get("variants", {}) else R.query_of(q)


def eval_variant(questions: list[dict], chunks: list[dict], bm25, emb_matrix, variant: str):
    R.VARIANT = variant
    qcache = R.build_embed_cache(
        R.query_cache_path(),
        [(q["qid"], QUERY_INSTRUCT + R.query_of(q)) for q in questions])

    agg = defaultdict(lambda: defaultdict(list))
    per_type = defaultdict(lambda: defaultdict(list))
    n = len(chunks)
    for q in questions:
        query = R.query_of(q)
        bm_rank = list(np.argsort(-bm25.scores(tokenize_korean(query).split())))
        qv = np.array(qcache[q["qid"]])
        qv = qv / np.linalg.norm(qv)
        em_rank = list(np.argsort(-(emb_matrix @ qv)))
        rankings = {"bm25": bm_rank, "embed": em_rank,
                    "rrf": R.rrf(bm_rank[:50], em_rank[:50], n)}
        if R.RERANK:
            cand = rankings["rrf"][:R.RERANK_TOP]
            rq = rerank_query(q, variant)
            scores = R._reranker().predict(
                [(rq, chunks[i]["content"][:1500]) for i in cand],
                batch_size=16, show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))
            rankings["rerank"] = [cand[j] for j in order] + rankings["rrf"][R.RERANK_TOP:]
        for method, rank in rankings.items():
            hits = [R.is_hit(chunks[i], q, "lenient") for i in rank[:R.TOP_K]]
            m = R.metrics(hits)
            for k, v in m.items():
                agg[method][k].append(v)
            per_type[(method, q["qtype"])]["R@5"].append(m["R@5"])
    return ({me: {k: float(np.mean(v)) for k, v in ms.items()} for me, ms in agg.items()},
            {k: float(np.mean(v["R@5"])) for k, v in per_type.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(R.EVAL_DIR / "questions_gen_v6.jsonl"))
    ap.add_argument("--ver", action="append", default=None, help="청크셋 버전(복수 지정 가능)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()

    R.QUESTIONS = Path(a.questions)
    questions = R.load_jsonl(R.QUESTIONS)
    vers = a.ver or ["v6"]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    methods = ("bm25", "embed", "rrf") + (("rerank",) if R.RERANK else ())

    lines = [f"# 랭그래프 질의 형태별 검색 품질 — `{R.QUESTIONS.name}` ({len(questions)}문항)", "",
             f"- 판정: lenient / 청크셋: {', '.join(vers)} / rerank: {'on' if R.RERANK else 'off'}",
             "- utterance = 챗봇 원문 발화(기존 평가가 재던 유일한 모양)",
             "- rewritten의 rerank는 원본 발화로 수행 (검색만 리라이팅 질의 — 검증된 구성)", ""]

    for ver in vers:
        chunks = R.load_jsonl(R.VERSIONS[ver])
        print(f"=== {ver}: {len(chunks)}청크 — BM25 인덱스 구축")
        bm25 = R.BM25([tokenize_korean(c["content"]).split() for c in chunks])
        cvecs = {r["key"]: r["vec"] for r in R.load_jsonl(R.EVAL_DIR / f"emb_cache_{ver}.jsonl")}
        emb = np.array([cvecs[str(c["chunk_index"])] for c in chunks])
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        results, types = {}, {}
        for v in variants:
            print(f"  variant={v}")
            results[v], types[v] = eval_variant(questions, chunks, bm25, emb, v)

        base = results.get("utterance", {})
        lines += [f"## {ver}", "", "| 질의 모양 | 방법 | R@1 | R@5 | MRR@10 | ΔR@5 vs utterance |",
                  "|---|---|---|---|---|---|"]
        for v in variants:
            for me in methods:
                m = results[v][me]
                d = m["R@5"] - base.get(me, {}).get("R@5", m["R@5"])
                lines.append(f"| {v} | {me} | {m['R@1']:.3f} | {m['R@5']:.3f} | "
                             f"{m['MRR']:.3f} | {d:+.3f} |")
        qtypes = sorted({q["qtype"] for q in questions})
        best = methods[-1]
        lines += ["", f"### 유형별 R@5 ({best})", "",
                  "| 질의 모양 | " + " | ".join(qtypes) + " |",
                  "|---|" + "---|" * len(qtypes)]
        for v in variants:
            cells = [f"{types[v].get((best, t), float('nan')):.2f}" for t in qtypes]
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")

    out = R.EVAL_DIR / f"variant_matrix_{R.QUESTIONS.stem}.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
