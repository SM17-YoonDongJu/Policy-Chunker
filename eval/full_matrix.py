"""설치된 모델 전부 × 약관 3종 풀 매트릭스.

Phase 1 — 표 VLM: Surya vs PaddleOCR-VL, 약관별 표 페이지 5개씩 추출 비교
Phase 2 — 분류 LLM: Qwen3.6, Gemma4로 3개 약관 전 청크 분류 (A.X는 이미 완료)
          → 모델별 분포 + A.X와의 합의율

이어받기 지원. 실행: nohup .venv/bin/python eval/full_matrix.py > eval/full_matrix.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")

import fitz
import requests

from insurance_chunker.extractor import (
    extract_pymupdf, extract_surya_tables, extract_vision_local,
)
from insurance_chunker.llm_classifier import _SYSTEM, _TYPES

OLLAMA_URL = "http://localhost:11434"

POLICIES = [
    ("meritz", "in/상해보험_단체안심생활보험_30327.pdf", "eval/chunks_30327.jsonl", "eval/chunks_30327_llm_types.jsonl"),
    ("samsung", "in/실손의료비_삼성화재다이렉트.pdf", "eval/chunks_samsung_silson.jsonl", "eval/chunks_samsung_silson_llm_types.jsonl"),
    ("db", "in/단체상해_빅히트_동부.pdf", "eval/chunks_db_bighit.jsonl", "eval/chunks_db_bighit_llm_types.jsonl"),
]
LLMS = [("qwen", "qwen3.6:35b-a3b"), ("gemma", "gemma4:26b-a4b-it-qat")]
N_VLM_PAGES = 5


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Phase 1: VLM 매트릭스 ────────────────────────────────────────────────────

def table_pages(pdf: str, n: int) -> list[int]:
    """PyMuPDF가 표를 감지한 페이지 중 문서 전체에 고르게 분산된 n개."""
    doc = fitz.open(pdf)
    pages = [p + 1 for p in range(len(doc)) if extract_pymupdf(doc[p])]
    doc.close()
    if len(pages) <= n:
        return pages
    step = len(pages) / n
    return [pages[int(i * step)] for i in range(n)]


def phase1() -> None:
    out_path = "eval/vlm_matrix.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {(j["policy"], j["page"], j["backend"]) for j in map(json.loads, open(out_path))}
    out = open(out_path, "a")

    for slug, pdf, _, _ in POLICIES:
        pages = table_pages(pdf, N_VLM_PAGES)
        log(f"[VLM] {slug}: 대상 페이지 {pages}")
        doc = fitz.open(pdf)

        todo_surya = [p for p in pages if (slug, p, "surya") not in done]
        if todo_surya:
            t0 = time.time()
            res = extract_surya_tables(pdf, todo_surya)
            dt = time.time() - t0
            for p in todo_surya:
                out.write(json.dumps({"policy": slug, "page": p, "backend": "surya",
                                      "sec_per_page": round(dt / len(todo_surya), 1),
                                      "markdown": res.get(p)}, ensure_ascii=False) + "\n")
            out.flush()
            log(f"[VLM] {slug} surya: {len([p for p in todo_surya if res.get(p)])}/{len(todo_surya)} 추출 ({dt:.0f}s)")

        for p in pages:
            if (slug, p, "paddle") in done:
                continue
            t0 = time.time()
            md = extract_vision_local(doc[p - 1], p)
            out.write(json.dumps({"policy": slug, "page": p, "backend": "paddle",
                                  "sec_per_page": round(time.time() - t0, 1),
                                  "markdown": md}, ensure_ascii=False) + "\n")
            out.flush()
        log(f"[VLM] {slug} paddle 완료")

        for p in pages:
            if (slug, p, "pymupdf") in done:
                continue
            md = extract_pymupdf(doc[p - 1])
            out.write(json.dumps({"policy": slug, "page": p, "backend": "pymupdf",
                                  "sec_per_page": 0.0, "markdown": md}, ensure_ascii=False) + "\n")
        out.flush()
        doc.close()
    out.close()
    log("[VLM] Phase 1 완료 → eval/vlm_matrix.jsonl")


# ── Phase 2: LLM 분류 매트릭스 ───────────────────────────────────────────────

def classify(model: str, text: str, title: str | None) -> str | None:
    user = f"제목: {title}\n\n{text[:2000]}" if title else text[:2000]
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model,
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": user}],
            "format": {"type": "object",
                       "properties": {"chunk_type": {"type": "string", "enum": _TYPES}},
                       "required": ["chunk_type"]},
            "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 30}}, timeout=180)
        r.raise_for_status()
        ct = json.loads(r.json()["message"]["content"]).get("chunk_type")
        return ct if ct in _TYPES else None
    except Exception:
        return None


def phase2() -> None:
    for mslug, model in LLMS:
        for pslug, _, chunk_file, _ in POLICIES:
            chunks = [json.loads(l) for l in open(chunk_file)]
            out_path = f"eval/llm_matrix_{mslug}_{pslug}.jsonl"
            done = set()
            if os.path.exists(out_path):
                done = {j["chunk_index"] for j in map(json.loads, open(out_path))}
            todo = [c for c in chunks if c["chunk_index"] not in done]
            log(f"[LLM] {model} × {pslug}: {len(done)}건 완료, {len(todo)}건 남음")
            out = open(out_path, "a")
            t0 = time.time()
            for i, c in enumerate(todo):
                ct = classify(model, c["content"], c.get("article_title"))
                out.write(json.dumps({"chunk_index": c["chunk_index"],
                                      "type": ct or c["chunk_type"]}, ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 100 == 0:
                    log(f"  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)")
            out.close()
    log("[LLM] Phase 2 완료")


# ── 요약 ─────────────────────────────────────────────────────────────────────

def summary() -> None:
    from collections import Counter
    log("\n===== 최종 매트릭스 요약 =====")
    for pslug, _, chunk_file, ax_file in POLICIES:
        keyword = {j["chunk_index"]: j["chunk_type"] for j in map(json.loads, open(chunk_file))}
        ax = {j["chunk_index"]: j["llm"] for j in map(json.loads, open(ax_file))}
        log(f"\n--- {pslug} ({len(keyword)}청크)")
        for mslug, model in [("ax", None)] + LLMS:
            preds = ax if mslug == "ax" else {
                j["chunk_index"]: j["type"]
                for j in map(json.loads, open(f"eval/llm_matrix_{mslug}_{pslug}.jsonl"))
            }
            dist = dict(Counter(preds.values()).most_common(4))
            kw_diff = sum(1 for k, v in preds.items() if keyword.get(k) != v)
            line = f"  {mslug:<6} 키워드불일치 {kw_diff}/{len(preds)} | 상위분포 {dist}"
            if mslug != "ax":
                agree = sum(1 for k, v in preds.items() if ax.get(k) == v)
                line += f" | A.X합의 {agree/len(preds):.0%}"
            log(line)


if __name__ == "__main__":
    phase1()
    phase2()
    summary()
