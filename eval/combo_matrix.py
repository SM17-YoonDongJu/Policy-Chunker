"""3 LLM × 2 VLM = 6조합 엔드투엔드 실험 (메리츠 30327 기준).

VLM 백엔드별로 전체 청킹 1벌씩 생성(표 청크 내용이 달라짐) →
각 청킹의 표 청크를 3개 LLM이 분류 → 조합별 분포/합의 비교.

이어받기 지원. 실행: nohup .venv/bin/python eval/combo_matrix.py > eval/combo_matrix.log 2>&1 &
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, ".")

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PDF = "in/상해보험_단체안심생활보험_30327.pdf"
VLMS = ["surya", "local"]  # local = PaddleOCR-VL
LLMS = [
    ("ax", "hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M"),
    ("qwen", "qwen3.6:35b-a3b"),
    ("gemma", "gemma4:26b-a4b-it-qat"),
]
OLLAMA_URL = "http://localhost:11434"


def log(m):
    print(m, flush=True)


def build_chunks(vlm: str) -> str:
    """VLM 백엔드로 전체 청킹 → jsonl 경로 반환 (이미 있으면 재사용)."""
    out = f"eval/chunks_30327_vlm_{vlm}.jsonl"
    if os.path.exists(out):
        log(f"[청킹] {vlm}: 기존 파일 재사용")
        return out
    os.environ["VLM_BACKEND"] = vlm
    # extractor가 모듈 상수로 읽으므로 재로드 필요
    for mod in list(sys.modules):
        if mod.startswith("insurance_chunker"):
            del sys.modules[mod]
    from insurance_chunker.chunker import chunk_document
    from insurance_chunker.models import DocMeta, compute_doc_hash
    from insurance_chunker.pdf_parser import parse_pdf

    meta = DocMeta(source_pdf=PDF.split("/")[-1], doc_hash=compute_doc_hash(PDF),
                   doc_type="policy_terms", insurer="메리츠화재",
                   product_name="단체안심생활보험", effective_date="2026-05-29")
    t0 = time.time()
    pages = parse_pdf(PDF, use_ocr=False, use_vision=True)
    chunks, tables = chunk_document(pages, meta, pdf_path=PDF)
    log(f"[청킹] {vlm}: {len(chunks)}청크, 표 {len(tables)}건 ({time.time()-t0:.0f}s)")
    src = {}
    for p in pages:
        if p.tables:
            src[p.tables[0]["source"]] = src.get(p.tables[0]["source"], 0) + 1
    log(f"[청킹] {vlm}: 표 소스 채택 {src}")
    with open(out, "w") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_index": c.chunk_index, "page": c.page_number,
                "section": c.section, "article": c.article_number,
                "article_title": c.article_title, "chunk_type": c.chunk_type,
                "token_count": c.token_count, "table_id": c.table_id,
                "is_table": bool(c.table_id) or c.chunk_type == "schedule",
                "content": c.content,
            }, ensure_ascii=False) + "\n")
    return out


def classify(model: str, text: str, title):
    from insurance_chunker.llm_classifier import _SYSTEM, _TYPES
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


def main() -> None:
    chunk_files = {vlm: build_chunks(vlm) for vlm in VLMS}

    # 표가 포함된 청크만 분류 대상 (VLM 차이가 반영되는 부분)
    for vlm, cf in chunk_files.items():
        chunks = [json.loads(l) for l in open(cf)]
        targets = [c for c in chunks if c["is_table"]]
        log(f"\n[분류] VLM={vlm}: 표 청크 {len(targets)}건 × LLM 3종")
        for mslug, model in LLMS:
            out_path = f"eval/combo_{vlm}_{mslug}.jsonl"
            done = set()
            if os.path.exists(out_path):
                done = {j["chunk_index"] for j in map(json.loads, open(out_path))}
            todo = [c for c in targets if c["chunk_index"] not in done]
            out = open(out_path, "a")
            t0 = time.time()
            for i, c in enumerate(todo):
                ct = classify(model, c["content"], c.get("article_title"))
                out.write(json.dumps({"chunk_index": c["chunk_index"],
                                      "type": ct or c["chunk_type"]},
                                     ensure_ascii=False) + "\n")
                out.flush()
            out.close()
            log(f"  {mslug}: {len(todo)}건 처리 ({time.time()-t0:.0f}s)")

    # 요약: 6조합 분포 + VLM 간 분류 일치율
    from collections import Counter
    log("\n===== 6조합 요약 =====")
    preds = {}
    for vlm in VLMS:
        for mslug, _ in LLMS:
            p = {j["chunk_index"]: j["type"]
                 for j in map(json.loads, open(f"eval/combo_{vlm}_{mslug}.jsonl"))}
            preds[(vlm, mslug)] = p
            log(f"  {vlm}×{mslug}: {dict(Counter(p.values()).most_common(4))}")
    for mslug, _ in LLMS:
        a, b = preds[("surya", mslug)], preds[("local", mslug)]
        common = set(a) & set(b)
        if common:
            agree = sum(1 for k in common if a[k] == b[k])
            log(f"  {mslug}: surya vs paddle 분류 일치 {agree}/{len(common)} ({agree/len(common):.0%})")


if __name__ == "__main__":
    main()
