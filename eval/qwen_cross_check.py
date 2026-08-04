"""A.X-키워드 불일치 416건을 Qwen3.6으로 교차 검증."""
import json, os, sys, time
sys.path.insert(0, ".")
os.environ["CLASSIFY_MODEL"] = "qwen3.6:35b-a3b"
import requests
from insurance_chunker.llm_classifier import _SYSTEM, _TYPES, OLLAMA_URL

chunks = {c["chunk_index"]: c for c in map(json.loads, open("eval/chunks_30327.jsonl"))}
rows = [json.loads(l) for l in open("eval/chunks_30327_llm_types.jsonl")]
diffs = [r for r in rows if r["keyword"] != r["llm"]]
out_path = "eval/qwen_cross.jsonl"
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["chunk_index"] for l in open(out_path)}
todo = [r for r in diffs if r["chunk_index"] not in done]
print(f"대상 {len(diffs)}건, 남음 {len(todo)}건")
out = open(out_path, "a")
t0 = time.time()
for i, r in enumerate(todo):
    c = chunks[r["chunk_index"]]
    body = c["content"][:2000]
    title = r.get("title")
    user = f"제목: {title}\n\n{body}" if title else body
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": "qwen3.6:35b-a3b",
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": user}],
            "format": {"type": "object", "properties": {"chunk_type": {"type": "string", "enum": _TYPES}},
                       "required": ["chunk_type"]},
            "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 30}}, timeout=120)
        qwen = json.loads(resp.json()["message"]["content"]).get("chunk_type")
    except Exception:
        qwen = None
    out.write(json.dumps({**r, "qwen": qwen}, ensure_ascii=False) + "\n")
    out.flush()
    if (i+1) % 100 == 0:
        print(f"{i+1}/{len(todo)} | {time.time()-t0:.0f}s", flush=True)
agree = ax_n = 0
for l in open(out_path):
    j = json.loads(l)
    if j.get("qwen"):
        ax_n += 1
        agree += j["llm"] == j["qwen"]
print(f"완료 — A.X vs Qwen3.6 합의: {agree}/{ax_n} ({agree/ax_n:.0%})")
