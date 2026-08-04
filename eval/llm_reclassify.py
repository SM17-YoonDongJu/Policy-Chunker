"""687청크 전체 LLM 재분류 — 중단 지점부터 이어받기 지원."""
import json, os, sys, time
sys.path.insert(0, ".")
from insurance_chunker.llm_classifier import classify_llm

chunks = [json.loads(l) for l in open("eval/chunks_30327.jsonl")]
out_path = "eval/chunks_30327_llm_types.jsonl"
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["chunk_index"] for l in open(out_path)}
print(f"이어받기: {len(done)}건 완료됨, {len(chunks)-len(done)}건 남음")
out = open(out_path, "a")
diff = 0
t0 = time.time()
todo = [c for c in chunks if c["chunk_index"] not in done]
for i, c in enumerate(todo):
    llm_ct = classify_llm(c["content"], title=c.get("article_title")) or c["chunk_type"]
    if llm_ct != c["chunk_type"]:
        diff += 1
    out.write(json.dumps({"chunk_index": c["chunk_index"], "article": c.get("article"),
                          "title": c.get("article_title"), "keyword": c["chunk_type"],
                          "llm": llm_ct}, ensure_ascii=False) + "\n")
    out.flush()
    if (i+1) % 100 == 0:
        print(f"{i+1}/{len(todo)} | 새 불일치 {diff} | {time.time()-t0:.0f}s", flush=True)
out.close()
print(f"완료: 신규 {len(todo)}건 처리 (누적 {len(done)+len(todo)}), {time.time()-t0:.0f}s")
