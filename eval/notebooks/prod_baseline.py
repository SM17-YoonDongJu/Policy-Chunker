"""운영 이력(items.jsonl·runs.jsonl·daemon.json)에서 VLM 전환 전후 기준선을 뽑는다. 인자: <state 디렉터리>

사용: kubectl exec <chunker 파드> -- cat /data/state/items.jsonl > state/items.jsonl (runs.jsonl·daemon.json도)
      python eval/notebooks/prod_baseline.py state/ > baseline.md
전환 전 수치는 이슈 #72 댓글에 있다. 같은 스크립트로 뽑아야 짝지어 비교된다."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

d = Path(sys.argv[1])
items = [json.loads(l) for l in (d / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
runs = [json.loads(l) for l in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (d / "runs.jsonl").exists() else []

def pct(vals, q):
    s = sorted(v for v in vals if v is not None)
    return round(s[min(len(s) - 1, int(len(s) * q))], 1) if s else None

def med(vals): return pct(vals, 0.5)

out = []
P = out.append
P(f"# 운영 기준선 — Paddle 전환 전 (items {len(items)}건, {items[0]['at'][:10]} ~ {items[-1]['at'][:10]})")
P("")
status = Counter(i["status"] for i in items)
P("## 문서 상태")
P("| 상태 | 건수 |"); P("|---|---|")
for k, v in status.most_common(): P(f"| {k} | {v} |")
P("")

ok = [i for i in items if i["status"] == "OK" and i.get("stages")]
P(f"## 처리한 문서 (OK, 단계 기록 있음: {len(ok)}건)")
el = [i.get("elapsed_s") for i in ok]
pages = [((i["stages"].get("text") or {}).get("pages")) for i in ok]
chunks = [i.get("chunks") for i in ok]
P("| 지표 | p50 | p95 | max |"); P("|---|---|---|---|")
P(f"| 문서당 처리 시간(초) | {med(el)} | {pct(el, .95)} | {pct(el, 1.0)} |")
P(f"| 쪽 수 | {med(pages)} | {pct(pages, .95)} | {pct(pages, 1.0)} |")
P(f"| 청크 수 | {med(chunks)} | {pct(chunks, .95)} | {pct(chunks, 1.0)} |")
P("")

P("## 단계별 시간 (OK 문서 합계 기준 비중, 문서당 p50)")
tot = defaultdict(float); per = defaultdict(list)
for i in ok:
    for name, st in i["stages"].items():
        s = float(st.get("seconds") or 0); tot[name] += s; per[name].append(s)
all_s = sum(tot.values()) or 1
P("| 단계 | 비중 | 문서당 p50(초) | 문서당 p95(초) |"); P("|---|---|---|---|")
for name, s in sorted(tot.items(), key=lambda kv: -kv[1]):
    P(f"| {name} | {s / all_s:.1%} | {med(per[name])} | {pct(per[name], .95)} |")
P("")

P("## VLM (qwen3-vl 8B, 페이지 전체 입력, target=gap)")
v_docs = [i for i in ok if i["stages"].get("vlm") and not i["stages"]["vlm"].get("skipped")]
lat = [x for i in v_docs for x in (i["stages"]["vlm"].get("latencies") or [])]
outcomes = Counter()
for i in v_docs: outcomes.update(i["stages"]["vlm"].get("outcomes") or {})
calls = [i["stages"]["vlm"].get("calls") for i in v_docs]
targets = [i["stages"]["vlm"].get("targets") for i in v_docs]
tables = [i["stages"]["vlm"].get("tables") for i in v_docs]
trunc = sum(int(i["stages"]["vlm"].get("truncated") or 0) for i in v_docs)
budget_hit = sum(1 for i in v_docs if (i["stages"]["vlm"].get("targets") or 0) > (i["stages"]["vlm"].get("calls") or 0))
circuit = sum(1 for i in v_docs if i["stages"]["vlm"].get("circuit_open"))
ptok = [i["stages"]["vlm"].get("prompt_tokens_p50") for i in v_docs]
ctok = [i["stages"]["vlm"].get("completion_tokens_p50") for i in v_docs]
vsec = [i["stages"]["vlm"].get("seconds") for i in v_docs]
vshare = [ (i["stages"]["vlm"].get("seconds") or 0) / i["elapsed_s"] for i in v_docs if i.get("elapsed_s")]
P("| 지표 | 값 |"); P("|---|---|")
P(f"| VLM을 부른 문서 | {len(v_docs)}건 / OK {len(ok)}건 |")
P(f"| 호출 총수 | {sum(calls or [0])} (결과 {dict(outcomes)}) |")
P(f"| 호출당 지연(초) p50 / p95 / max | {med(lat)} / {pct(lat, .95)} / {pct(lat, 1.0)} |")
P(f"| 잘림(max_tokens) | {trunc}건 ({trunc / max(1, sum(calls or [0])):.1%}) |")
P(f"| 입력 토큰 p50 / 생성 토큰 p50 (문서 중앙값) | {med(ptok)} / {med(ctok)} |")
P(f"| 문서당 VLM 대상 쪽 p50 / 호출 p50 / 표 p50 | {med(targets)} / {med(calls)} / {med(tables)} |")
P(f"| 문서당 VLM 시간(초) p50 / p95 | {med(vsec)} / {pct(vsec, .95)} |")
P(f"| VLM이 문서 시간에서 차지하는 비중 p50 | {med([x * 100 for x in vshare])}% |")
P(f"| 상한(VISION_MAX_PAGES)에 걸린 문서 / 차단기 열린 문서 | {budget_hit} / {circuit} |")
P("")

P("## gap 대상 선정 (tables_cpu 단계)")
gp = [(i["stages"].get("tables_cpu") or {}).get("gap_pages") for i in ok]
pm = [((i["stages"].get("tables_cpu") or {}).get("table_pages") or {}).get("pymupdf") for i in ok]
gshare = [g / p for g, p in zip(gp, pages) if g is not None and p]
P("| 지표 | p50 | p95 |"); P("|---|---|---|")
P(f"| 문서당 gap 쪽 수 | {med(gp)} | {pct(gp, .95)} |")
P(f"| 문서당 PyMuPDF 표 쪽 수 | {med(pm)} | {pct(pm, .95)} |")
P(f"| gap 쪽 비율(쪽 수 대비) | {med([x * 100 for x in gshare])}% | {pct([x * 100 for x in gshare], .95)}% |")
P("")

P("## 보험사별 (OK 문서)")
by = defaultdict(list)
for i in ok: by[i.get("insurer") or "미상"].append(i)
P("| 보험사 | 문서 | 쪽 p50 | 처리 p50(초) | VLM 호출 p50 | VLM 지연 p50(초) | VLM 비중 p50 |"); P("|---|---|---|---|---|---|---|")
for ins, docs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    vd = [x for x in docs if x["stages"].get("vlm") and not x["stages"]["vlm"].get("skipped")]
    l = [y for x in vd for y in (x["stages"]["vlm"].get("latencies") or [])]
    sh = [(x["stages"]["vlm"].get("seconds") or 0) / x["elapsed_s"] * 100 for x in vd if x.get("elapsed_s")]
    P(f"| {ins} | {len(docs)} | {med([(x['stages'].get('text') or {}).get('pages') for x in docs])} | {med([x.get('elapsed_s') for x in docs])} | "
      f"{med([x['stages']['vlm'].get('calls') for x in vd])} | {med(l)} | {med(sh)}% |")
P("")

if runs:
    P("## 사이클 (runs.jsonl 최근 5)")
    P("| at | 상태 | OK | EMPTY | SKIPPED | 격리 | ERROR | 청크 | 소요(초) |"); P("|---|---|---|---|---|---|---|---|---|")
    for r in runs[-5:]:
        P(f"| {r.get('at', '')[:16]} | {'stopped' if r.get('stopped') else 'done'} | {r.get('ok')} | {r.get('empty')} | {r.get('skipped')} | {r.get('quarantined')} | {r.get('error')} | {r.get('total_chunks')} | {r.get('elapsed_s')} |")
    P("")
if (d / "daemon.json").exists():
    P("## daemon.json"); P("```"); P((d / "daemon.json").read_text(encoding="utf-8").strip()); P("```")
print("\n".join(out))
