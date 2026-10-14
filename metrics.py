"""실행 이력에서 운영 지표를 뽑는다.

runlog가 남긴 items.jsonl / runs.jsonl을 읽어 처리 시간 분포·단계별 비중·성공률·멱등
스킵률을 계산한다. 여태 이 수치들은 어디에도 없었다 — 로그는 링버퍼라 사라지고, 집계는
stdout으로만 나갔다.

사용:
  python metrics.py                       # 전체 기간
  python metrics.py --last 5              # 최근 5개 사이클
  python metrics.py --json                # 기계가 읽을 형태(대시보드·알림에 물릴 때)

환경변수: INGEST_STATE_DIR (기본 /data/state)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Optional

import runlog


def _read_jsonl(fname: str) -> list[dict[str, Any]]:
    path = runlog.state_dir() / fname
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 쓰다 만 줄은 건너뛴다(기록 중 종료된 경우)
    return out


def _pct(values: list[float], q: float) -> float:
    """가장 가까운 순위 방식 백분위수. 표본이 적어 보간은 과하다."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def stage_breakdown(docs: list[dict[str, Any]], cpu_workers: int = 1,
                    gpu_parallel: int = 1) -> dict[str, Any]:
    """문서별 stages 기록을 합쳐 단계별 비중과 병렬화 하한을 낸다.

    docs는 runlog items나 dry-run 결과처럼 elapsed_s·stages를 가진 dict 목록이다.

    하한(t_bound)은 '자원을 완벽하게 겹쳐 써도 이보다 빨라질 수 없는 시간'이다 —
    max(CPU 단계 합 / CPU 작업자 수, GPU 단계 합 / GPU 동시도). 실제 시간(t_actual)과의
    차이가 단계 겹치기·병렬화로 줄일 수 있는 몫이고, 하한 자체를 내리려면 단계의 단가
    (표 탐지 범위, DPI, 엔진)를 바꿔야 한다. 둘을 가르는 게 이 수치의 쓰임이다.
    """
    per_stage: dict[str, list[float]] = defaultdict(list)
    resource: dict[str, str] = {}
    by_resource: dict[str, float] = defaultdict(float)
    vlm_lat: list[float] = []
    vlm_outcomes: Counter = Counter()
    vlm_truncated = 0
    measured = [d for d in docs if d.get("stages")]
    for d in measured:
        for name, st in d["stages"].items():
            secs = float(st.get("seconds") or 0.0)
            per_stage[name].append(secs)
            resource[name] = st.get("resource", "cpu")
            by_resource[resource[name]] += secs
        v = d["stages"].get("vlm") or {}
        vlm_lat.extend(v.get("latencies") or [])
        vlm_outcomes.update(v.get("outcomes") or {})
        vlm_truncated += int(v.get("truncated") or 0)

    total = sum(by_resource.values()) or 1.0
    t_actual = round(sum(float(d.get("elapsed_s") or 0.0) for d in measured), 1)
    cpu_bound = by_resource.get("cpu", 0.0) / max(cpu_workers, 1)
    gpu_bound = by_resource.get("gpu", 0.0) / max(gpu_parallel, 1)
    t_bound = round(max(cpu_bound, gpu_bound) + by_resource.get("io", 0.0), 1)
    calls = sum(vlm_outcomes.values())
    bottleneck = "gpu" if gpu_bound > cpu_bound else "cpu"
    # 단계별 하한 = 그 단계 시간 / 그 자원의 병렬도. 병목 자원에 속한 단계들의 하한 합이 곧 전체
    # 하한(IO 제외)이다 — 그래서 '전체 하한을 줄이려면 어느 단계를 얼마나 줄여야 하는가'가 여기서
    # 바로 읽힌다. 병목이 아닌 자원의 단계는 아무리 줄여도 전체 하한이 그대로다(critical=False).
    parallel = {"cpu": max(cpu_workers, 1), "gpu": max(gpu_parallel, 1), "io": 1}
    bound_total = t_bound or 1.0
    stages: dict[str, dict[str, Any]] = {}
    for k, v in sorted(per_stage.items(), key=lambda kv: -sum(kv[1])):
        res = resource[k]
        stage_bound = sum(v) / parallel.get(res, 1)
        critical = res in (bottleneck, "io")
        stages[k] = {"resource": res, "total_s": round(sum(v), 1),
                     "share": round(sum(v) / total, 3), "p50_s": _pct(v, 0.5),
                     "bound_s": round(stage_bound, 1), "critical": critical,
                     # 전체 하한에서 이 단계가 차지하는 몫. 병목 밖의 단계는 0이다
                     "bound_share": round(stage_bound / bound_total, 3) if critical else 0.0}
    return {
        "n": len(measured),
        "stages": stages,
        "by_resource": {k: round(v, 1) for k, v in by_resource.items()},
        "bound": {
            "t_actual_s": t_actual, "t_bound_s": t_bound,
            "gap_s": round(t_actual - t_bound, 1),
            "bottleneck": bottleneck,
            # 병목 자원 안에서 하한을 가장 많이 차지하는 단계 — 줄이면 전체 하한이 같이 준다
            "limiting_stage": next((k for k, st in stages.items()
                                    if st["resource"] == bottleneck), None),
            "cpu_workers": cpu_workers, "gpu_parallel": gpu_parallel,
        },
        "vlm": {
            "calls": calls, "outcomes": dict(vlm_outcomes),
            "timeout_rate": round(vlm_outcomes["timeout"] / calls, 3) if calls else None,
            # max_tokens에서 잘린 응답 — 생성 시간을 상한까지 쓰고 표 끝을 잃는다
            "truncated_rate": round(vlm_truncated / calls, 3) if calls else None,
            "latency_p50_s": _pct(vlm_lat, 0.5) if vlm_lat else None,
            "latency_p95_s": _pct(vlm_lat, 0.95) if vlm_lat else None,
        },
    }


def collect(last: Optional[int] = None, cpu_workers: int = 1,
            gpu_parallel: int = 1) -> dict[str, Any]:
    items, runs = _read_jsonl("items.jsonl"), _read_jsonl("runs.jsonl")
    if last:
        runs = runs[-last:]
        if runs:
            items = [i for i in items if i["at"] >= runs[0]["at"]]

    status = Counter(i["status"] for i in items)
    attempted = sum(status[s] for s in ("OK", "EMPTY", "ERROR"))
    # SKIPPED = doc_hash 멱등으로 다운로드조차 안 한 건. 이 비율이 곧 중복 제거가 아낀 양이다.
    considered = attempted + status["SKIPPED"] + status["QUARANTINED"]

    done = [i for i in items if i["status"] == "OK"]
    elapsed = [i["elapsed_s"] for i in done if i.get("elapsed_s")]
    phases: dict[str, list[float]] = defaultdict(list)
    for i in done:
        for k, v in (i.get("phases") or {}).items():
            phases[k].append(v)

    total_phase = sum(sum(v) for v in phases.values()) or 1.0
    return {
        "cycles": {
            "count": len(runs),
            "elapsed_s": {"p50": _pct([r.get("elapsed_s", 0) for r in runs], 0.5),
                          "max": max((r.get("elapsed_s", 0) for r in runs), default=0)},
        },
        "documents": {
            "considered": considered,
            "by_status": dict(status),
            "success_rate": round(status["OK"] / attempted, 3) if attempted else None,
            "idempotent_skip_rate": (round(status["SKIPPED"] / considered, 3)
                                     if considered else None),
        },
        "per_document_seconds": {
            "n": len(elapsed),
            "p50": _pct(elapsed, 0.5), "p95": _pct(elapsed, 0.95),
            "max": max(elapsed, default=0),
        },
        "phase_share": {k: {"total_s": round(sum(v), 1),
                            "share": round(sum(v) / total_phase, 3),
                            "p50_s": _pct(v, 0.5)}
                        for k, v in sorted(phases.items(), key=lambda kv: -sum(kv[1]))},
        "chunks": {
            "total": sum(i.get("chunks", 0) for i in done),
            "per_second": (round(sum(i.get("chunks", 0) for i in done) / sum(elapsed), 1)
                           if sum(elapsed) else None),
        },
        "freshness": runlog.daemon_state().get("last_success_at"),
        # stages는 단계 분리 이후 문서에만 있다. 그 전 이력은 n에서 빠진다.
        "stage_breakdown": stage_breakdown(done, cpu_workers, gpu_parallel),
    }


def _print(m: dict[str, Any]) -> None:
    d, p = m["documents"], m["per_document_seconds"]
    print(f"사이클 {m['cycles']['count']}회 | 마지막 성공 {m['freshness'] or '없음'}")
    print(f"\n문서 {d['considered']}건 — {d['by_status']}")
    if d["success_rate"] is not None:
        print(f"  시도 대비 성공률   {d['success_rate']:.1%}")
    if d["idempotent_skip_rate"] is not None:
        print(f"  멱등 스킵률        {d['idempotent_skip_rate']:.1%}  "
              f"(doc_hash 중복 제거로 다운로드·파싱을 아예 안 한 비율)")
    print(f"\n문서당 처리 시간(n={p['n']})  "
          f"p50 {p['p50']}s  p95 {p['p95']}s  max {p['max']}s")
    if m["phase_share"]:
        print("\n단계별 비중 — 여기가 병목 후보다")
        for k, v in m["phase_share"].items():
            bar = "█" * round(v["share"] * 40)
            print(f"  {k:<6} {v['share']:>6.1%} {bar} (합 {v['total_s']}s, p50 {v['p50_s']}s)")
    sb = m.get("stage_breakdown") or {}
    if sb.get("n"):
        print_stage_breakdown(sb)
    c = m["chunks"]
    rate = f" | {c['per_second']}청크/초" if c["per_second"] else ""
    print(f"\n적재 청크 {c['total']:,}개{rate}")


def print_stage_breakdown(sb: dict[str, Any]) -> None:
    b, v = sb["bound"], sb["vlm"]
    print(f"\n단계별 비중(n={sb['n']}) — 자원 태그로 하한을 계산한다")
    for k, st in sb["stages"].items():
        bar = "█" * round(st["share"] * 40)
        print(f"  {k:<11} {st['resource']:<4} {st['share']:>6.1%} {bar} "
              f"(합 {st['total_s']}s, p50 {st['p50_s']}s)")
    print("\n단계별 하한 — ★는 전체 하한을 결정하는 단계(병목 자원·IO)")
    for k, st in sb["stages"].items():
        mark = "★" if st["critical"] else " "
        share = f"하한의 {st['bound_share']:.1%}" if st["critical"] else "줄여도 하한 불변"
        print(f"  {mark} {k:<11} {st['resource']:<4} {st['bound_s']:>9}s  {share}")
    print(f"\n실제 {b['t_actual_s']}s | 하한 {b['t_bound_s']}s | 차이 {b['gap_s']}s "
          f"| 병목 {b['bottleneck']} → {b['limiting_stage']} (CPU 작업자 {b['cpu_workers']}, "
          f"GPU 동시도 {b['gpu_parallel']})")
    if v["calls"]:
        print(f"VLM 호출 {v['calls']}건 {v['outcomes']} | 타임아웃 {v['timeout_rate']:.1%} "
              f"| 잘림 {v['truncated_rate']:.1%} "
              f"| p50 {v['latency_p50_s']}s p95 {v['latency_p95_s']}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="인덱싱 실행 이력 → 운영 지표")
    ap.add_argument("--last", type=int, default=None, help="최근 N개 사이클만")
    ap.add_argument("--json", action="store_true", help="JSON으로 출력")
    ap.add_argument("--cpu-workers", type=int, default=1,
                    help="하한 계산에 쓸 CPU 작업자 수(INGEST_CONCURRENCY에 맞춘다)")
    ap.add_argument("--gpu-parallel", type=int, default=1,
                    help="하한 계산에 쓸 GPU 동시도(OLLAMA_NUM_PARALLEL에 맞춘다)")
    args = ap.parse_args()

    m = collect(args.last, args.cpu_workers, args.gpu_parallel)
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    else:
        _print(m)


if __name__ == "__main__":
    main()
