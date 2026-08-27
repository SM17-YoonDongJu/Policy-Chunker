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


def collect(last: Optional[int] = None) -> dict[str, Any]:
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
    c = m["chunks"]
    rate = f" | {c['per_second']}청크/초" if c["per_second"] else ""
    print(f"\n적재 청크 {c['total']:,}개{rate}")


def main() -> None:
    ap = argparse.ArgumentParser(description="인덱싱 실행 이력 → 운영 지표")
    ap.add_argument("--last", type=int, default=None, help="최근 N개 사이클만")
    ap.add_argument("--json", action="store_true", help="JSON으로 출력")
    args = ap.parse_args()

    m = collect(args.last)
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    else:
        _print(m)


if __name__ == "__main__":
    main()
