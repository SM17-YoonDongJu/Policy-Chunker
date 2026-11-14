"""vLLM /metrics 스냅샷을 요청 단위 분포로 풀어 본다.

vLLM은 요청마다 대기·prefill·decode 시간과 입출력 토큰 수를 히스토그램으로 누적한다. 서버를 띄운
뒤의 누적이라, 실험 하나의 몫을 보려면 실행 전후 스냅샷을 받아 뺀다(after − before).

  # 스냅샷 받기 (클러스터 vLLM — 이미지에 curl이 없어 파이썬으로 읽는다)
  kubectl exec deploy/vllm -- python3 -c "import urllib.request as u; \\
    print(u.urlopen('http://localhost:8000/metrics').read().decode())" > a.txt

  python eval/bench/vllm_metrics.py a.txt          # 서버 기동 뒤 전체
  python eval/bench/vllm_metrics.py a.txt b.txt    # 두 스냅샷 사이의 몫

분위수는 버킷 경계 사이를 선형 보간한 근사값이다. 버킷이 성기면(예: 출력 토큰 500~1000) 그 구간 안의
값은 실제와 크게 다를 수 있으므로, 결론은 평균과 버킷 경계에 걸리는 비율로 낸다.
라벨(engine, model_name 등)은 합친다 — 이 서버는 모델 하나·엔진 하나다.

클라이언트가 먼저 끊은 요청(chunker VLM_TIMEOUT)은 여기 없다. vLLM 0.33은 연결이 끊기면 생성을
멈추지만 그 요청을 결과 사유 abort로도, 지연 히스토그램에도 남기지 않는다(2026-11-14 확인 — 5초에
끊은 7,777토큰 요청이 abort 0, 5,000토큰 초과 버킷 0으로 남았고, 끊지 않은 같은 요청은 버킷에
잡혔다). 그래서 지연 꼬리는 실제보다 짧게 보인다. 타임아웃은 chunker의 vlm_page
(outcome=timeout)로 센다.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)')
_LE = re.compile(r'le="([^"]+)"')
_LABEL = re.compile(r'(\w+)="([^"]*)"')

# (메트릭, 단위, 뜻) — 표에 나오는 순서
HISTOGRAMS = [
    ("vllm:e2e_request_latency_seconds", "s", "요청 전체(서버 도착 → 끝)"),
    ("vllm:request_queue_time_seconds", "s", "대기열(스케줄 전)"),
    ("vllm:request_prefill_time_seconds", "s", "prefill(이미지 인코딩 + 입력 토큰)"),
    ("vllm:request_decode_time_seconds", "s", "decode(출력 토큰 생성)"),
    ("vllm:time_to_first_token_seconds", "s", "첫 토큰까지"),
    ("vllm:inter_token_latency_seconds", "s", "토큰 사이 간격(스텝 단위)"),
    ("vllm:request_prompt_tokens", "tok", "입력 토큰(이미지 포함)"),
    ("vllm:request_generation_tokens", "tok", "출력 토큰"),
    ("vllm:iteration_tokens_total", "tok", "스텝당 처리 토큰(배치 크기 감)"),
]
COUNTERS = [
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
]


def parse(path: Path) -> tuple[dict, dict, dict]:
    """→ (히스토그램 {이름: {le: 누적수}}, 합계 {이름: (sum, count)}, 스칼라 {이름[라벨]: 값})."""
    buckets: dict[str, dict[float, float]] = defaultdict(lambda: defaultdict(float))
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    scalars: dict[str, float] = defaultdict(float)
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        if name.endswith("_created"):
            continue
        if name.endswith("_bucket"):
            le = _LE.search(labels)
            if le:
                buckets[name[:-7]][float(le.group(1))] += value
        elif name.endswith("_sum"):
            sums[name[:-4]][0] += value
        elif name.endswith("_count"):
            sums[name[:-6]][1] += value
        else:
            # 결과 사유처럼 라벨이 뜻을 갖는 카운터는 그 라벨을 살린다
            keep = [f"{k}={v}" for k, v in _LABEL.findall(labels)
                    if k in ("finished_reason", "source", "handler", "status")]
            scalars[name + (f"[{','.join(keep)}]" if keep else "")] += value
    return buckets, sums, scalars


def diff(after: tuple, before: tuple | None) -> tuple:
    if before is None:
        return after
    ab, asum, asc = after
    bb, bsum, bsc = before
    buckets = {n: {le: c - bb.get(n, {}).get(le, 0.0) for le, c in les.items()}
               for n, les in ab.items()}
    sums = {n: [s - bsum.get(n, [0, 0])[0], c - bsum.get(n, [0, 0])[1]]
            for n, (s, c) in asum.items()}
    scalars = {k: v - bsc.get(k, 0.0) for k, v in asc.items()}
    return buckets, sums, scalars


def quantile(cum: dict[float, float], q: float) -> tuple[float, str]:
    """누적 버킷에서 q 분위수 → (값, 표시 접두어).

    경계 사이는 선형 보간한다. 첫 버킷 안이면 아래 경계를 모르므로 "≤첫 경계"로, 마지막 유한 경계를
    넘으면 ">그 경계"로 낸다 — 보간값을 지어내지 않는다.
    """
    edges = sorted(cum)
    total = cum[edges[-1]]
    if total <= 0:
        return math.nan, ""
    target = q * total
    prev_le, prev_c = None, 0.0
    for le in edges:
        c = cum[le]
        if c >= target:
            if prev_le is None:
                return le, "≤"
            if math.isinf(le):
                return prev_le, ">"
            return prev_le + (le - prev_le) * (target - prev_c) / (c - prev_c), ""
        prev_le, prev_c = le, c
    return prev_le, ">"


def share_over(cum: dict[float, float], edge: float) -> float:
    """edge 경계를 넘은 비율. edge는 버킷 경계여야 정확하다."""
    edges = sorted(cum)
    total = cum[edges[-1]]
    below = max((cum[le] for le in edges if le <= edge), default=0.0)
    return (total - below) / total if total else math.nan


def fmt(v: float, unit: str) -> str:
    if math.isnan(v):
        return "-"
    if unit == "s":
        return f"{v * 1000:.0f}ms" if v < 1 else f"{v:.2f}s"
    return f"{v:,.0f}"


def report(data: tuple, out=sys.stdout) -> None:
    buckets, sums, scalars = data
    w = out.write
    n = sums.get("vllm:e2e_request_latency_seconds", [0, 0])[1]
    w(f"요청 {n:,.0f}건\n\n")

    w("| 구간 | 평균 | p50 | p90 | p99 | 뜻 |\n|---|---:|---:|---:|---:|---|\n")
    for name, unit, desc in HISTOGRAMS:
        if name not in buckets:
            continue
        s, c = sums.get(name, [0, 0])
        mean = s / c if c else math.nan
        qs = [quantile(buckets[name], q) for q in (0.5, 0.9, 0.99)]
        short = name.removeprefix("vllm:")
        w(f"| {short} | {fmt(mean, unit)} | " + " | ".join(pre + fmt(x, unit) for x, pre in qs)
          + f" | {desc} |\n")

    # 요청 시간이 어디서 쓰였나 — 평균의 비율(대기 + prefill + decode ≈ 전체, 나머지는 전처리·응답)
    def mean_of(name: str) -> float:
        s, c = sums.get(name, [0, 0])
        return s / c if c else math.nan

    e2e = mean_of("vllm:e2e_request_latency_seconds")
    parts = {
        "대기열": mean_of("vllm:request_queue_time_seconds"),
        "prefill": mean_of("vllm:request_prefill_time_seconds"),
        "decode": mean_of("vllm:request_decode_time_seconds"),
    }
    if e2e and not math.isnan(e2e):
        rest = e2e - sum(v for v in parts.values() if not math.isnan(v))
        w("\n요청 평균 시간의 구성: ")
        w(", ".join(f"{k} {v / e2e:.0%}" for k, v in parts.items() if not math.isnan(v)))
        w(f", 그 밖(이미지 전처리·토크나이즈·응답) {rest / e2e:.0%}\n")

    gen = buckets.get("vllm:request_generation_tokens")
    if gen:
        w("\n출력 토큰 꼬리: " + ", ".join(
            f">{int(e)} {share_over(gen, e):.1%}" for e in (500, 1000, 2000) if e in gen) + "\n")
    e2eb = buckets.get("vllm:e2e_request_latency_seconds")
    if e2eb:
        edges = [e for e in (30.0, 60.0, 120.0) if e in e2eb]
        if edges:
            w("요청 시간 꼬리: "
              + ", ".join(f">{int(e)}s {share_over(e2eb, e):.2%}" for e in edges) + "\n")

    w("\n| 카운터 | 값 |\n|---|---:|\n")
    for k in sorted(scalars):
        if any(k.startswith(c) for c in COUNTERS) or k.startswith("vllm:request_success_total"):
            w(f"| {k.removeprefix('vllm:')} | {scalars[k]:,.0f} |\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("first", type=Path, help="스냅샷(하나만 주면 서버 기동 뒤 전체)")
    p.add_argument("second", type=Path, nargs="?", help="나중 스냅샷 — 주면 second − first")
    a = p.parse_args(argv)
    if a.second:
        report(diff(parse(a.second), parse(a.first)))
    else:
        report(parse(a.first))


if __name__ == "__main__":
    main()
