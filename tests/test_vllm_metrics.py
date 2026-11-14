"""eval/bench/vllm_metrics.py — 히스토그램 분위수와 스냅샷 차분."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "vllm_metrics", REPO / "eval" / "bench" / "vllm_metrics.py")
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)


def _snapshot(path: Path, buckets: dict[str, int], total_s: float, aborts: int = 0) -> Path:
    """e2e 지연 히스토그램 하나 + 결과 사유 카운터. buckets는 경계 → 누적 수."""
    lab = 'engine="0",model_name="m"'
    lines = ["# HELP vllm:e2e_request_latency_seconds x", "# TYPE x histogram"]
    for le, c in buckets.items():
        lines.append(f'vllm:e2e_request_latency_seconds_bucket{{{lab},le="{le}"}} {float(c)}')
    n = buckets["+Inf"]
    lines += [
        f"vllm:e2e_request_latency_seconds_sum{{{lab}}} {total_s}",
        f"vllm:e2e_request_latency_seconds_count{{{lab}}} {float(n)}",
        f"vllm:e2e_request_latency_seconds_created{{{lab}}} 1.7e9",
        f'vllm:request_success_total{{{lab},finished_reason="stop"}} {float(n)}',
        f'vllm:request_success_total{{{lab},finished_reason="abort"}} {float(aborts)}',
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_quantile_interpolates_between_edges():
    cum = {1.0: 0.0, 2.0: 50.0, 4.0: 100.0, float("inf"): 100.0}
    assert vm.quantile(cum, 0.5) == (2.0, "")
    value, prefix = vm.quantile(cum, 0.75)
    assert prefix == "" and abs(value - 3.0) < 1e-9


def test_quantile_does_not_invent_values_outside_known_edges():
    # 전부 첫 버킷 안 — 아래 경계를 모르므로 보간하지 않고 "≤첫 경계"
    assert vm.quantile({0.3: 10.0, 1.0: 10.0, float("inf"): 10.0}, 0.5) == (0.3, "≤")
    # 마지막 유한 경계를 넘음 — ">그 경계"
    assert vm.quantile({1.0: 1.0, float("inf"): 10.0}, 0.9) == (1.0, ">")


def test_diff_isolates_the_window_between_snapshots(tmp_path):
    before = vm.parse(_snapshot(tmp_path / "a.txt",
                                {"1.0": 10, "10.0": 10, "+Inf": 10}, total_s=5.0))
    after = vm.parse(_snapshot(tmp_path / "b.txt",
                               {"1.0": 10, "10.0": 30, "+Inf": 30}, total_s=105.0, aborts=2))
    buckets, sums, scalars = vm.diff(after, before)
    e2e = "vllm:e2e_request_latency_seconds"
    assert sums[e2e] == [100.0, 20.0]  # 창 안의 20건, 평균 5초
    assert buckets[e2e][1.0] == 0.0 and buckets[e2e][10.0] == 20.0
    assert scalars["vllm:request_success_total[finished_reason=abort]"] == 2.0
    assert not any(k.endswith("_created") for k in sums)


def test_report_renders_table(tmp_path):
    data = vm.parse(_snapshot(tmp_path / "a.txt",
                              {"1.0": 5, "10.0": 10, "+Inf": 10}, total_s=30.0))
    out = io.StringIO()
    vm.report(data, out)
    text = out.getvalue()
    assert "요청 10건" in text
    assert "| e2e_request_latency_seconds | 3.00s |" in text
