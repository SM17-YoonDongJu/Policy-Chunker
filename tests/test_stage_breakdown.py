"""단계별 비중과 병렬화 하한(metrics.stage_breakdown).

하한은 '자원을 완벽하게 겹쳐도 이보다 빨라질 수 없는 시간'이다. 실제 시간과의 차이가
병렬화로 줄일 몫, 하한 자체가 단가를 바꿔야 줄일 몫이다. 계산이 틀리면 엉뚱한 쪽을 손댄다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import metrics  # noqa: E402


def _doc(elapsed, text=10.0, vlm=60.0, chunk=20.0, embed=10.0, outcomes=None, lat=None):
    return {"elapsed_s": elapsed, "stages": {
        "text": {"seconds": text, "resource": "cpu"},
        "vlm": {"seconds": vlm, "resource": "gpu", "outcomes": outcomes or {"table": 2},
                "latencies": lat or [30.0, 30.0]},
        "chunk": {"seconds": chunk, "resource": "cpu"},
        "embed": {"seconds": embed, "resource": "gpu"},
    }}


def test_sequential_pipeline_has_a_gap_equal_to_the_smaller_resource():
    """순차 실행이면 실제 = CPU + GPU, 하한 = 둘 중 큰 쪽. 차이가 겹치기로 얻을 몫이다."""
    sb = metrics.stage_breakdown([_doc(100.0)])
    b = sb["bound"]
    assert sb["by_resource"] == {"cpu": 30.0, "gpu": 70.0}
    assert b["t_actual_s"] == 100.0 and b["t_bound_s"] == 70.0 and b["gap_s"] == 30.0
    assert b["bottleneck"] == "gpu"


def test_gpu_parallelism_lowers_the_bound_until_cpu_takes_over():
    sb = metrics.stage_breakdown([_doc(100.0)], gpu_parallel=4)
    assert sb["bound"]["t_bound_s"] == 30.0
    assert sb["bound"]["bottleneck"] == "cpu"


def test_shares_sum_to_one_and_are_sorted():
    sb = metrics.stage_breakdown([_doc(100.0), _doc(90.0, vlm=50.0)])
    shares = [st["share"] for st in sb["stages"].values()]
    assert abs(sum(shares) - 1.0) < 0.01
    assert list(sb["stages"])[0] == "vlm"


def test_vlm_percentiles_pool_raw_latencies_across_documents():
    """문서별 p95를 다시 p95 내면 틀린다 — 원값을 합쳐 낸다."""
    docs = [_doc(10.0, lat=[1.0] * 19), _doc(10.0, lat=[100.0])]
    v = metrics.stage_breakdown(docs)["vlm"]
    assert v["latency_p50_s"] == 1.0 and v["latency_p95_s"] == 100.0


def test_timeout_rate():
    v = metrics.stage_breakdown([_doc(10.0, outcomes={"table": 3, "timeout": 1})])["vlm"]
    assert v["calls"] == 4 and v["timeout_rate"] == 0.25


def test_history_before_stage_split_is_left_out():
    """단계 분리 이전 이력엔 stages가 없다. 섞으면 실제 시간만 늘어 차이가 부풀려진다."""
    sb = metrics.stage_breakdown([_doc(100.0), {"elapsed_s": 500.0}])
    assert sb["n"] == 1 and sb["bound"]["t_actual_s"] == 100.0


def test_empty_input_does_not_crash():
    sb = metrics.stage_breakdown([])
    assert sb["n"] == 0 and sb["vlm"]["calls"] == 0


def test_truncated_rate():
    d = _doc(10.0, outcomes={"table": 4})
    d["stages"]["vlm"]["truncated"] = 1
    assert metrics.stage_breakdown([d])["vlm"]["truncated_rate"] == 0.25


# ── 단계별 하한 ───────────────────────────────────────────────────────────────

def test_stage_bounds_sum_to_the_total_bound_on_the_bottleneck():
    """병목 자원 단계들의 하한 합 + IO = 전체 하한. 어느 단계를 줄여야 하는지가 여기서 읽힌다."""
    d = _doc(100.0)
    d["stages"]["store"] = {"seconds": 5.0, "resource": "io"}
    sb = metrics.stage_breakdown([d], gpu_parallel=2)
    st = sb["stages"]
    assert st["vlm"]["bound_s"] == 30.0 and st["embed"]["bound_s"] == 5.0
    crit = sum(s["bound_s"] for s in st.values() if s["critical"])
    assert crit == sb["bound"]["t_bound_s"] == 40.0
    assert sb["bound"]["limiting_stage"] == "vlm"


def test_non_bottleneck_stages_do_not_move_the_bound():
    """GPU가 병목이면 CPU 단계를 아무리 줄여도 전체 하한은 그대로다 — 그걸 표시해야 한다."""
    st = metrics.stage_breakdown([_doc(100.0)])["stages"]
    assert st["text"]["critical"] is False and st["text"]["bound_share"] == 0.0
    assert st["vlm"]["critical"] is True and st["vlm"]["bound_share"] > 0.8


def test_bottleneck_flips_to_cpu_with_enough_gpu_parallelism():
    sb = metrics.stage_breakdown([_doc(100.0)], gpu_parallel=4)
    assert sb["bound"]["limiting_stage"] == "chunk"
    assert sb["stages"]["vlm"]["critical"] is False
