"""Prometheus /metrics — runlog를 읽어 배치 잡 지표를 낸다.

배치 잡은 보통 Prometheus의 pull 모델과 안 맞는다. 주기적으로 떴다 죽는 프로세스는
스크랩 시점에 없을 수 있어서 Pushgateway나 node-exporter textfile collector를 끌어오게 된다.

우리는 그 문제가 없다 — 호스트 cron에서 컨테이너 상시 데몬으로 옮긴 덕에(2c4577c)
프로세스가 계속 떠 있으므로 pull이 그대로 성립한다. 그래서 추가 부품 없이 /metrics만 연다.

## 왜 커스텀 컬렉터인가

worker.py가 적재 CLI를 subprocess로 띄운다(사이클이 죽어도 데몬은 살리려는 설계). 그래서
prometheus_client를 적재 CLI 안에 심으면 프로세스 종료와 함께 메트릭이 사라진다. 대신
스크랩 시점에 runlog(/data/state)를 읽어 렌더링한다 — 앱이 상태를 들고 있지 않으니
재시작에도 정확하고, 이미 있는 계측을 그대로 쓴다.

## 왜 게이지 위주인가

주기가 7일(INGEST_INTERVAL_SECONDS=604800)이라 대부분의 시간 동안 카운터는 안 움직인다.
15초 스크랩에 rate(...[5m])는 거의 항상 0이고 그게 정상이다. 그래서 주 신호는 rate가
아니라 "마지막 실행 상태" 게이지다.

히스토그램은 두지 않는다 — 주당 문서 수백 건이면 Prometheus에서 p95를 낼 표본이 안 된다.
분포는 metrics.py와 Loki가 맡는다.

환경변수:
  METRICS_PORT     리슨 포트. 기본 9101. 0이면 노출하지 않는다.
  METRICS_ADDR     바인드 주소. 기본 0.0.0.0 (원격 스크랩 — 접근은 보안그룹으로 통제).
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime
from typing import Any, Iterator

import metrics as metrics_mod
import runlog

logger = logging.getLogger(__name__)

_PREFIX = "insurance_chunker"
# last_cycle_documents의 라벨 값 — 항상 전부 내보내 패널에 구멍이 안 생기게 한다.
_STATUSES = ("ok", "empty", "skipped", "quarantined", "error")


def _iso_to_epoch(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


class RunlogCollector:
    """스크랩마다 runlog를 읽어 지표를 만든다(상태를 들고 있지 않는다)."""

    def collect(self) -> Iterator[Any]:
        from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

        state = runlog.daemon_state()
        run = runlog.last_run() or {}

        def gauge(name: str, doc: str, value: float | None, **kw: Any):
            g = GaugeMetricFamily(f"{_PREFIX}_{name}", doc, **kw)
            if not kw.get("labels") and value is not None:
                g.add_metric([], value)
            return g

        # ── SLI ──────────────────────────────────────────────────────────────
        ts = _iso_to_epoch(state.get("last_success_at"))
        if ts is not None:
            yield gauge("last_success_timestamp_seconds",
                        "마지막 인덱싱 사이클이 성공한 시각(unix). 신선도 알림의 근거", ts)
        yield gauge("last_cycle_success",
                    "마지막 사이클 성패 (1=성공)", float(bool(state.get("last_run_ok"))))
        started = _iso_to_epoch(state.get("started_at"))
        if started is not None:
            yield gauge("start_timestamp_seconds", "데몬 기동 시각(unix)", started)

        # ── 마지막 사이클 결과 ────────────────────────────────────────────────
        yield gauge("last_cycle_duration_seconds",
                    "마지막 적재 실행 소요(초)", float(run.get("elapsed_s", 0)))
        yield gauge("last_cycle_chunks_indexed",
                    "마지막 적재 실행에서 저장한 청크 수", float(run.get("total_chunks", 0)))

        docs = GaugeMetricFamily(f"{_PREFIX}_last_cycle_documents",
                                 "마지막 적재 실행의 상태별 문서 수", labels=["status"])
        for st in _STATUSES:
            docs.add_metric([st], float(run.get(st, 0)))
        yield docs

        # ── 현재 상태 ─────────────────────────────────────────────────────────
        attempts = runlog._read_json("attempts.json", {})
        quarantined = sum(1 for a in attempts.values()
                          if a.get("attempts", 0) >= _max_retry())
        yield gauge("quarantined_documents",
                    "재시도 상한에 걸려 격리된 문서 수(누적 현재값)", float(quarantined))

        # 경계 검출이 약한 문서는 적재는 됐지만 섹션이 안 갈려 조번호가 어긋난다.
        # status는 OK로 남으므로 이 지표가 없으면 품질 저하가 성공으로 집계된다.
        weak = sum(1 for i in metrics_mod._read_jsonl("items.jsonl")
                   if i.get("boundary_confidence") == "weak")
        yield gauge("weak_boundary_documents",
                    "경계 검출 신뢰도가 낮게 판정된 문서 수(누적)", float(weak))

        # ── 장기 추세용 누적 ──────────────────────────────────────────────────
        totals, phase_totals, chunks_total = _aggregate_items()
        processed = CounterMetricFamily(f"{_PREFIX}_documents_processed_total",
                                        "상태별 누적 문서 처리 수", labels=["status"])
        for st in _STATUSES:
            processed.add_metric([st], float(totals.get(st.upper(), 0)))
        yield processed

        chunks = CounterMetricFamily(f"{_PREFIX}_chunks_indexed_total",
                                     "누적 저장 청크 수")
        chunks.add_metric([], float(chunks_total))
        yield chunks

        phases = CounterMetricFamily(f"{_PREFIX}_phase_duration_seconds_total",
                                     "단계별 누적 소요(초)", labels=["phase"])
        for phase, secs in sorted(phase_totals.items()):
            phases.add_metric([phase], secs)
        yield phases


def _max_retry() -> int:
    return int(os.environ.get("INGEST_MAX_RETRY", "3"))


def _aggregate_items() -> tuple[Counter, dict[str, float], int]:
    """items.jsonl 전체를 접어 누적치를 만든다.

    파일을 매 스크랩마다 읽는다. 주 1회 사이클에 문서 수백 건이라 줄 수가 연 단위로도
    수만 줄이고, 15초 스크랩에도 부담이 안 된다. 이보다 커지면 롤링 집계로 바꾼다.
    """
    totals: Counter = Counter()
    phase_totals: dict[str, float] = {}
    chunks_total = 0
    for item in metrics_mod._read_jsonl("items.jsonl"):
        totals[item.get("status", "")] += 1
        chunks_total += item.get("chunks", 0) or 0
        for phase, secs in (item.get("phases") or {}).items():
            phase_totals[phase] = phase_totals.get(phase, 0.0) + float(secs)
    return totals, phase_totals, chunks_total


def start() -> bool:
    """백그라운드 스레드로 /metrics를 연다. 노출 여부를 돌려준다.

    실패해도 예외를 올리지 않는다 — 메트릭 노출이 안 된다고 인덱싱이 멈추면 안 된다.
    """
    port = int(os.environ.get("METRICS_PORT", "9101"))
    if port <= 0:
        logger.info("METRICS_PORT=0 — /metrics 노출 안 함")
        return False
    addr = os.environ.get("METRICS_ADDR", "0.0.0.0")  # noqa: S104 - 접근은 보안그룹이 통제
    try:
        from prometheus_client import REGISTRY, start_http_server

        # 기본 수집기(process_*, python_gc_*)는 그대로 둔다 — 데몬 자체의 생존·메모리
        # 추이를 보는 데 쓰인다.
        REGISTRY.register(RunlogCollector())
        start_http_server(port, addr=addr)
    except Exception as e:  # noqa: BLE001 - 노출 실패가 데몬을 죽이면 안 된다
        logger.warning(f"/metrics 노출 실패: {e}")
        return False
    logger.info(f"/metrics 노출 — {addr}:{port}", extra={"event": "metrics_started",
                                                        "port": port})
    return True
