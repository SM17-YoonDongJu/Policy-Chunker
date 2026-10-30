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

## /status — 청크 뷰어용

같은 포트에 JSON을 하나 더 낸다. 청크 뷰어(viewer/)가 "지금 무엇을 처리 중인가 / 방금 무엇이
끝났나"를 묻는 경로다. 상태 파일은 이 파드의 PVC에만 있어 뷰어가 직접 읽을 수 없고, 원칙은
/metrics와 같다 — 앱이 상태를 들고 있지 않고 요청 시점에 runlog를 읽는다.
메트릭 전용 ClusterIP Service 뒤에 있으므로 클러스터 밖으로는 나가지 않는다.

환경변수:
  METRICS_PORT     리슨 포트. 기본 9101. 0이면 노출하지 않는다.
  METRICS_ADDR     바인드 주소. 기본 0.0.0.0 (파드 밖에서 스크랩 — 노출 범위는 메트릭
                   전용 ClusterIP Service가 정하고, ServiceMonitor가 그것만 긁는다).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
from collections import Counter
from datetime import datetime
from typing import Any, Iterator

import metrics as metrics_mod
import runlog

logger = logging.getLogger(__name__)

_PREFIX = "insurance_chunker"
# last_cycle_documents의 라벨 값 — 항상 전부 내보내 패널에 구멍이 안 생기게 한다.
_STATUSES = ("ok", "empty", "skipped", "quarantined", "error")
# vlm_calls_total의 라벨 값 — 0이어도 전부 내보낸다(타임아웃 비율 패널의 분모·분자).
_VLM_OUTCOMES = ("table", "empty", "timeout", "error")

# /status — 실제로 처리를 시도한 결과만 "최근 결과"로 본다. SKIPPED·QUARANTINED는 매 주기
# 카탈로그 전체 분량으로 쌓이는 생략 기록이라 섞으면 처리 결과가 묻힌다.
_PROCESSED = frozenset({"OK", "EMPTY", "ERROR"})
_RECENT_DEFAULT = 50
_RECENT_MAX = 500
# items.jsonl을 뒤에서 읽는 상한. 생략 기록 사이에서 처리 결과를 찾느라 파일 끝까지 가지 않게 한다.
_SCAN_MAX = 20000
# 뷰어에 넘길 문서 결과 필드. stages는 단계별 상세라 목록에는 무겁다.
_ITEM_FIELDS = ("at", "sha256", "name", "status", "chunks", "warnings", "elapsed_s", "phases",
                "error", "insurer", "product", "boundary_confidence", "rss_peak_mb")


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

        # phases의 parse는 텍스트·CPU 표·VLM·best-of를 한데 묶은 값이다. 그 안에서 GPU(VLM)가
        # 얼마인지는 단계(stages) 기록으로만 보인다. 단계 분리 이전 이력에는 없다.
        stage_totals, vlm_outcomes = _aggregate_stages()
        stage_fam = CounterMetricFamily(f"{_PREFIX}_stage_duration_seconds_total",
                                        "파이프라인 단계별 누적 소요(초)",
                                        labels=["stage", "resource"])
        for (stage, resource), secs in sorted(stage_totals.items()):
            stage_fam.add_metric([stage, resource], secs)
        yield stage_fam

        vlm = CounterMetricFamily(f"{_PREFIX}_vlm_calls_total",
                                  "결과별 누적 VLM 호출 수(table·empty·timeout·error)",
                                  labels=["outcome"])
        for outcome in _VLM_OUTCOMES:
            vlm.add_metric([outcome], float(vlm_outcomes.get(outcome, 0)))
        yield vlm


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


def _aggregate_stages() -> tuple[dict[tuple[str, str], float], Counter]:
    """items.jsonl의 stages를 접어 (단계, 자원)별 누적 초와 VLM 결과별 호출 수를 만든다."""
    totals: dict[tuple[str, str], float] = {}
    outcomes: Counter = Counter()
    for item in metrics_mod._read_jsonl("items.jsonl"):
        for name, st in (item.get("stages") or {}).items():
            key = (name, st.get("resource", "cpu"))
            totals[key] = totals.get(key, 0.0) + float(st.get("seconds") or 0.0)
        outcomes.update(((item.get("stages") or {}).get("vlm") or {}).get("outcomes") or {})
    return totals, outcomes


# ── /status ───────────────────────────────────────────────────────────────────

def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 살아 있지만 신호를 보낼 권한이 없다
    except OSError:
        return False
    return True


def _run_state(progress: dict[str, Any], daemon: dict[str, Any]) -> str:
    """마지막 실행의 상태: none | running | finished | stopped | interrupted."""
    if not progress:
        return "none"
    if progress.get("finished_at"):
        return "stopped" if progress.get("stopped") else "finished"
    # 끝 기록이 없다. 데몬이 그 뒤에 다시 떴으면 그 pid는 이미 다른 수명의 것이다.
    started = _iso_to_epoch(progress.get("started_at"))
    boot = _iso_to_epoch(daemon.get("started_at"))
    if started is not None and boot is not None and started < boot:
        return "interrupted"
    # 프로세스가 살아 있으면 진행 중, 아니면 결과를 못 남기고 죽었다(OOM·강제 종료).
    return "running" if _pid_alive(progress.get("pid")) else "interrupted"


def status_snapshot(limit: int = _RECENT_DEFAULT) -> dict[str, Any]:
    """지금 처리 중인 문서, 이번 실행의 진행률, 최근 처리 결과를 한 번에 돌려준다.

    attempts.json의 STARTED는 두 가지가 섞여 있다 — 지금 처리 중인 문서와, 지난 시도가 끝을
    못 본 문서. 이번 실행이 살아 있고 그 시작 뒤에 시작한 것만 processing으로, 나머지는
    unfinished로 가른다(다음 시도에서 격리 상한에 걸릴 수 있는 문서들이다).
    """
    daemon = runlog.daemon_state()
    progress = runlog.progress()
    state = _run_state(progress, daemon)
    run_start = _iso_to_epoch(progress.get("started_at")) if progress else None

    processing: list[dict[str, Any]] = []
    unfinished: list[dict[str, Any]] = []
    for sha, a in runlog.all_attempts().items():
        if a.get("status") != "STARTED":
            continue
        row = {"sha256": sha, "name": a.get("name"), "started_at": a.get("last_at"),
               "attempts": a.get("attempts", 0)}
        at = _iso_to_epoch(a.get("last_at"))
        if state == "running" and run_start is not None and at is not None and at >= run_start:
            processing.append(row)
        else:
            unfinished.append(row)
    processing.sort(key=lambda r: r["started_at"] or "")
    unfinished.sort(key=lambda r: r["started_at"] or "", reverse=True)

    # 최신부터 읽으며 최근 결과를 모으고, 이번 실행 안의 처리 건수(진행률 분자)를 센다.
    # 이번 실행의 기록은 파일 끝에 몰려 있으므로 실행 시작보다 오래된 줄에 닿고 최근 결과도
    # 다 모았으면 멈춘다.
    recent: list[dict[str, Any]] = []
    done = 0
    for item in runlog.iter_items_reverse(_SCAN_MAX):
        at = _iso_to_epoch(item.get("at"))
        in_run = run_start is not None and at is not None and at >= run_start
        if item.get("status") in _PROCESSED:
            done += in_run
            if len(recent) < limit:
                recent.append({k: item.get(k) for k in _ITEM_FIELDS})
        if len(recent) >= limit and not in_run:
            break

    run = None
    if progress:
        run = {"state": state, "done": done,
               **{k: progress.get(k) for k in ("started_at", "finished_at", "stopped", "total",
                                               "pending", "skipped", "quarantined",
                                               "concurrency")}}
    return {
        "generated_at": runlog.now_iso(),
        "interval_s": int(os.environ.get("INGEST_INTERVAL_SECONDS", "604800")),
        "daemon": daemon,
        "run": run,
        "processing": processing,
        "unfinished": unfinished,
        "recent": recent,
    }


def _handler_class(registry: Any) -> type:
    """/status만 따로 받고 나머지 경로는 prometheus_client의 기본 처리(/metrics)에 넘긴다."""
    from urllib.parse import parse_qs, urlsplit

    from prometheus_client.exposition import MetricsHandler

    class Handler(MetricsHandler):
        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path != "/status":
                return super().do_GET()
            try:
                raw = parse_qs(url.query).get("limit", [str(_RECENT_DEFAULT)])[0]
                limit = max(0, min(int(raw), _RECENT_MAX))
                body = json.dumps(status_snapshot(limit), ensure_ascii=False, default=str)
                code = 200
            except Exception as e:  # noqa: BLE001 - 조회 실패가 서버 스레드를 죽이면 안 된다
                body = json.dumps({"error": str(e)}, ensure_ascii=False)
                code = 400 if isinstance(e, ValueError) else 500
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    Handler.registry = registry
    return Handler


def _serve(addr: str, port: int, registry: Any) -> Any:
    """HTTP 서버를 데몬 스레드로 띄우고 돌려준다. 바인드 실패는 예외로 올린다(start가 삼킨다)."""
    from http.server import ThreadingHTTPServer

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if ":" in addr else socket.AF_INET

    server = Server((addr, port), _handler_class(registry))
    threading.Thread(target=server.serve_forever, name="metrics-http", daemon=True).start()
    return server


def start() -> bool:
    """백그라운드 스레드로 /metrics와 /status를 연다. 노출 여부를 돌려준다.

    실패해도 예외를 올리지 않는다 — 메트릭 노출이 안 된다고 인덱싱이 멈추면 안 된다.
    """
    port = int(os.environ.get("METRICS_PORT", "9101"))
    if port <= 0:
        logger.info("METRICS_PORT=0 — /metrics 노출 안 함")
        return False
    addr = os.environ.get("METRICS_ADDR", "0.0.0.0")  # noqa: S104 - 노출 범위는 ClusterIP Service
    try:
        from prometheus_client import REGISTRY

        # 기본 수집기(process_*, python_gc_*)는 그대로 둔다 — 데몬 자체의 생존·메모리
        # 추이를 보는 데 쓰인다.
        REGISTRY.register(RunlogCollector())
        _serve(addr, port, REGISTRY)
    except Exception as e:  # noqa: BLE001 - 노출 실패가 데몬을 죽이면 안 된다
        logger.warning(f"/metrics 노출 실패: {e}")
        return False
    logger.info(f"/metrics·/status 노출 — {addr}:{port}",
                extra={"event": "metrics_started", "port": port})
    return True
