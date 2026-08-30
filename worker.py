"""정기 인덱싱 데몬.

호스트 cron 대신 컨테이너가 스스로 주기를 관리한다(brbs-corpus-worker와 같은 방식):
`docker ps`에 상시 노출되고, 로그는 `docker logs brbs-insurance-chunker`로 본다.

한 사이클 = 적재(신규/변경 문서만 doc_hash 멱등) → rebuild_search_terms(BM25 용어).
CLI를 subprocess로 부른다 — 한 사이클이 죽어도 데몬은 살아 다음 주기를 계속한다.

적재 소스는 두 가지다(INGEST_SOURCE).
  catalog  — corpus_worker가 S3에 스테이징한 약관을 ai.corpus_file 카탈로그로 받아온다(기본).
  manifest — 손으로 쓴 /data/docs.yaml의 로컬 PDF를 읽는다(카탈로그 권한이 없을 때의 우회로).

환경변수:
  INGEST_INTERVAL_SECONDS  주기(초). 기본 604800(7일).
  INGEST_SOURCE            catalog | manifest. 기본 catalog.
  MANIFEST_PATH            manifest 모드의 매니페스트 경로. 기본 /data/docs.yaml.
  RUN_ON_START             기동 즉시 1회 실행 여부. 기본 1.
  DISCORD_WEBHOOK_INGEST   사이클 결과 알림 웹훅. 없으면 알림만 건너뛴다.
  INGEST_NOTIFY            always | failure. 기본 always.
  METRICS_PORT             Prometheus /metrics 포트. 기본 9101, 0이면 노출 안 함.

사이클 결과는 runlog가 /data/state에 남긴다 — 로그(json-file 링버퍼)와 달리 지워지지 않아
성공률·처리시간·마지막 성공 시각 같은 지표를 나중에 뽑을 수 있고, healthcheck.py가 그
'마지막 성공 시각'으로 좀비 상태를 판정한다.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

import exporter
import logging_setup
import notify
import runlog
import shutdown
import slo

logging_setup.configure()
logger = logging.getLogger("worker")

_INTERVAL = int(os.environ.get("INGEST_INTERVAL_SECONDS", "604800"))  # 기본 7일
_SOURCE = os.environ.get("INGEST_SOURCE", "catalog").strip().lower()
_MANIFEST = os.environ.get("MANIFEST_PATH", "/data/docs.yaml")
_RUN_ON_START = os.environ.get("RUN_ON_START", "1").strip().lower() in ("1", "true", "yes")

# 진행 중인 CLI. 시그널 핸들러가 여기로 SIGTERM을 전달한다.
# RLock인 이유: 핸들러는 메인 스레드에서 실행되므로, 이 락을 잡은 채 신호를 받으면
# 일반 Lock에서는 자기 자신을 기다리며 굳는다(Popen 호출 구간이 실제로 그 창이다).
_proc_lock = threading.RLock()
_proc: Optional[subprocess.Popen] = None


def _forward_stop() -> None:
    """정지 신호를 진행 중인 CLI에 넘긴다.

    SIGTERM은 PID 1(이 프로세스)에만 온다 — 자식은 아무것도 못 받은 채 일하다 유예가
    끝나면 SIGKILL로 끊긴다. 전달해야 자식이 문서 경계에서 접고 이력을 남길 수 있다.
    """
    with _proc_lock:
        if _proc is None or _proc.poll() is not None:
            return
        logger.info(f"진행 중인 작업(pid={_proc.pid})에 SIGTERM 전달")
        try:
            _proc.send_signal(signal.SIGTERM)
        except OSError as e:  # 그 사이에 끝났으면 그만이다
            logger.warning(f"신호 전달 실패: {e}")


def _run(label: str, argv: list[str]) -> bool:
    """CLI를 subprocess로 실행. 성공 여부만 반환하고 예외는 삼킨다(데몬 생존 우선)."""
    global _proc
    logger.info(f"{label} 시작: {' '.join(argv)}")
    try:
        with _proc_lock:
            if shutdown.stopping():
                logger.info(f"{label} 시작 전에 정지 신호 — 실행하지 않는다")
                return False
            _proc = subprocess.Popen(argv, cwd=Path(__file__).parent)
        # Popen이 도는 동안 신호가 왔다면 핸들러는 아직 비어 있는 _proc을 보고 지나갔다.
        # 그 창을 여기서 메운다 — 안 그러면 자식이 신호를 영영 못 받는다.
        if shutdown.stopping():
            _forward_stop()
        rc = _proc.wait()
    except Exception as e:  # noqa: BLE001 - 어떤 실패든 데몬은 계속 살아야 한다
        logger.error(f"{label} 실행 실패: {e}")
        return False
    finally:
        with _proc_lock:
            _proc = None
    if rc != 0:
        logger.error(f"{label} 실패 (exit={rc})")
        return False
    logger.info(f"{label} 완료")
    return True


def _summary_fields(detail: str) -> dict[str, object]:
    """알림에 실을 항목. 적재 CLI가 남긴 마지막 실행 기록에서 수치를 가져온다."""
    fields: dict[str, object] = {"결과": detail, "소스": _SOURCE}
    run = runlog.last_run() or {}
    if run:
        fields["처리"] = (f"OK {run.get('ok', 0)} / 0청크 {run.get('empty', 0)} / "
                         f"생략 {run.get('skipped', 0)} / 격리 {run.get('quarantined', 0)} / "
                         f"실패 {run.get('error', 0)}")
        fields["적재 청크"] = f"{run.get('total_chunks', 0):,}개"
        fields["소요"] = f"{run.get('elapsed_s', 0)}s"
    state = runlog.daemon_state()
    if state.get("last_success_at"):
        fields["마지막 성공"] = state["last_success_at"]
    return fields


def _check_slo():
    """적재 결과가 검색에 쓸 만한 상태인지 확인한다.

    "사이클이 성공했다"와 "인덱스가 쓸 만하다"는 다르다 — 임베딩 차원이 어긋나거나
    문서 하나에 임베딩이 통째로 없어도 적재 자체는 성공으로 끝난다. 실패해도 사이클을
    실패로 만들지는 않는다(이미 적재는 됐다). 알림으로만 드러낸다.
    """
    try:
        return slo.run(interval_s=_INTERVAL)
    except Exception as e:  # noqa: BLE001 - 점검이 데몬을 죽이면 안 된다
        logger.warning(f"SLO 점검 실패: {e}", extra={"event": "slo_error"})
        return None


def _cycle() -> None:
    """인덱싱 1회. 소스가 준비 안 됐으면 건너뛴다(데몬은 계속 떠 있는다).

    결과는 반드시 runlog.record_cycle로 남긴다 — 이게 healthcheck의 판정 근거이자,
    "언제 마지막으로 인덱싱이 성공했나"에 답하는 유일한 기록이다.
    """
    if _SOURCE == "manifest":
        if not Path(_MANIFEST).exists():
            logger.warning(f"매니페스트 없음: {_MANIFEST} — 이번 주기 건너뜀. "
                           "호스트 ~/insurance-chunker/data/docs.yaml에 매니페스트와 PDF를 두세요.")
            # 성공이 아니므로 마지막 성공 시각을 갱신하지 않는다 → 계속 없으면 healthcheck가 잡는다.
            runlog.record_cycle(False, "매니페스트 없음 — 건너뜀")
            notify.notify("warning", "insurance-chunker 인덱싱",
                          {"결과": f"매니페스트 없음({_MANIFEST}) — 이번 주기 건너뜀"})
            return
        label, argv = "ingest_many", [sys.executable, "ingest_many.py", "--manifest", _MANIFEST]
    else:
        label, argv = "ingest_catalog", [sys.executable, "ingest_catalog.py"]

    ok = _run(label, argv)
    if shutdown.stopping():
        # 정지로 잘린 사이클이다. 성공으로 세면 부분 적재로 last_success_at이 갱신돼
        # healthcheck가 "최근에 성공했다"고 거짓말한다. 그렇다고 실패로만 두면 배포할
        # 때마다 실패 알림이 울린다 — 원인을 적어 warning으로 구분한다.
        detail = f"{label} 중단 — 정지 신호(배포·재시작). 남은 문서는 다음 주기로"
        logger.warning(detail, extra={"event": "cycle_interrupted", "step": label,
                                      "source": _SOURCE})
        runlog.record_cycle(False, detail)
        notify.notify("warning", "insurance-chunker 인덱싱", _summary_fields(detail))
        return

    if not ok:
        logger.error("적재 실패 — search_terms 재구성은 건너뛴다(부분 상태로 덮지 않기 위해)",
                     extra={"event": "cycle_done", "ok": False, "failed_step": label,
                            "source": _SOURCE})
        runlog.record_cycle(False, f"{label} 실패")
        notify.notify("failure", "insurance-chunker 인덱싱",
                      _summary_fields(f"{label} 실패 — search_terms 재구성 생략"))
        return

    terms_ok = _run("rebuild_search_terms", [sys.executable, "rebuild_search_terms.py"])
    slo_report = _check_slo()
    # 적재가 됐으면 사이클은 성공으로 본다. search_terms 실패는 BM25 용어가 잠시 낡을 뿐
    # 색인 자체는 갱신됐고, 여기서 실패로 처리하면 healthcheck가 과하게 운다.
    detail = "완료" if terms_ok else "적재 완료 · search_terms 재구성 실패"
    if slo_report and slo_report.violations:
        detail += f" · SLO 위반 {len(slo_report.violations)}건"
    logger.info(f"사이클 {detail}", extra={"event": "cycle_done", "ok": True,
                                          "search_terms_ok": terms_ok, "source": _SOURCE})
    runlog.record_cycle(True, detail)
    status = "success"
    if not terms_ok or (slo_report and slo_report.violations):
        status = "warning"
    fields = _summary_fields(detail)
    if slo_report and slo_report.violations:
        fields["SLO 위반"] = "\n".join(f"{c.name}: {c.detail}" for c in slo_report.violations)
    notify.notify(status, "insurance-chunker 인덱싱", fields)


def main() -> None:
    shutdown.install(on_stop=_forward_stop)

    src = f"매니페스트 {_MANIFEST}" if _SOURCE == "manifest" else "카탈로그(ai.corpus_file + S3)"
    logger.info(f"인덱싱 데몬 시작 — 주기 {_INTERVAL}s ({_INTERVAL / 3600:.1f}h), 소스: {src}")
    # healthcheck가 첫 사이클 유예를 계산하는 기준점.
    runlog.record_start()
    logger.info(f"상태 디렉터리: {runlog.state_dir()}")
    # 스크랩은 데몬이 떠 있는 내내 받는다 — 사이클이 안 도는 6일 23시간에도 신선도를
    # 답해야 하므로 사이클과 무관하게 상시 열어둔다.
    exporter.start()

    if _RUN_ON_START and not shutdown.stopping():
        _cycle()

    while not shutdown.stopping():
        # TZ(compose에서 Asia/Seoul)를 반영한 로컬 시각으로 표시한다.
        nxt = datetime.now(UTC).astimezone() + timedelta(seconds=_INTERVAL)
        logger.info(f"다음 실행 예정: {nxt:%Y-%m-%d %H:%M:%S} (대기 {_INTERVAL}s)")
        # wait()는 시그널로 즉시 깨어난다 — sleep과 달리 종료가 지연되지 않는다.
        if shutdown.event().wait(_INTERVAL):
            break
        _cycle()

    logger.info("데몬 종료")


if __name__ == "__main__":
    main()
