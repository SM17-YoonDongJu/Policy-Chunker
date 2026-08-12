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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("worker")

_INTERVAL = int(os.environ.get("INGEST_INTERVAL_SECONDS", "604800"))  # 기본 7일
_SOURCE = os.environ.get("INGEST_SOURCE", "catalog").strip().lower()
_MANIFEST = os.environ.get("MANIFEST_PATH", "/data/docs.yaml")
_RUN_ON_START = os.environ.get("RUN_ON_START", "1").strip().lower() in ("1", "true", "yes")

_stop = threading.Event()


def _handle_signal(signum: int, _frame: object) -> None:
    logger.info(f"시그널 {signal.Signals(signum).name} 수신 — 정지 예약(진행 중 작업은 마저 끝낸다)")
    _stop.set()


def _run(label: str, argv: list[str]) -> bool:
    """CLI를 subprocess로 실행. 성공 여부만 반환하고 예외는 삼킨다(데몬 생존 우선)."""
    logger.info(f"{label} 시작: {' '.join(argv)}")
    try:
        rc = subprocess.run(argv, cwd=Path(__file__).parent, check=False).returncode
    except Exception as e:  # noqa: BLE001 - 어떤 실패든 데몬은 계속 살아야 한다
        logger.error(f"{label} 실행 실패: {e}")
        return False
    if rc != 0:
        logger.error(f"{label} 실패 (exit={rc})")
        return False
    logger.info(f"{label} 완료")
    return True


def _cycle() -> None:
    """인덱싱 1회. 소스가 준비 안 됐으면 건너뛴다(데몬은 계속 떠 있는다)."""
    if _SOURCE == "manifest":
        if not Path(_MANIFEST).exists():
            logger.warning(f"매니페스트 없음: {_MANIFEST} — 이번 주기 건너뜀. "
                           "호스트 ~/insurance-chunker/data/docs.yaml에 매니페스트와 PDF를 두세요.")
            return
        label, argv = "ingest_many", [sys.executable, "ingest_many.py", "--manifest", _MANIFEST]
    else:
        label, argv = "ingest_catalog", [sys.executable, "ingest_catalog.py"]

    if not _run(label, argv):
        logger.error("적재 실패 — search_terms 재구성은 건너뛴다(부분 상태로 덮지 않기 위해)")
        return
    _run("rebuild_search_terms", [sys.executable, "rebuild_search_terms.py"])


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    src = f"매니페스트 {_MANIFEST}" if _SOURCE == "manifest" else "카탈로그(ai.corpus_file + S3)"
    logger.info(f"인덱싱 데몬 시작 — 주기 {_INTERVAL}s ({_INTERVAL / 3600:.1f}h), 소스: {src}")

    if _RUN_ON_START and not _stop.is_set():
        _cycle()

    while not _stop.is_set():
        # TZ(compose에서 Asia/Seoul)를 반영한 로컬 시각으로 표시한다.
        nxt = datetime.now(UTC).astimezone() + timedelta(seconds=_INTERVAL)
        logger.info(f"다음 실행 예정: {nxt:%Y-%m-%d %H:%M:%S} (대기 {_INTERVAL}s)")
        # wait()는 시그널로 즉시 깨어난다 — sleep과 달리 종료가 지연되지 않는다.
        if _stop.wait(_INTERVAL):
            break
        _cycle()

    logger.info("데몬 종료")


if __name__ == "__main__":
    main()
