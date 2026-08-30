"""정지 신호 처리 — 배포가 컨테이너를 재생성할 때 문서 경계에서 멈추기 위한 공용 스위치.

배포는 `docker compose up -d --force-recreate`로 컨테이너를 다시 만든다. 그때 SIGTERM은
PID 1(worker.py)에만 가고, stop_grace_period가 지나면 SIGKILL이 프로세스를 전부 끊는다.
그 사이에 하던 일을 접고 이력을 남기는 것이 여기 목적이다.

문서 하나를 중간에 끊지는 않는다. 저장 트랜잭션의 단위가 문서이고, 끊어봐야 다음 주기에
그 문서를 처음부터 다시 해야 하기 때문이다. 그래서 CLI는 문서와 문서 사이에서만
stopping()을 확인한다 — 여기서 약속하는 건 "곧 멈춘다"지 "즉시 멈춘다"가 아니다.

두 번째 신호는 기본 동작으로 되돌려 즉시 죽인다. 사람이 Ctrl-C를 두 번 눌렀다면
기다리지 않겠다는 뜻이다.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_stop = threading.Event()


def install(on_stop: Optional[Callable[[], None]] = None) -> None:
    """SIGTERM/SIGINT를 정지 예약으로 바꾼다.

    on_stop: 신호 즉시 해야 할 일(예: worker가 자식 프로세스에 SIGTERM 전달).
        핸들러 안에서 돌므로 짧아야 한다.
    """
    def _handle(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        if _stop.is_set():
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        _stop.set()
        logger.warning(f"{name} 수신 — 진행 중인 문서까지만 하고 멈춘다",
                       extra={"event": "stop_requested", "signal": name})
        if on_stop is not None:
            on_stop()

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _handle)


def stopping() -> bool:
    """정지가 예약됐는지. 문서와 문서 사이에서 확인한다."""
    return _stop.is_set()


def event() -> threading.Event:
    """대기용. wait()는 신호에 즉시 깨어난다 — sleep과 달리 종료가 지연되지 않는다."""
    return _stop


def request_stop() -> None:
    """신호 없이 같은 상태로 만든다(테스트·내부용)."""
    _stop.set()


def reset() -> None:
    """정지 상태를 되돌린다(테스트용)."""
    _stop.clear()
