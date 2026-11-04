"""문서 처리의 단계·쪽 진행을 바깥에 알리는 훅.

파이프라인은 긴 문서 하나에 수십 분을 쓰는데, 그동안 밖에서는 어느 쪽을 지나는지 보이지
않았다 — 뷰어의 "처리 중인 문서"가 시작 시각과 경과만 보여줘 멈춘 것인지 가는 중인지
구분할 수 없었다. 단계 모듈들이 쪽 루프에서 여기로 보고하면, 적재 쪽이 끼운 reporter가
그걸 기록한다(ingest_catalog → runlog.record_doc_progress).

라이브러리는 기록 방법을 모른다 — reporter를 안 끼우면 전부 no-op이다(단독 CLI·테스트·
벤치). 보고는 문서를 처리하는 스레드에서만 부른다(겹치기 경로에서도 레이아웃 스윕과 VLM
렌더·제출이 같은 호출 스레드다 — stages.overlapped). 워커 프로세스는 문서를 한 번에
하나만 맡으므로 모듈 전역으로 충분하다.

빈도는 시간으로만 묶는다(기본 1초). 단계가 바뀔 때 바로 쓰는 규칙은 두지 않는다 —
겹치기에선 레이아웃과 VLM 보고가 쪽마다 번갈아 와서, 전환마다 쓰면 묶은 의미가 없다.
쪽 루프가 없는 단계 전환(청킹·임베딩·저장)만 force로 바로 쓴다.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

Reporter = Callable[[str, Optional[int], Optional[int]], None]

# 같은 문서 안에서 이 간격보다 자주 쓰지 않는다 — 본문 추출처럼 쪽당 수 ms인 단계가
# 쪽마다 파일을 갈아치우지 않게. 뷰어 새로고침(기본 10초)보다 충분히 짧으면 된다.
_MIN_INTERVAL_S = 1.0

_reporter: Optional[Reporter] = None
_last_ts = float("-inf")  # 문서의 첫 보고는 항상 나간다


def set_reporter(fn: Optional[Reporter]) -> None:
    """문서 하나를 맡기기 직전에 끼우고, 끝나면 None으로 되돌린다."""
    global _reporter, _last_ts
    _reporter = fn
    _last_ts = float("-inf")


def report(stage: str, page: Optional[int], pages: Optional[int], *,
           force: bool = False) -> None:
    """지금 stage 단계의 page/pages쪽을 지나고 있다.

    보고 실패가 처리를 죽이면 안 된다 — reporter의 예외는 삼킨다.
    """
    global _last_ts
    if _reporter is None:
        return
    now = time.monotonic()
    if not force and now - _last_ts < _MIN_INTERVAL_S:
        return
    _last_ts = now
    try:
        _reporter(stage, page, pages)
    except Exception as e:  # noqa: BLE001 - 진행 보고 때문에 파이프라인이 죽으면 안 된다
        logger.debug(f"진행 보고 실패({stage} {page}/{pages}): {e}")
