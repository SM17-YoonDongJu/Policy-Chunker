"""VLM 동시도 자동 조정 — vLLM 대기열을 보고 동시 호출 수를 줄이고 늘린다.

vLLM을 AI 레포의 OCR worker와 나눠 쓴다. 이 파이프라인은 백로그를 미는 배치 작업이고 OCR worker는
그렇지 않으므로, 서버가 붐비면 이쪽이 물러선다(OCR worker 우선). 신호는 vLLM /metrics의
num_requests_waiting 하나다 — KV 캐시 여유를 넘긴 요청이 대기열에 쌓이기 시작하면 서버가 다 받지
못하고 있다는 뜻이고, 그 대기는 양쪽의 읽기 타임아웃에 그대로 들어간다.

  - 대기 > 0: 상한을 절반으로(최소 VLM_CONCURRENCY_MIN). 붐빔이 이어지면 폴링마다 다시 절반이다
  - 대기 0이 VLM_ADAPT_COOLDOWN번 이어지고, 그사이 이쪽이 상한에 막혀 기다린 적이 있으면 +1
    (최대 VLM_CONCURRENCY). 막힌 적이 없으면 올릴 이유가 없다 — 표 영역이 드문 구간이다
  - /metrics를 못 읽으면 상한을 그대로 둔다(올리지도 내리지도 않는다). 로그는 한 번만 남긴다

상한은 프로세스 하나 안에서 모든 문서가 나눠 쓴다. 문서 병렬(INGEST_CONCURRENCY)은 프로세스
단위라 프로세스마다 조정기가 따로 돌고, 같은 대기열을 보고 같이 물러선다.

결과는 바꾸지 않는다 — 언제 부르느냐만 바뀐다. 동시도 1(상한 1)이면 조정기를 만들지 않는다.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("insurance_chunker.extractor")  # 운영 로그 쿼리가 이 이름을 본다



def settings() -> dict:
    """설정은 부를 때마다 읽는다(extractor처럼 모듈 재로드에 기대지 않게).

    VLM_ADAPTIVE         "1"(기본)이면 자동 조정, "0"이면 VLM_CONCURRENCY 고정
    VLM_CONCURRENCY_MIN  내려갈 수 있는 바닥(기본 1)
    VLM_ADAPT_INTERVAL   폴링 간격(초, 기본 2). 표 영역 하나가 p50 3~4초(c16)라 그보다 짧게
    VLM_ADAPT_COOLDOWN   내린 뒤 다시 올리기 시작할 때까지 '대기 0' 폴링 수(기본 3). OCR worker의
                         순간 몰림이 지나가기 전에 다시 밀어 넣지 않게 한다
    """
    return {"adaptive": os.environ.get("VLM_ADAPTIVE", "1") == "1",
            "floor": max(1, int(os.environ.get("VLM_CONCURRENCY_MIN", "1"))),
            "interval": float(os.environ.get("VLM_ADAPT_INTERVAL", "2")),
            "cooldown": max(1, int(os.environ.get("VLM_ADAPT_COOLDOWN", "3")))}


def parse_waiting(text: str) -> Optional[int]:
    """vLLM /metrics 본문에서 대기 요청 수의 합. 모델·엔진 라벨이 여럿이면 더한다. 없으면 None."""
    total = None
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_waiting"):
            try:
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
    return None if total is None else int(total)


class AdaptiveLimit:
    """크기를 바꿀 수 있는 세마포어 + 조정 규칙. 작업 스레드가 VLM 호출을 이 안에서 한다."""

    def __init__(self, ceiling: int, floor: int = 1, cooldown: int = 3) -> None:
        self.ceiling = max(1, ceiling)
        self.floor = max(1, min(floor, self.ceiling))
        self.cooldown = cooldown
        self.limit = self.ceiling
        self.inflight = 0
        self._calm = cooldown      # 시작은 이미 '조용한' 상태로 본다
        self._blocked = False      # 지난 폴링 이후 상한에 막혀 기다린 적이 있는가
        self._cond = threading.Condition()

    def acquire(self) -> None:
        with self._cond:
            if self.inflight >= self.limit:
                self._blocked = True
            while self.inflight >= self.limit:
                self._cond.wait()
            self.inflight += 1

    def release(self) -> None:
        with self._cond:
            self.inflight -= 1
            self._cond.notify()

    def observe(self, waiting: Optional[int]) -> Optional[tuple[int, int]]:
        """폴링 한 번의 판정. 상한이 바뀌면 (이전, 이후), 아니면 None."""
        with self._cond:
            old = self.limit
            if waiting is None:
                return None
            if waiting > 0:
                self._calm = 0
                self.limit = max(self.floor, self.limit // 2)
            else:
                self._calm += 1
                if self._calm >= self.cooldown and self._blocked:
                    self.limit = min(self.ceiling, self.limit + 1)
            self._blocked = False
            if self.limit > old:
                self._cond.notify(self.limit - old)
            return (old, self.limit) if self.limit != old else None


class _Poller:
    def __init__(self, lim: AdaptiveLimit, url: str, interval: float) -> None:
        self.lim, self.url, self.interval = lim, url, interval
        self._stop = threading.Event()
        self._unavailable_logged = False
        self._thread = threading.Thread(target=self._loop, name="vlm-adapt", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> None:
        import requests
        try:
            resp = requests.get(f"{self.url}/metrics", timeout=1.0)
            resp.raise_for_status()
            waiting = parse_waiting(resp.text)
        except Exception as e:  # noqa: BLE001 - 조정기 고장이 VLM 호출을 막으면 안 된다
            waiting = None
            reason = str(e)
        else:
            reason = "no vllm:num_requests_waiting"
        if waiting is None:
            if not self._unavailable_logged:
                self._unavailable_logged = True
                logger.warning(f"vLLM /metrics를 못 읽어 VLM 동시도를 {self.lim.limit}로 고정한다: "
                               f"{reason}", extra={"event": "vlm_adapt_unavailable",
                                                   "limit": self.lim.limit})
            return
        self._unavailable_logged = False
        changed = self.lim.observe(waiting)
        if changed:
            old, new = changed
            logger.info(f"VLM 동시도 {old} → {new} (vLLM 대기 {waiting})",
                        extra={"event": "vlm_concurrency_change", "old": old, "new": new,
                               "waiting": waiting})

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.poll_once()


_lock = threading.Lock()
_current: Optional[tuple[tuple, AdaptiveLimit, _Poller]] = None


def get() -> Optional[AdaptiveLimit]:
    """이 프로세스의 조정기. 자동 조정이 꺼졌거나 상한 1이면 None(고정 동시도 그대로)."""
    global _current
    from . import extractor
    cfg = settings()
    if not cfg["adaptive"] or extractor.VLM_BACKEND != "local" or extractor.VLM_CONCURRENCY <= 1:
        return None
    key = (extractor.VLM_URL, extractor.VLM_CONCURRENCY, cfg["floor"], cfg["interval"],
           cfg["cooldown"])
    with _lock:
        if _current is not None and _current[0] == key:
            return _current[1]
        if _current is not None:
            _current[2].stop()   # 설정이 바뀌었다(테스트의 모듈 재로드) — 새로 만든다
        lim = AdaptiveLimit(extractor.VLM_CONCURRENCY, cfg["floor"], cfg["cooldown"])
        _current = (key, lim, _Poller(lim, extractor.VLM_URL, cfg["interval"]))
        return lim


def reset_for_tests() -> None:
    global _current
    with _lock:
        if _current is not None:
            _current[2].stop()
        _current = None
