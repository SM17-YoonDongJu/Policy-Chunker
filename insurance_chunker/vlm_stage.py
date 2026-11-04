"""VLM 단계 — 문서 하나의 VLM 대상 페이지를 정해진 동시도로 호출한다.

운영에서 VLM은 문서 처리 시간의 95~97%를 먹는다(2026-10-14 로그: parse 4,655~7,016초 중 대부분,
페이지당 p50 88.6초). 페이지 사이에 의존이 없는데도 한 장씩 순차로 불러 왔고, 서버(vLLM)는
continuous batching으로 여러 요청을 한 번에 디코딩할 수 있는데도 늘 한 건만 받았다.

이 클래스는 순차 루프가 하던 일(호출 상한·연속 실패 차단기·호출 기록)을 한곳에 모으고 동시 호출을
얹는다. 예전엔 그 상태가 extractor의 모듈 전역 변수였는데, 여러 스레드가 만지면 경합한다.

  - 렌더링은 호출한 스레드(문서를 연 스레드)에서 한다. PyMuPDF 객체는 스레드 간에 안전하지 않다.
    렌더된 PNG만 작업 스레드로 넘기고, 미리 렌더해 두는 양은 동시도 x 2로 묶어 메모리를 제한한다
  - 호출 상한(쪽 대상은 VISION_MAX_PAGES, crop 대상은 VISION_MAX_CROPS)은 대상 목록을
    미리 잘라 지킨다. 대상 순서가 곧 우선순위다
  - 차단기: 연속 VLM_FAIL_STREAK회 실패하면 남은 페이지는 부르지 않는다. 실패 집계와 차단 판정은
    작업 스레드 안에서 락을 잡고 한다 — 동시도 1이면 순차 루프와 정확히 같게 '기준에 닿은 다음
    호출'이 나가지 않는다. 동시도 N이면 이미 날아간 N-1건까지는 끝난다(되돌릴 수 없다)
  - 결과는 페이지 번호로 모으므로 완료 순서와 무관하다. 동시도를 바꿔도 같은 입력이면 같은 결과다

동시도는 VLM_CONCURRENCY가 상한이고, 자동 조정(vlm_limiter, 기본 켜짐)이면 작업 스레드가 호출 직전에
조정기의 자리를 얻는다 — vLLM을 OCR worker와 나눠 쓰므로 서버 대기열이 생기면 이쪽이 물러선다.

동시도의 상한은 서버가 정한다. vLLM은 KV 캐시 여유만큼만 동시에 처리하고(운영 설정에서 6,144토큰
요청 기준 3.17배), 넘치면 대기시킨다. 대기 시간도 클라이언트 읽기 타임아웃(VLM_TIMEOUT)에 들어가므로
동시도를 서버 여유보다 높이면 빨라지지 않고 타임아웃만 는다.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Iterable, Optional

from . import extractor, progress, vlm_limiter

logger = logging.getLogger("insurance_chunker.extractor")  # 운영 로그 쿼리가 이 이름을 본다

_SKIPPED = object()  # 차단기가 열려 호출하지 않은 페이지


class VlmStage:
    def __init__(self, concurrency: Optional[int] = None, budget: Optional[int] = None,
                 fail_streak: Optional[int] = None) -> None:
        self.concurrency = max(1, concurrency or extractor.VLM_CONCURRENCY)
        self.budget = extractor.VISION_MAX_PAGES if budget is None else budget
        self.fail_streak = fail_streak or extractor.VLM_FAIL_STREAK
        self.calls = 0           # 실제로 시도한 페이지 수(렌더 실패 포함 — 예전 계약과 같다)
        self.streak = 0          # 지금까지의 연속 실패 수
        self.circuit_open = False
        self.stats: list[dict] = []
        self.ocr_texts: dict[int, str] = {}  # OCR 대상({"page","ocr"})의 복구 본문
        self.total_targets = 0               # 받은 대상 수(상한 밖 포함) — 흘려 받으면 끝나야 안다
        self._spans: list[tuple[float, float]] = []  # 호출마다 (시작, 끝) — busy_seconds 계산용
        self._lock = threading.Lock()
        # 자동 조정이면 작업 스레드(= 상한 concurrency개)가 호출 직전에 이 안으로 들어간다.
        # None이면 고정 동시도 — 스레드 수가 곧 동시도다.
        self._limit = vlm_limiter.get()
        self.limit_range: Optional[list[int]] = None  # 이 문서 동안 실제로 적용된 상한 [최소, 최대]

    # ── 작업 스레드 ──────────────────────────────────────────────────────────

    def _record(self, stat: dict) -> None:
        with self._lock:
            self.stats.append(stat)
        logger.info(f"p{stat['page']}: VLM {stat['outcome']} {stat['seconds']}s",
                    extra={"event": "vlm_page", **stat})

    def _task(self, i: int, total: int, pno: int,
              rendered: Optional[tuple[str, int]], task: str = "table") -> object:
        # 자동 조정이면 상한 자리를 얻은 뒤에 차단기를 본다 — 기다리는 사이 차단기가 열렸으면
        # 부르지 않는다. 그래야 차단 뒤에 더 끝나는 호출이 '그 순간 날아가 있던 것'으로 묶인다.
        if self._limit is not None:
            self._limit.acquire()
        try:
            return self._call(i, total, pno, rendered, task)
        finally:
            if self._limit is not None:
                self._limit.release()

    def _call(self, i: int, total: int, pno: int,
              rendered: Optional[tuple[str, int]], task: str) -> object:
        with self._lock:
            if self.circuit_open:
                return _SKIPPED
            self.calls += 1
            if self._limit is not None:
                lim = self._limit.limit
                lo, hi = self.limit_range or (lim, lim)
                self.limit_range = [min(lo, lim), max(hi, lim)]
        logger.info(f"VLM [{i}/{total}] p{pno} 처리 중...")
        if rendered is None:  # 렌더 실패는 서버 장애가 아니다 — 연속 실패를 끊는다
            with self._lock:
                self.streak = 0
            return None
        img_b64, image_px = rendered
        t0 = time.perf_counter()
        try:
            md = extractor.call_vlm(img_b64, pno, image_px, record=self._record, task=task)
        except extractor._VLMCallFailed:
            with self._lock:
                self._spans.append((t0, time.perf_counter()))
            with self._lock:
                self.streak += 1
                opened = not self.circuit_open and self.streak >= self.fail_streak
                if opened:
                    self.circuit_open = True
            if opened:
                # 조용히 접으면 "표가 원래 없는 문서"와 구분이 안 된다. 반드시 소리를 낸다.
                logger.error(
                    f"VLM이 연속 {self.fail_streak}회 실패했다 — 이 문서의 남은 페이지는 "
                    f"VLM을 건너뛴다 (p{pno}에서 차단)",
                    extra={"event": "vlm_circuit_open", "page": pno, "streak": self.streak})
            return None
        # 호출이 성공했으면 표가 있든 없든 서버는 살아 있다 — 연속 카운터를 푼다.
        with self._lock:
            self.streak = 0
            self._spans.append((t0, time.perf_counter()))
        if md:
            logger.info(f"VLM [{i}/{total}] p{pno} 완료 (표 추출됨)")
        return md

    # ── 호출한 스레드 ────────────────────────────────────────────────────────

    def run(self, doc, targets: Iterable, total: Optional[int] = None,
            window: Optional[int] = None) -> dict[int, str]:
        """targets(순서 = 우선순위)를 호출해 {page: markdown}을 돌려준다. doc은 열린 fitz 문서.

        target은 쪽 번호(페이지 전체 입력), {"page","bbox"}(레이아웃 crop),
        {"page","ocr":True}(본문 복구 — 쪽 전체를 OCR 프롬프트로) 중 하나다. crop은
        한 쪽에 여럿일 수 있으므로 결과는 쪽마다 박스 순서대로 빈 줄로 잇는다 — 완료
        순서와 무관하게 대상 목록의 순서가 곧 출력 순서다. OCR 결과는 표가 아니라
        본문이므로 반환값에 섞지 않고 self.ocr_texts에 쌓는다.

        targets는 목록이 아니어도 된다 — 레이아웃이 쪽마다 박스를 찾는 대로 내놓는
        제너레이터면 그 쪽의 crop이 바로 호출된다(레이아웃·VLM 겹치기, stages의 겹치기 경로).
        이때 제너레이터는 이 메서드가 도는 스레드(문서를 연 스레드)에서 소비된다. 상한을
        넘은 대상과 차단기가 열린 뒤의 대상도 끝까지 소비한다 — 레이아웃 결과가 뒤 단계에
        필요해서다. 호출은 하지 않는다. total은 로그용이고 모르면 비워 둔다.

        window는 렌더해 두고 기다리게 하는 호출 수의 상한이다(메모리). 기본은 동시도 x 2.
        흘려 받을 때는 크게 준다 — 작으면 표 없는 쪽이 이어지는 동안 VLM이 놀게 된다.
        """
        if total is None and isinstance(targets, (list, tuple)):
            total = len(targets)
        shown = total if total is not None else "?"
        window = window or self.concurrency * 2
        jobs: list[tuple[int, Optional[list], str]] = []
        results: dict[int, str] = {}  # 대상 인덱스 → markdown/본문
        pending: dict[Future, int] = {}
        consumed = 0

        def collect(done) -> None:
            for fut in done:
                idx = pending.pop(fut)
                if fut.cancelled():
                    continue
                md = fut.result()
                if md and md is not _SKIPPED:
                    results[idx] = md

        with ThreadPoolExecutor(max_workers=self.concurrency,
                                thread_name_prefix="vlm") as pool:
            for t in targets:
                consumed += 1
                if len(jobs) >= self.budget or self.circuit_open:
                    continue  # 호출은 안 하지만 대상(=레이아웃 진행)은 끝까지 받는다
                if isinstance(t, dict):
                    job = (t["page"], t.get("bbox"), "ocr" if t.get("ocr") else "table")
                else:
                    job = (t, None, "table")
                jobs.append(job)
                i = len(jobs) - 1
                pno, clip, task = job
                # 쪽 진행 보고는 여기(렌더하는 호출 스레드)에서 한다 — 미리 렌더해 두는 양이
                # window로 묶여 있어 실제 호출과 몇 쪽 이상 벌어지지 않는다. 상한·차단으로
                # 호출하지 않는 대상은 일이 아니므로 보고하지 않는다.
                progress.report("vlm", pno, doc.page_count)
                while len(pending) >= window:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done)
                rendered = (extractor.render_page(doc[pno - 1], pno) if clip is None
                            else extractor.render_page(doc[pno - 1], pno, clip))
                pending[pool.submit(self._task, i + 1, shown, pno, rendered, task)] = i
            if self.circuit_open:
                for fut in pending:
                    fut.cancel()  # 아직 시작 안 한 것만 취소된다
            done, _ = wait(pending)
            collect(done)

        self.total_targets = consumed
        total = consumed if total is None else total
        self._log_stop(total)
        by_page: dict[int, list[str]] = {}
        for i, (pno, _clip, task) in enumerate(jobs):
            if i not in results:
                continue
            if task == "ocr":
                self.ocr_texts[pno] = results[i]
            else:
                by_page.setdefault(pno, []).append(results[i])
        return {p: "\n\n".join(parts) for p, parts in by_page.items()}

    def busy_seconds(self) -> float:
        """호출이 하나라도 진행 중이던 시간의 합(구간 합집합). GPU가 일한 시간의 근사다.

        순차 경로에선 VLM 단계 벽시계와 거의 같다. 겹치기 경로에선 단계 벽시계가 레이아웃과
        겹쳐 의미를 잃으므로, T_bound의 GPU 몫으로는 이 값을 쓴다.
        """
        total = 0.0
        end = None
        for a, b in sorted(self._spans):
            if end is None or a > end:
                total += b - a
                end = b
            elif b > end:
                total += b - end
                end = b
        return round(total, 2)

    def _log_stop(self, total: int) -> None:
        left = total - self.calls
        if left <= 0:
            return
        if self.circuit_open:
            # 차단 뒤에는 호출도 "처리 중" 로그도 하지 않는다. 남은 수를 한 줄로 남긴다.
            logger.warning(f"VLM 차단 — 남은 {left}건 건너뜀",
                           extra={"event": "vlm_pages_skipped", "skipped": left,
                                  "total": total})
        else:
            # 품질이 아니라 시간을 위해 접는 것이라 level이 다르다 —
            # 남은 대상도 PyMuPDF 결과는 그대로 쓴다.
            logger.info(f"VLM 상한 {self.budget}건 도달 — 남은 {left}건은 "
                        "PyMuPDF 결과만 쓴다",
                        extra={"event": "vlm_budget_exhausted", "limit": self.budget,
                               "skipped": left, "total": total})
