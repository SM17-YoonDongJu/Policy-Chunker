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

동시도의 상한은 서버가 정한다. vLLM은 KV 캐시 여유만큼만 동시에 처리하고(운영 설정에서 6,144토큰
요청 기준 3.17배), 넘치면 대기시킨다. 대기 시간도 클라이언트 읽기 타임아웃(VLM_TIMEOUT)에 들어가므로
동시도를 서버 여유보다 높이면 빨라지지 않고 타임아웃만 는다.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Optional

from . import extractor

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
        self._lock = threading.Lock()

    # ── 작업 스레드 ──────────────────────────────────────────────────────────

    def _record(self, stat: dict) -> None:
        with self._lock:
            self.stats.append(stat)
        logger.info(f"p{stat['page']}: VLM {stat['outcome']} {stat['seconds']}s",
                    extra={"event": "vlm_page", **stat})

    def _task(self, i: int, total: int, pno: int,
              rendered: Optional[tuple[str, int]], task: str = "table") -> object:
        with self._lock:
            if self.circuit_open:
                return _SKIPPED
            self.calls += 1
        logger.info(f"VLM [{i}/{total}] p{pno} 처리 중...")
        if rendered is None:  # 렌더 실패는 서버 장애가 아니다 — 연속 실패를 끊는다
            with self._lock:
                self.streak = 0
            return None
        img_b64, image_px = rendered
        try:
            md = extractor.call_vlm(img_b64, pno, image_px, record=self._record, task=task)
        except extractor._VLMCallFailed:
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
        if md:
            logger.info(f"VLM [{i}/{total}] p{pno} 완료 (표 추출됨)")
        return md

    # ── 호출한 스레드 ────────────────────────────────────────────────────────

    def run(self, doc, targets: list) -> dict[int, str]:
        """targets(순서 = 우선순위)를 호출해 {page: markdown}을 돌려준다. doc은 열린 fitz 문서.

        target은 쪽 번호(페이지 전체 입력), {"page","bbox"}(레이아웃 crop),
        {"page","ocr":True}(본문 복구 — 쪽 전체를 OCR 프롬프트로) 중 하나다. crop은
        한 쪽에 여럿일 수 있으므로 결과는 쪽마다 박스 순서대로 빈 줄로 잇는다 — 완료
        순서와 무관하게 대상 목록의 순서가 곧 출력 순서다. OCR 결과는 표가 아니라
        본문이므로 반환값에 섞지 않고 self.ocr_texts에 쌓는다.
        """
        total = len(targets)
        jobs = []
        for t in targets[:self.budget]:
            if isinstance(t, dict):
                jobs.append((t["page"], t.get("bbox"), "ocr" if t.get("ocr") else "table"))
            else:
                jobs.append((t, None, "table"))
        results: dict[int, str] = {}  # 대상 인덱스 → markdown/본문
        pending: dict[Future, int] = {}

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
            for i, (pno, clip, task) in enumerate(jobs):
                if self.circuit_open:
                    break
                while len(pending) >= self.concurrency * 2:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done)
                rendered = (extractor.render_page(doc[pno - 1], pno) if clip is None
                            else extractor.render_page(doc[pno - 1], pno, clip))
                pending[pool.submit(self._task, i + 1, total, pno, rendered, task)] = i
            if self.circuit_open:
                for fut in pending:
                    fut.cancel()  # 아직 시작 안 한 것만 취소된다
            done, _ = wait(pending)
            collect(done)

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
