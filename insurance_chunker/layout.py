"""레이아웃 단계 — PP-DocLayout 계열(CPU)로 쪽마다 표 영역을 찾는다.

왜 레이아웃인가: 운영의 VLM_TARGET=gap 휴리스틱은 "PyMuPDF가 못 닿은 쪽"을 고르는데, 그게
고른 건 대부분 백지·목차·안내 쪽이었다(434쪽 문서에서 211쪽 — 진짜 표는 PyMuPDF가 잡은
쪽에 있었다). 페이지 전체를 VLM에 넣으면 백지에서 반복 루프가 돌고 본문이 표로 둔갑한다.
레이아웃이 잡은 표 영역만 잘라 넣으면 진짜 표만, 영역당 1~3초에 처리된다(2026-10-20
Colab T4 실험 — 진짜 표 9쪽 전부 구조 정확, 표 아닌 11쪽 전부 건너뜀).

PaddleOCR 공식 파이프라인(PaddleOCRVL)은 들이지 않는다 — 레이아웃 모듈(LayoutDetection)만
쓰고 인식은 기존 VLM 경로(vLLM의 PaddleOCR-VL)가 한다. 결과는 실험에서 공식 파이프라인과
동일했고, 공식 파이프라인은 vLLM 쪽 추가 플래그(trust-request-mm-kwargs)까지 요구한다.

chunker 파드에는 GPU가 없으므로 device="cpu"다. 모델은 프로세스당 한 번 lazy 로드한다 —
paddle 런타임+모델이 수백 MB라 임포트 시점에 열면 레이아웃을 안 쓰는 실행도 그 값을 낸다.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# "paddle"(기본) | "off". off면 기존 경로 그대로다 — 텍스트의 표 영역 제외는 find_tables,
# VLM 대상은 gap 휴리스틱, 입력은 페이지 전체. 롤백 스위치로 남긴다.
LAYOUT_BACKEND = os.environ.get("LAYOUT_BACKEND", "paddle")
LAYOUT_MODEL = os.environ.get("LAYOUT_MODEL", "PP-DocLayoutV3")
# V3 로드가 실패하면(구버전 paddleocr 등) 이 순서로 내려간다. 전부 실패하면 레이아웃을 끈다.
_MODEL_FALLBACKS = ("PP-DocLayoutV2", "PP-DocLayout_plus-L")
LAYOUT_MIN_SCORE = float(os.environ.get("LAYOUT_MIN_SCORE", "0.5"))
# 레이아웃 입력 렌더 해상도. 탐지는 박스만 내면 되므로 crop 렌더(VLM_DPI)보다 낮아도 된다.
LAYOUT_DPI = int(os.environ.get("LAYOUT_DPI", "150"))

_model = None
_load_failed = False
_lock = threading.Lock()


def available() -> bool:
    """레이아웃을 쓸 수 있는 상태인가. 로드를 시도하기 전에는 설정만 본다."""
    return LAYOUT_BACKEND != "off" and not _load_failed


def _load_model():
    """LayoutDetection을 프로세스당 한 번 로드. 실패하면 _load_failed를 세우고 None.

    실패를 조용히 삼키면 운영에서 의존이 빠진 이미지가 "표 없는 문서"처럼 보인다 —
    반드시 소리를 내고, 호출부(stages.layout)는 enabled=False로 기존 경로로 간다.
    """
    global _model, _load_failed
    with _lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from paddleocr import LayoutDetection
        except ImportError as e:
            _load_failed = True
            logger.error(
                f"paddleocr 미설치 — 레이아웃을 끄고 기존 경로(gap + 페이지 전체)로 간다: {e}",
                extra={"event": "layout_disabled", "reason": "import"})
            return None
        names = [LAYOUT_MODEL] + [m for m in _MODEL_FALLBACKS if m != LAYOUT_MODEL]
        for name in names:
            try:
                _model = LayoutDetection(model_name=name, device="cpu")
            except Exception as e:  # noqa: BLE001 - 모델 하나의 실패가 다음 후보를 막으면 안 된다
                logger.warning(f"레이아웃 모델 {name} 로드 실패: {e}")
                continue
            if name != LAYOUT_MODEL:
                logger.warning(f"레이아웃 모델 폴백: {LAYOUT_MODEL} → {name}",
                               extra={"event": "layout_model_fallback", "model": name})
            return _model
        _load_failed = True
        logger.error("레이아웃 모델을 하나도 못 열었다 — 기존 경로로 간다",
                     extra={"event": "layout_disabled", "reason": "load"})
        return None


# 이 비율 이상을 그림이 덮는데 텍스트 레이어가 없으면 스캔 쪽으로 본다(본문 복구 후보).
# 표지 로고처럼 작은 이미지만 있는 쪽을 스캔으로 오인하지 않기 위한 하한이다.
_OCR_IMG_FRAC = 0.5


def needs_ocr(fitz_page) -> bool:
    """텍스트 레이어가 없는데 그림이 쪽을 덮는다 — 본문 복구(OCR) 후보.

    이런 쪽은 parse_text가 빈 문자열을 내고, 지금까지는 표만 묻는 VLM이 받아 본문이
    조용히 유실됐다(적재는 OK로 집계된다, 이슈 #74). 커버 이미지 없이 그냥 빈 쪽은
    복구할 것도 없으므로 해당하지 않는다.
    """
    if fitz_page.get_text("words"):
        return False
    area = abs(fitz_page.rect.width * fitz_page.rect.height) or 1.0
    img = 0.0
    for blk in fitz_page.get_text("dict")["blocks"]:
        if blk.get("type") == 1:
            x0, y0, x1, y1 = blk["bbox"]
            img += abs((x1 - x0) * (y1 - y0)) / area
    return img >= _OCR_IMG_FRAC


def skip_blank(fitz_page) -> bool:
    """텍스트도 이미지도 괘선도 없는 쪽(백지)은 레이아웃을 부를 필요가 없다.

    레이아웃은 쪽당 수 초(CPU)라 674쪽 문서에서 백지 선필터가 분 단위를 아낀다.
    싼 신호부터 본다 — 단어 목록, 이미지 목록, 괘선 순.
    """
    if fitz_page.get_text("words"):
        return False
    if fitz_page.get_images():
        return False
    return not fitz_page.get_drawings()


def page_table_boxes(fitz_page) -> list[dict[str, Any]]:
    """쪽 하나의 표 박스 목록(위→아래, 왼→오른쪽 순).

    각 박스: {"bbox_pdf": [x0,y0,x1,y1](PDF 포인트), "bbox_px": [...](렌더 픽셀),
    "score": float}. 좌표를 둘 다 두는 이유 — crop 렌더(clip)는 포인트를 쓰고,
    디버그 그림·report.html류는 픽셀을 쓴다.
    """
    model = _load_model()
    if model is None:
        return []
    import numpy as np

    pix = fitz_page.get_pixmap(dpi=LAYOUT_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n > 3:
        img = img[:, :, :3]
    img = img[:, :, ::-1].copy()  # PyMuPDF는 RGB, Paddle은 BGR을 받는다
    out = model.predict(img, batch_size=1)

    scale = 72.0 / LAYOUT_DPI  # 렌더 픽셀 → PDF 포인트
    boxes: list[dict[str, Any]] = []
    for res in out:
        data = res.json["res"] if hasattr(res, "json") else res
        for b in data.get("boxes", []):
            if b.get("label") != "table" or float(b.get("score", 0)) < LAYOUT_MIN_SCORE:
                continue
            px = [float(v) for v in b["coordinate"]]
            boxes.append({"bbox_px": px,
                          "bbox_pdf": [round(v * scale, 2) for v in px],
                          "score": round(float(b["score"]), 3)})
    boxes.sort(key=lambda b: (b["bbox_pdf"][1], b["bbox_pdf"][0]))
    return boxes


def detect_tables(
    pdf_path: str,
) -> tuple[dict[int, list[dict[str, Any]]], list[int], dict[str, Any]]:
    """문서 전체의 쪽별 표 박스 + 본문 복구(OCR) 대상 쪽. (boxes, ocr_pages, 지표) 반환.

    지표에 쪽당 초를 남긴다 — 레이아웃의 쪽당 CPU 시간이 이 전환의 미확인 설계 변수라서
    (Colab 표본은 24쪽뿐), 운영 분포를 보고 모델을 PP-DocLayout-S로 내릴지 정한다.
    """
    import pymupdf as fitz

    boxes_by_page: dict[int, list[dict[str, Any]]] = {}
    ocr_pages: list[int] = []
    secs: list[float] = []
    blank = 0
    with fitz.open(pdf_path) as doc:
        for pno in range(1, doc.page_count + 1):
            page = doc[pno - 1]
            if skip_blank(page):
                blank += 1
                continue
            if needs_ocr(page):
                ocr_pages.append(pno)
            t0 = time.perf_counter()
            boxes = page_table_boxes(page)
            secs.append(round(time.perf_counter() - t0, 3))
            if boxes:
                boxes_by_page[pno] = boxes
            if _load_failed:
                break  # 모델이 없다 — 남은 쪽을 돌아도 전부 빈손이다
    n_boxes = sum(len(v) for v in boxes_by_page.values())
    metrics = {
        "model": LAYOUT_MODEL, "min_score": LAYOUT_MIN_SCORE,
        "blank_skipped": blank, "table_pages": len(boxes_by_page), "boxes": n_boxes,
        "ocr_pages": len(ocr_pages),
        "page_sec_p50": sorted(secs)[len(secs) // 2] if secs else None,
        "page_sec_max": max(secs) if secs else None,
    }
    return boxes_by_page, ocr_pages, metrics


def reset_for_tests() -> None:
    """테스트 전용 — 로드 상태를 되돌린다(모듈 reload 없이 가짜 모델을 끼우기 위해)."""
    global _model, _load_failed
    with _lock:
        _model = None
        _load_failed = False
