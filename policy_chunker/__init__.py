"""
Policy-Chunker — 한국 보험약관(약관) PDF를 RAG 임베딩용 청크로 자르는 파이프라인.

핵심 아이디어: 구조 경계(약관/별표)는 본문 텍스트에서 추정하지 않고
PDF 조판 신호(제목 폰트 크기)에서 직접 가져온다. 그 위에 조번호 재추출,
인용 가드, 호→항→조 병합, 헤더 prepend, 중복 제거를 얹는다.
"""
from .boundaries import Boundary, Detection, detect, find, label_for, assess
from .rechunk import clean, merge, finalize, report
from .combine import combine
from .extract import extract

__version__ = "0.1.0"
__all__ = [
    "Boundary", "Detection", "detect", "find", "label_for", "assess",
    "clean", "merge", "finalize", "report", "run",
    "combine", "extract",
]


def run(pdf_path: str, chunks: list[dict], target: int = 700, hard_max: int = 1400):
    """PDF(경계) + 결합 청크 → 최종 청크 + 리포트.

    Returns: (final_chunks, report_dict, detection)
    """
    import fitz
    doc = fitz.open(pdf_path)
    bounds, det = find(doc)
    source = chunks[0]["source"] if chunks else pdf_path
    cleaned = clean(chunks, bounds)
    merged = merge(cleaned, target=target, hard_max=hard_max)
    final = finalize(merged, source)
    return final, report(final, bounds), det
