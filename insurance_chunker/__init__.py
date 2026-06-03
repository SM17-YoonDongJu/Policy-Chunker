"""insurance-chunker — 보험 약관 RAG 청킹 파이프라인.

Policy-Chunker 구조(폰트 기반 경계 감지 + best-of 표 선택)를
rag/ 파이프라인(풍부한 메타데이터 + 임베딩 + pgvector 저장)으로 보완.
"""
from .boundaries import Boundary, Detection, detect, find, assess, label_for
from .combine import select_best_tables
from .rechunk import clean, merge, finalize, report
from .chunker import chunk_document, auto_doc_type
from .models import DocMeta, InsuranceChunk, PageResult, compute_doc_hash

__version__ = "0.1.0"
__all__ = [
    "Boundary", "Detection", "detect", "find", "assess", "label_for",
    "select_best_tables",
    "clean", "merge", "finalize", "report",
    "chunk_document", "auto_doc_type",
    "DocMeta", "InsuranceChunk", "PageResult", "compute_doc_hash",
]
