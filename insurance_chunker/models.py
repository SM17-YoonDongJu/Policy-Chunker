from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PageResult:
    page_num: int
    text: str
    tables: list[dict] = field(default_factory=list)
    is_ocr: bool = False


@dataclass
class TableMeta:
    """policy_tables 행 + S3 업로드 대상."""
    doc_hash: str
    source_pdf: str
    insurer: str
    product_name: str
    page_number: int
    extractor: str              # 'pymupdf' | 'pdfplumber' | 'camelot' | 'vlm'
    markdown: str               # S3 저장 원본
    effective_date: Optional[str] = None
    section: Optional[str] = None
    table_index: int = 0
    caption: Optional[str] = None
    row_count: Optional[int] = None
    col_count: Optional[int] = None
    table_id: Optional[str] = None  # gen_random_uuid() — DB 저장 후 채워짐


@dataclass
class DocMeta:
    source_pdf: str
    doc_hash: str
    doc_type: str          # product_summary | policy_terms | schedule | claim_form
    insurer: str
    product_name: str
    product_code: Optional[str] = None
    effective_date: Optional[str] = None  # YYYY-MM-DD
    generation: Optional[str] = None      # 세대 (예: "4세대")
    product_id: Optional[str] = None      # insurance_products.id (UUID)


@dataclass
class InsuranceChunk:
    chunk_id: str
    content: str
    content_tokens: str                 # Kiwi 형태소 결과 (공백 구분)
    token_count: int
    section: str                        # 경계 라벨 또는 편/장 경로
    page_number: int
    doc_type: str                       # 내부용 — DB 저장 안 함
    chunk_type: str                     # coverage | exclusion | definition | special_clause | duty | claim | termination | schedule | general

    source_pdf: str                     # 내부용 — DB 저장 안 함
    doc_hash: str
    insurer: str
    product_name: str
    product_code: Optional[str]
    effective_date: Optional[str]

    article_number: Optional[str] = None   # "제12조"
    article_title: Optional[str] = None    # "보험금을 지급하지 않는 사유"
    generation: Optional[str] = None       # 세대
    chunk_index: int = 0                   # 문서 전체 기준 순서 (조항 복원 시 ORDER BY)
    product_id: Optional[str] = None       # insurance_products.id (UUID)

    # 표 row 청크 전용 (텍스트 청크는 None)
    table_id: Optional[str] = None         # S3 key: policy-tables/{table_id}.md
    row_start: Optional[int] = None
    row_end: Optional[int] = None

    embedding: Optional[list[float]] = None


def make_chunk_id(source_pdf: str, page_num: int, idx: int, doc_hash: str = "") -> str:
    key = f"{doc_hash}:{source_pdf}:{page_num}:{idx}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def compute_doc_hash(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
