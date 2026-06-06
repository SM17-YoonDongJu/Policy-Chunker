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
class DocMeta:
    source_pdf: str
    doc_hash: str
    doc_type: str          # product_summary | policy_terms | schedule | claim_form
    insurer: str
    product_name: str
    product_code: Optional[str] = None
    effective_date: Optional[str] = None  # YYYY-MM-DD
    generation: Optional[str] = None      # 세대 (예: "4세대")


@dataclass
class InsuranceChunk:
    chunk_id: str
    parent_id: Optional[str]           # 내부용 — DB 저장 안 함
    content: str
    content_tokens: str                 # Kiwi 형태소 결과 (공백 구분)
    structured_json: Optional[dict]     # 내부용 — DB 저장 안 함
    token_count: int
    section_path: list[str]             # 내부용
    section: str                        # DB 저장용: section_path 또는 경계 라벨 평탄화
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

    embedding: Optional[list[float]] = None


def make_chunk_id(source_pdf: str, page_num: int, idx: int) -> str:
    key = f"{source_pdf}:{page_num}:{idx}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def compute_doc_hash(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
