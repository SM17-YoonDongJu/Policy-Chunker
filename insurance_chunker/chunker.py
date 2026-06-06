"""doc_type별 청킹 오케스트레이터.

policy_terms  : boundaries.py(폰트 기반) + rechunk.py(병합·메타) → InsuranceChunk
product_summary: 표 행 병합 + contextual prefix (rag/ 방식)
schedule       : 표 행 단위 청크
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from .models import DocMeta, InsuranceChunk, PageResult, make_chunk_id
from .tokenizer import tokenize_korean

logger = logging.getLogger(__name__)

def _tok(text: str) -> int:
    return len(text)

_TYPE_MAP: dict[str, str] = {
    "면책": "exclusion", "부지급": "exclusion", "지급하지 않": "exclusion",
    "보상하지 않": "exclusion",
    "알릴 의무": "duty", "고지의무": "duty", "통지 의무": "duty",
    "보험금 청구": "claim", "청구 절차": "claim",
    "해지": "termination", "효력상실": "termination", "소멸": "termination",
    "특약": "special_clause", "특별": "special_clause",
    "정의": "definition", "용어": "definition",
    "보장": "coverage", "지급": "coverage", "보험금": "coverage",
}

_DOC_TYPE_PATTERNS = [
    (r"요약서|상품안내|상품설명서", "product_summary"),
    (r"약관|보통약관|특별약관", "policy_terms"),
    (r"산출기준표|보험료표|지급률표|장해분류표", "schedule"),
]


def auto_doc_type(filename: str) -> str:
    for pattern, doc_type in _DOC_TYPE_PATTERNS:
        if re.search(pattern, filename):
            return doc_type
    return "product_summary"


def _classify(text: str) -> str:
    for kw, ct in _TYPE_MAP.items():
        if kw in text:
            return ct
    return "general"


def _prefix(meta: DocMeta, section: Optional[str] = None) -> str:
    parts = [meta.insurer, meta.product_name]
    if section:
        parts.append(section)
    return " | ".join(parts)


def _make(
    content: str,
    page_num: int,
    idx: int,
    section_path: list[str],
    chunk_type: str,
    meta: DocMeta,
    article_number: Optional[str] = None,
    article_title: Optional[str] = None,
    structured_json=None,
) -> InsuranceChunk:
    return InsuranceChunk(
        chunk_id=make_chunk_id(meta.source_pdf, page_num, idx),
        parent_id=None,
        content=content,
        content_tokens=tokenize_korean(content),
        structured_json=structured_json,
        token_count=_tok(content),
        section_path=section_path,
        section=" > ".join(section_path) if section_path else "",
        page_number=page_num,
        doc_type=meta.doc_type,
        chunk_type=chunk_type,
        source_pdf=meta.source_pdf,
        doc_hash=meta.doc_hash,
        insurer=meta.insurer,
        product_name=meta.product_name,
        product_code=meta.product_code,
        effective_date=meta.effective_date,
        article_number=article_number,
        article_title=article_title,
        generation=meta.generation,
    )


# ── 공개 API ──────────────────────────────────────────────────────────────────

def chunk_document(
    pages: list[PageResult],
    meta: DocMeta,
    pdf_path: Optional[str] = None,
    target: int = 500,
    hard_max: int = 1000,
) -> list[InsuranceChunk]:
    """PDF 페이지 목록 → InsuranceChunk 목록.

    Args:
        pdf_path: policy_terms에서 폰트 기반 경계 감지에 사용. None이면 건너뜀.
        target: 목표 토큰 수 (병합 기준)
        hard_max: 강제 분할 상한 토큰 수

    Note:
        product_summary(요약서)는 ingest 대상이 아님. ingest.py에서 진입 전 차단.
    """
    if meta.doc_type == "policy_terms":
        return _chunk_policy_terms(pages, meta, pdf_path, target, hard_max)
    elif meta.doc_type == "schedule":
        return _chunk_schedule(pages, meta)
    else:
        return _chunk_plain_text(pages, meta)


# ── policy_terms: boundaries + rechunk ────────────────────────────────────────

def _chunk_policy_terms(
    pages: list[PageResult],
    meta: DocMeta,
    pdf_path: Optional[str],
    target: int,
    hard_max: int,
) -> list[InsuranceChunk]:
    from .boundaries import find as find_bounds
    from .rechunk import clean, merge, finalize, report

    # 폰트 기반 경계 감지
    bounds = None
    if pdf_path:
        try:
            import fitz
            from .boundaries import assess
            with fitz.open(pdf_path) as doc:
                bounds, det = find_bounds(doc)
                level, reasons = assess(doc, det, bounds)
            logger.info(f"폰트 경계 감지: {len(bounds)}개 (본문폰트={det.body_size}, 제목폰트={det.title_size})")
            if level == "weak":
                for r in reasons:
                    logger.warning(f"[신뢰도 WEAK] {r}")
                logger.warning("경계 신뢰도가 낮습니다 — 청킹 결과를 확인하세요")
            else:
                logger.info(f"[신뢰도 OK] {reasons[0]}")
        except Exception as e:
            logger.warning(f"폰트 경계 감지 실패, 경계 없이 진행: {e}")

    # 페이지 텍스트 → base_chunks (rechunk 입력 형식)
    base_chunks = _pages_to_base_chunks(pages, meta)

    if not bounds:
        # 경계 없으면 단순 텍스트 청킹으로 폴백
        logger.warning("경계 없음 → 단순 페이지 청킹으로 폴백")
        return _chunk_plain_text(pages, meta)

    cleaned = clean(base_chunks, bounds)
    merged = merge(cleaned, meta, target=target, hard_max=hard_max)
    chunks = finalize(merged, meta)
    stats = report(chunks, bounds)
    logger.info(
        f"[policy_terms] {stats['n_chunks']}청크 | "
        f"tok_mean={stats['tok_mean']} | over_600={stats['over_600']} | "
        f"yakwan={stats['n_unique_yakwan']}종"
    )
    return chunks


def _pages_to_base_chunks(pages: list[PageResult], meta: DocMeta) -> list[dict]:
    """PageResult → rechunk이 소비할 base_chunk dict 목록.

    텍스트: 단락(빈 줄) 단위 분할, chunk_id에 #pN# 태그.
    표: is_table=True 청크로 별도 추가.
    """
    chunks = []
    source = meta.source_pdf

    for page in pages:
        pno = page.page_num

        # 텍스트 단락 분할
        paras = [p.strip() for p in re.split(r"\n{2,}", page.text) if p.strip()]
        if not paras and page.text.strip():
            paras = [page.text.strip()]

        for seq, para in enumerate(paras):
            chunks.append({
                "chunk_id": f"{source}#p{pno}#{seq:04d}",
                "source": source,
                "doc_type": meta.doc_type,
                "chunk_type": "general",
                "text": para,
                "is_table": False,
                "page": pno,
            })

        # 표 청크
        for ti, tbl in enumerate(page.tables):
            md = tbl.get("markdown", "")
            src = tbl.get("source", "unknown")
            if md.strip():
                chunks.append({
                    "chunk_id": f"{source}#p{pno}#t{ti:02d}",
                    "source": source,
                    "doc_type": meta.doc_type,
                    "chunk_type": "general",
                    "text": md,
                    "is_table": True,
                    "table_source": src,
                    "page": pno,
                })

    return chunks


# ── schedule ──────────────────────────────────────────────────────────────────

def _chunk_schedule(pages: list[PageResult], meta: DocMeta) -> list[InsuranceChunk]:
    chunks: list[InsuranceChunk] = []
    pfx = _prefix(meta)

    for page in pages:
        for tbl in page.tables:
            md = tbl.get("markdown", "")
            if not md.strip():
                continue
            content = f"{pfx}\n{md}"
            chunks.append(_make(
                content=content, page_num=page.page_num, idx=len(chunks),
                section_path=[meta.product_name], chunk_type="schedule",
                meta=meta, structured_json={"markdown": md},
            ))

    logger.info(f"[schedule] {len(chunks)}청크")
    return chunks


# ── plain text fallback ───────────────────────────────────────────────────────

def _chunk_plain_text(pages: list[PageResult], meta: DocMeta) -> list[InsuranceChunk]:
    chunks: list[InsuranceChunk] = []
    pfx = _prefix(meta)
    for page in pages:
        for para in _split_paragraphs(page.text, max_tok=450):
            if not para.strip() or _tok(para) < 10:
                continue
            chunks.append(_make(
                content=f"{pfx}\n{para}",
                page_num=page.page_num, idx=len(chunks),
                section_path=[meta.product_name], chunk_type=_classify(para), meta=meta,
            ))
    logger.info(f"[plain_text] {len(chunks)}청크")
    return chunks


def _split_paragraphs(text: str, max_tok: int = 450) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    result, cur, cur_tok = [], [], 0
    for para in paras:
        pt = _tok(para)
        if cur and cur_tok + pt > max_tok:
            result.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(para)
        cur_tok += pt
    if cur:
        result.append("\n".join(cur))
    return result
